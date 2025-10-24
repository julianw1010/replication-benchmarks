/*
 * Canneal Benchmark - NUMA-Optimized with First-Touch Allocation
 * 
 * Simulated annealing for chip design routing optimization
 * Scaled from Mitosis paper: 382GB -> 100GB
 * Parameters: 2130M elements, 61k x 5k grid
 *
 * NUMA Optimization Strategy:
 * - All large arrays use malloc + parallel first-touch initialization
 * - Thread-local data allocated inside parallel regions
 * - Static scheduling for initialization to maintain NUMA locality
 * - Designed for: numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

/* Benchmark parameters (from scaling calculation) */
#define DEFAULT_NUM_ELEMENTS  2130000000L
#define DEFAULT_X_DIM         61000
#define DEFAULT_Y_DIM         5000
#define DEFAULT_SWAPS_PER_TEMP 5000000L   /* 5M for fast execution */
#define DEFAULT_TEMP_STEPS    20          /* 20 steps */
#define MAX_NETS_PER_ELEMENT  4  /* Reduced to lower memory footprint */

/* Simulated annealing parameters */
#define INITIAL_TEMP 2000.0
#define MIN_TEMP     0.001
#define TEMP_DECAY   0.95

/* Data structures */
typedef struct {
    int x;
    int y;
    int net_id;
    uint8_t num_nets;
    int nets[MAX_NETS_PER_ELEMENT];
} element_t;

typedef struct {
    int num_elements;
    int *element_ids;
} net_t;

typedef struct {
    long num_elements;
    int x_dim;
    int y_dim;
    long num_nets;
    element_t *elements;
    net_t *nets;
    int *location_grid;
    long swaps_per_temp;
    int temp_steps;
} workload_t;

/* RNG state - thread-local */
typedef struct {
    uint64_t state;
} rng_t;

static inline uint64_t rng_next(rng_t *rng) {
    rng->state = rng->state * 6364136223846793005ULL + 1;
    return rng->state >> 32;
}

static inline int rng_range(rng_t *rng, int max) {
    return rng_next(rng) % max;
}

static inline double rng_double(rng_t *rng) {
    return (double)rng_next(rng) / (double)0xFFFFFFFF;
}

/* Calculate routing cost for a net */
static inline double calculate_net_cost(workload_t *work, int net_id) {
    net_t *net = &work->nets[net_id];
    if (net->num_elements < 2) return 0.0;
    
    int min_x = work->x_dim, max_x = 0;
    int min_y = work->y_dim, max_y = 0;
    
    for (int i = 0; i < net->num_elements; i++) {
        element_t *elem = &work->elements[net->element_ids[i]];
        if (elem->x < min_x) min_x = elem->x;
        if (elem->x > max_x) max_x = elem->x;
        if (elem->y < min_y) min_y = elem->y;
        if (elem->y > max_y) max_y = elem->y;
    }
    
    /* Half-perimeter wirelength */
    return (double)((max_x - min_x) + (max_y - min_y));
}

/* Calculate cost delta for swapping two elements */
static double calculate_swap_cost(workload_t *work, int elem1_id, int elem2_id) {
    element_t *e1 = &work->elements[elem1_id];
    element_t *e2 = &work->elements[elem2_id];
    
    double old_cost = 0.0;
    double new_cost = 0.0;
    
    /* Calculate cost of affected nets before swap */
    for (int i = 0; i < e1->num_nets; i++) {
        old_cost += calculate_net_cost(work, e1->nets[i]);
    }
    for (int i = 0; i < e2->num_nets; i++) {
        int net_id = e2->nets[i];
        int duplicate = 0;
        for (int j = 0; j < e1->num_nets; j++) {
            if (e1->nets[j] == net_id) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            old_cost += calculate_net_cost(work, net_id);
        }
    }
    
    /* Swap positions temporarily */
    int temp_x = e1->x, temp_y = e1->y;
    e1->x = e2->x; e1->y = e2->y;
    e2->x = temp_x; e2->y = temp_y;
    
    /* Calculate cost after swap */
    for (int i = 0; i < e1->num_nets; i++) {
        new_cost += calculate_net_cost(work, e1->nets[i]);
    }
    for (int i = 0; i < e2->num_nets; i++) {
        int net_id = e2->nets[i];
        int duplicate = 0;
        for (int j = 0; j < e1->num_nets; j++) {
            if (e1->nets[j] == net_id) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) {
            new_cost += calculate_net_cost(work, net_id);
        }
    }
    
    /* Swap back */
    e2->x = e1->x; e2->y = e1->y;
    e1->x = temp_x; e1->y = temp_y;
    
    return new_cost - old_cost;
}

/* Initialize workload with NUMA-aware first-touch allocation */
void init_workload(workload_t *work, long num_elements, int x_dim, int y_dim,
                   long swaps_per_temp, int temp_steps) {
    work->num_elements = num_elements;
    work->x_dim = x_dim;
    work->y_dim = y_dim;
    work->swaps_per_temp = swaps_per_temp;
    work->temp_steps = temp_steps;
    work->num_nets = num_elements / 5; /* ~20% of elements are unique nets */
    
    printf("Initializing workload:\n");
    printf("  Elements: %ld (%.2f GB)\n", num_elements, 
           (double)(num_elements * sizeof(element_t)) / (1024*1024*1024));
    printf("  Nets: %ld (%.2f GB)\n", work->num_nets,
           (double)(work->num_nets * sizeof(net_t)) / (1024*1024*1024));
    printf("  Grid: %d x %d (%.2f GB)\n", x_dim, y_dim,
           (double)(x_dim * y_dim * sizeof(int)) / (1024*1024*1024));
    
    /* CRITICAL: Use malloc (NOT calloc) for first-touch allocation */
    work->elements = (element_t*)malloc(num_elements * sizeof(element_t));
    work->nets = (net_t*)malloc(work->num_nets * sizeof(net_t));
    work->location_grid = (int*)malloc(x_dim * y_dim * sizeof(int));
    
    if (!work->elements || !work->nets || !work->location_grid) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(1);
    }
    
    printf("Memory allocated, performing first-touch initialization...\n");
    fflush(stdout);
    
    /* FIRST-TOUCH: Initialize elements array in parallel with static scheduling */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        
        /* Thread-local RNG - allocated inside parallel region for NUMA locality */
        rng_t rng;
        rng.state = 12345 + tid * 67890;
        
        #pragma omp for schedule(static)
        for (long i = 0; i < num_elements; i++) {
            work->elements[i].x = i % x_dim;
            work->elements[i].y = (i / x_dim) % y_dim;
            work->elements[i].net_id = i % work->num_nets;
            work->elements[i].num_nets = 1 + (rng_next(&rng) % (MAX_NETS_PER_ELEMENT - 1));
            
            /* Assign element to nets */
            for (int j = 0; j < work->elements[i].num_nets; j++) {
                work->elements[i].nets[j] = (i + j * 997) % work->num_nets;
            }
        }
    }
    
    /* FIRST-TOUCH: Initialize nets array in parallel with static scheduling */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < work->num_nets; i++) {
        work->nets[i].num_elements = 0;
        work->nets[i].element_ids = NULL;
    }
    
    /* Build net membership - memory-efficient parallel approach */
    printf("Building net membership...\n");
    fflush(stdout);
    
    /* Pass 1: Count elements per net in parallel with atomics */
    long *net_counts = (long*)malloc(work->num_nets * sizeof(long));
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < work->num_nets; i++) {
        net_counts[i] = 0;
    }
    
    printf("  Counting net memberships...\n");
    fflush(stdout);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < num_elements; i++) {
        for (int j = 0; j < work->elements[i].num_nets; j++) {
            int net_id = work->elements[i].nets[j];
            __sync_fetch_and_add(&net_counts[net_id], 1);
        }
    }
    
    /* Allocate net element arrays with first-touch */
    printf("  Allocating net arrays...\n");
    fflush(stdout);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < work->num_nets; i++) {
        if (net_counts[i] > 0) {
            work->nets[i].element_ids = (int*)malloc(net_counts[i] * sizeof(int));
            work->nets[i].num_elements = net_counts[i];
            /* First-touch initialization */
            for (long j = 0; j < net_counts[i]; j++) {
                work->nets[i].element_ids[j] = 0;
            }
        }
    }
    
    /* Pass 2: Fill net membership in parallel with atomics */
    long *net_indices = (long*)malloc(work->num_nets * sizeof(long));
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < work->num_nets; i++) {
        net_indices[i] = 0;
    }
    
    printf("  Filling net memberships...\n");
    fflush(stdout);
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < num_elements; i++) {
        for (int j = 0; j < work->elements[i].num_nets; j++) {
            int net_id = work->elements[i].nets[j];
            long idx = __sync_fetch_and_add(&net_indices[net_id], 1);
            work->nets[net_id].element_ids[idx] = i;
        }
    }
    
    free(net_counts);
    free(net_indices);
    
    /* FIRST-TOUCH: Initialize location grid in parallel with static scheduling */
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)x_dim * y_dim; i++) {
        work->location_grid[i] = i % num_elements;
    }
    
    printf("Initialization complete.\n\n");
    fflush(stdout);
}

/* Free workload */
void free_workload(workload_t *work) {
    for (long i = 0; i < work->num_nets; i++) {
        if (work->nets[i].element_ids) {
            free(work->nets[i].element_ids);
        }
    }
    free(work->elements);
    free(work->nets);
    free(work->location_grid);
}

/* Run simulated annealing */
void run_annealing(workload_t *work) {
    double temp = INITIAL_TEMP;
    int temp_step = 0;
    
    printf("Starting simulated annealing...\n");
    printf("  Temperature steps: %d\n", work->temp_steps);
    printf("  Swaps per temperature: %ld\n", work->swaps_per_temp);
    printf("  Initial temperature: %.2f\n", INITIAL_TEMP);
    printf("\n");
    fflush(stdout);
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    while (temp > MIN_TEMP && temp_step < work->temp_steps) {
        long accepted = 0;
        long rejected = 0;
        
        /* Parallel annealing with thread-local state */
        #pragma omp parallel reduction(+:accepted,rejected)
        {
            /* Thread-local RNG - allocated inside parallel region */
            rng_t rng;
            rng.state = 12345 + omp_get_thread_num() * 67890 + temp_step * 111;
            
            long local_accepted = 0;
            long local_rejected = 0;
            
            #pragma omp for schedule(dynamic, 10000)
            for (long swap = 0; swap < work->swaps_per_temp; swap++) {
                /* Select two random elements */
                int elem1_id = rng_range(&rng, work->num_elements);
                int elem2_id = rng_range(&rng, work->num_elements);
                
                if (elem1_id == elem2_id) {
                    local_rejected++;
                    continue;
                }
                
                /* Calculate cost delta */
                double delta_cost = calculate_swap_cost(work, elem1_id, elem2_id);
                
                /* Accept or reject swap */
                if (delta_cost < 0.0 || rng_double(&rng) < exp(-delta_cost / temp)) {
                    /* Accept swap - actually perform it */
                    element_t *e1 = &work->elements[elem1_id];
                    element_t *e2 = &work->elements[elem2_id];
                    
                    int temp_x = e1->x, temp_y = e1->y;
                    e1->x = e2->x; e1->y = e2->y;
                    e2->x = temp_x; e2->y = temp_y;
                    
                    local_accepted++;
                } else {
                    local_rejected++;
                }
            }
            
            accepted += local_accepted;
            rejected += local_rejected;
        }
        
        /* Update temperature */
        temp *= TEMP_DECAY;
        temp_step++;
        
        if (temp_step % 10 == 0 || temp_step == 1) {
            printf("Step %3d: T=%.4f, Accepted=%ld (%.1f%%), Rejected=%ld\n",
                   temp_step, temp, accepted,
                   100.0 * accepted / (accepted + rejected), rejected);
            fflush(stdout);
        }
    }
    
    gettimeofday(&end, NULL);
    double elapsed = (end.tv_sec - start.tv_sec) + 
                     (end.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("\nAnnealing complete.\n");
    printf("Total time: %.2f seconds\n", elapsed);
    printf("Throughput: %.2f M swaps/sec\n", 
           (work->swaps_per_temp * temp_step / 1000000.0) / elapsed);
}

int main(int argc, char **argv) {
    long num_elements = DEFAULT_NUM_ELEMENTS;
    int x_dim = DEFAULT_X_DIM;
    int y_dim = DEFAULT_Y_DIM;
    long swaps_per_temp = DEFAULT_SWAPS_PER_TEMP;
    int temp_steps = DEFAULT_TEMP_STEPS;
    
    if (argc >= 4) {
        num_elements = atol(argv[1]);
        x_dim = atoi(argv[2]);
        y_dim = atoi(argv[3]);
    }
    if (argc >= 5) {
        swaps_per_temp = atol(argv[4]);
    }
    if (argc >= 6) {
        temp_steps = atoi(argv[5]);
    }
    
    printf("======================================\n");
    printf("Canneal Benchmark (NUMA-Optimized)\n");
    printf("======================================\n");
    printf("Threads: %d\n", omp_get_max_threads());
    printf("\n");
    
    workload_t work;
    init_workload(&work, num_elements, x_dim, y_dim, swaps_per_temp, temp_steps);
    
    run_annealing(&work);
    
    free_workload(&work);
    
    printf("\nBenchmark complete.\n");
    return 0;
}

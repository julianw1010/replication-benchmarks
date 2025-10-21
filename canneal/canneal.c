// canneal_bench.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>

#define MAX_DEGREE 16
#define SWAP_ATTEMPTS_PER_TEMP 100000000

typedef struct {
    int x, y;
    int netlist_id;
    uint64_t pad[5];
} Element;

typedef struct {
    int *elements;
    int count;
    int capacity;
} Netlist;

typedef struct {
    Element *elements;
    Netlist *netlists;
    int *locations;
    int num_elements;
    int num_netlists;
    int max_x, max_y;
} Chip;

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline int calculate_wire_cost(Chip *chip, int net_id) {
    Netlist *net = &chip->netlists[net_id];
    if (net->count < 2) return 0;
    
    int min_x = chip->max_x, max_x = 0;
    int min_y = chip->max_y, max_y = 0;
    
    for (int i = 0; i < net->count; i++) {
        int elem_id = net->elements[i];
        int loc = chip->locations[elem_id];
        Element *e = &chip->elements[loc];
        if (e->x < min_x) min_x = e->x;
        if (e->x > max_x) max_x = e->x;
        if (e->y < min_y) min_y = e->y;
        if (e->y > max_y) max_y = e->y;
    }
    
    return (max_x - min_x) + (max_y - min_y);
}

void init_chip(Chip *chip, int max_x, int max_y, int num_elements, int num_netlists) {
    chip->max_x = max_x;
    chip->max_y = max_y;
    chip->num_elements = num_elements;
    chip->num_netlists = num_netlists;
    
    chip->elements = malloc(num_elements * sizeof(Element));
    chip->locations = malloc(num_elements * sizeof(int));
    chip->netlists = malloc(num_netlists * sizeof(Netlist));
    
    if (!chip->elements || !chip->locations || !chip->netlists) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_elements; i++) {
        chip->elements[i].x = i % max_x;
        chip->elements[i].y = (i / max_x) % max_y;
        chip->elements[i].netlist_id = i % num_netlists;
        chip->locations[i] = i;
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_netlists; i++) {
        chip->netlists[i].capacity = MAX_DEGREE;
        chip->netlists[i].count = 0;
        chip->netlists[i].elements = malloc(MAX_DEGREE * sizeof(int));
        if (!chip->netlists[i].elements) {
            fprintf(stderr, "Netlist allocation failed\n");
            exit(1);
        }
    }
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_elements; i++) {
        int net_id = chip->elements[i].netlist_id;
        int pos;
        #pragma omp atomic capture
        pos = chip->netlists[net_id].count++;
        
        if (pos < MAX_DEGREE) {
            chip->netlists[net_id].elements[pos] = i;
        }
    }
}

int64_t anneal(Chip *chip, int iterations) {
    int64_t total_cost = 0;
    int nthreads = omp_get_max_threads();
    
    printf("Running with %d threads\n", nthreads);
    
    #pragma omp parallel reduction(+:total_cost)
    {
        uint64_t seed = (uint64_t)time(NULL) ^ ((uint64_t)omp_get_thread_num() << 32) ^ ((uint64_t)getpid() << 16);
        int tid = omp_get_thread_num();
        int64_t local_cost = 0;
        
        #pragma omp for schedule(dynamic, 1024)
        for (int iter = 0; iter < iterations; iter++) {
            int a = xorshift64(&seed) % chip->num_elements;
            int b = xorshift64(&seed) % chip->num_elements;
            
            if (a == b) continue;
            
            int loc_a = chip->locations[a];
            int loc_b = chip->locations[b];
            
            int net_a = chip->elements[loc_a].netlist_id;
            int net_b = chip->elements[loc_b].netlist_id;
            
            int old_cost = calculate_wire_cost(chip, net_a) + calculate_wire_cost(chip, net_b);
            
            chip->locations[a] = loc_b;
            chip->locations[b] = loc_a;
            
            int new_cost = calculate_wire_cost(chip, net_a) + calculate_wire_cost(chip, net_b);
            
            if (new_cost >= old_cost) {
                chip->locations[a] = loc_a;
                chip->locations[b] = loc_b;
                local_cost += old_cost;
            } else {
                local_cost += new_cost;
            }
            
            if (iter % 10000000 == 0 && tid == 0) {
                printf("Progress: %d/%d iterations\n", iter, iterations);
            }
        }
        
        total_cost += local_cost;
    }
    
    return total_cost;
}

int main(int argc, char **argv) {
    int max_x = 120000;
    int max_y = 11000;
    int num_elements = 1400000000;
    int num_netlists = 84000000;
    int iterations = SWAP_ATTEMPTS_PER_TEMP;
    
    printf("Canneal Benchmark - Multi-socket (100GB target)\n");
    printf("Grid: %dx%d\n", max_x, max_y);
    printf("Elements: %d\n", num_elements);
    printf("Netlists: %d\n", num_netlists);
    printf("Estimated memory: ~%.2f GB\n", 
           (num_elements * sizeof(Element) + 
            num_elements * sizeof(int) +
            num_netlists * (sizeof(Netlist) + MAX_DEGREE * sizeof(int))) / 1e9);
    
    Chip chip;
    printf("Initializing chip...\n");
    double init_start = omp_get_wtime();
    init_chip(&chip, max_x, max_y, num_elements, num_netlists);
    double init_end = omp_get_wtime();
    printf("Init time: %.2f seconds\n", init_end - init_start);
    
    printf("Starting annealing...\n");
    double start = omp_get_wtime();
    int64_t cost = anneal(&chip, iterations);
    double end = omp_get_wtime();
    
    printf("Final cost: %ld\n", cost);
    printf("Time: %.2f seconds\n", end - start);
    printf("Throughput: %.2f Mswaps/s\n", iterations / (end - start) / 1e6);
    
    free(chip.elements);
    free(chip.locations);
    for (int i = 0; i < chip.num_netlists; i++) {
        free(chip.netlists[i].elements);
    }
    free(chip.netlists);
    
    return 0;
}

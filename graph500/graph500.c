#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define SCALE 28
#define EDGEFACTOR 16

typedef int64_t vertex_t;
typedef struct {
    vertex_t src;
    vertex_t dst;
} edge_t;

typedef struct {
    vertex_t *offsets;
    vertex_t *neighbors;
    vertex_t num_vertices;
    int64_t num_edges;
} csr_graph_t;

static uint64_t rng_seed = 12345;

static inline uint64_t xorshift64star(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static inline double rand_double(uint64_t *seed) {
    return (xorshift64star(seed) >> 11) * (1.0 / 9007199254740992.0);
}

void generate_rmat_edges(edge_t *edges, int64_t num_edges, int scale) {
    const double A = 0.57, B = 0.19, C = 0.19;
    
    #pragma omp parallel
    {
        uint64_t local_seed = rng_seed + omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_edges; i++) {
            vertex_t u = 0, v = 0;
            
            for (int depth = 0; depth < scale; depth++) {
                double r = rand_double(&local_seed);
                int u_bit, v_bit;
                
                if (r < A) {
                    u_bit = 0; v_bit = 0;
                } else if (r < A + B) {
                    u_bit = 0; v_bit = 1;
                } else if (r < A + B + C) {
                    u_bit = 1; v_bit = 0;
                } else {
                    u_bit = 1; v_bit = 1;
                }
                
                u = (u << 1) | u_bit;
                v = (v << 1) | v_bit;
            }
            
            edges[i].src = u;
            edges[i].dst = v;
            
            if (rand_double(&local_seed) < 0.5) {
                vertex_t tmp = edges[i].src;
                edges[i].src = edges[i].dst;
                edges[i].dst = tmp;
            }
        }
    }
}

#define RADIX 256

void parallel_radix_sort_pass(edge_t *src, edge_t *dst, int64_t n, int shift, int is_src) {
    int num_threads = omp_get_max_threads();
    
    int64_t (*local_counts)[RADIX] = malloc(num_threads * sizeof(int64_t[RADIX]));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        memset(local_counts[tid], 0, RADIX * sizeof(int64_t));
        
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            uint64_t key = is_src ? (uint64_t)src[i].src : (uint64_t)src[i].dst;
            int bucket = (key >> shift) & 0xFF;
            local_counts[tid][bucket]++;
        }
    }
    
    int64_t global_counts[RADIX];
    memset(global_counts, 0, RADIX * sizeof(int64_t));
    
    #pragma omp parallel for
    for (int b = 0; b < RADIX; b++) {
        for (int t = 0; t < num_threads; t++) {
            global_counts[b] += local_counts[t][b];
        }
    }
    
    int64_t offsets[RADIX];
    offsets[0] = 0;
    for (int i = 1; i < RADIX; i++) {
        offsets[i] = offsets[i-1] + global_counts[i-1];
    }
    
    int64_t (*thread_offsets)[RADIX] = malloc(num_threads * sizeof(int64_t[RADIX]));
    
    #pragma omp parallel for
    for (int b = 0; b < RADIX; b++) {
        thread_offsets[0][b] = offsets[b];
        for (int t = 1; t < num_threads; t++) {
            thread_offsets[t][b] = thread_offsets[t-1][b] + local_counts[t-1][b];
        }
    }
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            uint64_t key = is_src ? (uint64_t)src[i].src : (uint64_t)src[i].dst;
            int bucket = (key >> shift) & 0xFF;
            int64_t pos = thread_offsets[tid][bucket]++;
            dst[pos] = src[i];
        }
    }
    
    free(local_counts);
    free(thread_offsets);
}

void parallel_radix_sort_edges(edge_t *edges, edge_t *temp, int64_t n) {
    edge_t *current = edges;
    edge_t *buffer = temp;
    
    // Sort by src (8 passes for 64-bit)
    for (int pass = 0; pass < 8; pass++) {
        parallel_radix_sort_pass(current, buffer, n, pass * 8, 1);
        edge_t *swap = current;
        current = buffer;
        buffer = swap;
    }
    
    // Sort by dst within each src group
    for (int pass = 0; pass < 8; pass++) {
        parallel_radix_sort_pass(current, buffer, n, pass * 8, 0);
        edge_t *swap = current;
        current = buffer;
        buffer = swap;
    }
    
    // If final result is in temp, copy back to edges
    if (current != edges) {
        #pragma omp parallel for
        for (int64_t i = 0; i < n; i++) {
            edges[i] = temp[i];
        }
    }
}

void remove_duplicates(edge_t *edges, int64_t *num_edges_ptr) {
    int64_t num_edges = *num_edges_ptr;
    if (num_edges == 0) return;
    
    int num_threads = omp_get_max_threads();
    int64_t chunk_size = (num_edges + num_threads - 1) / num_threads;
    
    char *keep = malloc(num_edges * sizeof(char));
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int64_t start = tid * chunk_size;
        int64_t end = (start + chunk_size < num_edges) ? start + chunk_size : num_edges;
        
        if (start < end) {
            keep[start] = 1;
            for (int64_t i = start + 1; i < end; i++) {
                keep[i] = (edges[i].src != edges[i-1].src || 
                          edges[i].dst != edges[i-1].dst) ? 1 : 0;
            }
        }
    }
    
    // Fix boundaries between chunks
    for (int t = 1; t < num_threads; t++) {
        int64_t boundary = t * chunk_size;
        if (boundary < num_edges && boundary > 0) {
            if (edges[boundary].src == edges[boundary-1].src &&
                edges[boundary].dst == edges[boundary-1].dst) {
                keep[boundary] = 0;
            }
        }
    }
    
    // Parallel prefix sum to compute new positions
    int64_t *prefix_sum = malloc((num_edges + 1) * sizeof(int64_t));
    prefix_sum[0] = 0;
    
    int64_t block_size = 1000000;
    int64_t num_blocks = (num_edges + block_size - 1) / block_size;
    int64_t *block_sums = malloc(num_blocks * sizeof(int64_t));
    
    #pragma omp parallel for
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = ((b + 1) * block_size < num_edges) ? (b + 1) * block_size : num_edges;
        int64_t sum = 0;
        for (int64_t i = start; i < end; i++) {
            sum += keep[i];
            prefix_sum[i + 1] = sum;
        }
        block_sums[b] = sum;
    }
    
    // Sequential scan of block sums
    int64_t total = 0;
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t tmp = block_sums[b];
        block_sums[b] = total;
        total += tmp;
    }
    
    // Add block offsets to local prefix sums
    #pragma omp parallel for
    for (int64_t b = 0; b < num_blocks; b++) {
        int64_t start = b * block_size;
        int64_t end = ((b + 1) * block_size < num_edges) ? (b + 1) * block_size : num_edges;
        int64_t offset = block_sums[b];
        for (int64_t i = start; i < end; i++) {
            prefix_sum[i + 1] += offset;
        }
    }
    
    // Compact array in parallel
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        if (keep[i]) {
            edges[prefix_sum[i]] = edges[i];
        }
    }
    
    *num_edges_ptr = total;
    
    free(keep);
    free(prefix_sum);
    free(block_sums);
}

csr_graph_t *build_csr(edge_t *edges, int64_t num_edges, vertex_t num_vertices) {
    csr_graph_t *g = malloc(sizeof(csr_graph_t));
    g->num_vertices = num_vertices;
    g->num_edges = num_edges;
    g->offsets = calloc(num_vertices + 1, sizeof(vertex_t));
    g->neighbors = malloc(num_edges * sizeof(vertex_t));
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        __sync_fetch_and_add(&g->offsets[edges[i].src + 1], 1);
    }
    
    for (vertex_t i = 1; i <= num_vertices; i++) {
        g->offsets[i] += g->offsets[i - 1];
    }
    
    vertex_t *temp_offsets = malloc((num_vertices + 1) * sizeof(vertex_t));
    #pragma omp parallel for
    for (vertex_t i = 0; i <= num_vertices; i++) {
        temp_offsets[i] = g->offsets[i];
    }
    
    #pragma omp parallel
    {
        int num_threads = omp_get_max_threads();
        int tid = omp_get_thread_num();
        int64_t chunk_size = (num_edges + num_threads - 1) / num_threads;
        int64_t start = tid * chunk_size;
        int64_t end = (start + chunk_size < num_edges) ? start + chunk_size : num_edges;
        
        for (int64_t i = start; i < end; i++) {
            vertex_t src = edges[i].src;
            vertex_t dst = edges[i].dst;
            int64_t pos = __sync_fetch_and_add(&temp_offsets[src], 1);
            g->neighbors[pos] = dst;
        }
    }
    
    free(temp_offsets);
    return g;
}

typedef struct {
    vertex_t *queue;
    int64_t head;
    int64_t tail;
} queue_t;

int64_t parallel_bfs(csr_graph_t *g, vertex_t root, vertex_t *parent) {
    vertex_t num_vertices = g->num_vertices;
    
    #pragma omp parallel for
    for (vertex_t i = 0; i < num_vertices; i++) {
        parent[i] = -1;
    }
    
    parent[root] = root;
    int64_t edges_traversed = 0;
    
    queue_t *local_queues = malloc(omp_get_max_threads() * sizeof(queue_t));
    for (int t = 0; t < omp_get_max_threads(); t++) {
        local_queues[t].queue = malloc(num_vertices * sizeof(vertex_t));
        local_queues[t].head = 0;
        local_queues[t].tail = 0;
    }
    
    vertex_t *current_frontier = malloc(num_vertices * sizeof(vertex_t));
    vertex_t *next_frontier = malloc(num_vertices * sizeof(vertex_t));
    int64_t current_size = 1;
    current_frontier[0] = root;
    
    while (current_size > 0) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (int t = 0; t < omp_get_max_threads(); t++) {
            local_queues[t].tail = 0;
        }
        
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:edges_traversed)
        for (int64_t i = 0; i < current_size; i++) {
            vertex_t u = current_frontier[i];
            int tid = omp_get_thread_num();
            
            vertex_t start = g->offsets[u];
            vertex_t end = g->offsets[u + 1];
            edges_traversed += (end - start);
            
            for (vertex_t j = start; j < end; j++) {
                vertex_t v = g->neighbors[j];
                if (parent[v] == -1) {
                    if (__sync_bool_compare_and_swap(&parent[v], -1, u)) {
                        local_queues[tid].queue[local_queues[tid].tail++] = v;
                    }
                }
            }
        }
        
        int64_t next_size = 0;
        for (int t = 0; t < omp_get_max_threads(); t++) {
            memcpy(&next_frontier[next_size], local_queues[t].queue, 
                   local_queues[t].tail * sizeof(vertex_t));
            next_size += local_queues[t].tail;
        }
        
        vertex_t *tmp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = tmp;
        current_size = next_size;
    }
    
    for (int t = 0; t < omp_get_max_threads(); t++) {
        free(local_queues[t].queue);
    }
    free(local_queues);
    free(current_frontier);
    free(next_frontier);
    
    return edges_traversed;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
    printf("Graph500 Benchmark (Scaled)\n");
    printf("Scale: %d, Edge Factor: %d\n", SCALE, EDGEFACTOR);
    
    vertex_t num_vertices = (vertex_t)1 << SCALE;
    int64_t num_edges = (int64_t)EDGEFACTOR * num_vertices;
    
    printf("Target vertices: %ld\n", (long)num_vertices);
    printf("Target edges: %ld\n", (long)num_edges);
    
    double t_start = get_time();
    
    printf("\n[1/4] Generating R-MAT edges...\n");
    edge_t *edges = malloc(num_edges * sizeof(edge_t));
    if (!edges) {
        fprintf(stderr, "Failed to allocate edge array\n");
        return 1;
    }
    
    double t_gen_start = get_time();
    generate_rmat_edges(edges, num_edges, SCALE);
    double t_gen = get_time() - t_gen_start;
    printf("  Time: %.2f seconds\n", t_gen);
    
    printf("\n[2/4] Sorting edges...\n");
    double t_sort_start = get_time();
    edge_t *temp = malloc(num_edges * sizeof(edge_t));
    if (!temp) {
        fprintf(stderr, "Failed to allocate temporary sort buffer\n");
        return 1;
    }
    parallel_radix_sort_edges(edges, temp, num_edges);
    free(temp);
    double t_sort = get_time() - t_sort_start;
    printf("  Time: %.2f seconds\n", t_sort);
    
    printf("\n[3/4] Removing duplicate edges...\n");
    int64_t orig_edges = num_edges;
    remove_duplicates(edges, &num_edges);
    printf("  Unique edges: %ld (%.1f%% of original)\n", 
           (long)num_edges, 100.0 * num_edges / orig_edges);
    
    printf("\n[4/4] Building CSR graph...\n");
    double t_csr_start = get_time();
    csr_graph_t *graph = build_csr(edges, num_edges, num_vertices);
    double t_csr = get_time() - t_csr_start;
    printf("  Time: %.2f seconds\n", t_csr);
    
    free(edges);
    
    double memory_gb = (num_edges * sizeof(edge_t) + 
                        num_vertices * sizeof(vertex_t) * 2 + 
                        num_edges * sizeof(vertex_t)) / (1024.0 * 1024.0 * 1024.0);
    printf("\nEstimated memory usage: %.2f GB\n", memory_gb);
    
    printf("\n=== Running BFS Traversals ===\n");
    vertex_t *parent = malloc(num_vertices * sizeof(vertex_t));
    
    int num_bfs_roots = 64;
    uint64_t seed = 9876;
    double total_bfs_time = 0.0;
    int64_t total_teps = 0;
    
    for (int i = 0; i < num_bfs_roots; i++) {
        vertex_t root = (vertex_t)(rand_double(&seed) * num_vertices);
        while (graph->offsets[root + 1] == graph->offsets[root]) {
            root = (vertex_t)(rand_double(&seed) * num_vertices);
        }
        
        double t_bfs_start = get_time();
        int64_t edges_traversed = parallel_bfs(graph, root, parent);
        double t_bfs = get_time() - t_bfs_start;
        
        double teps = edges_traversed / t_bfs / 1e9;
        total_bfs_time += t_bfs;
        total_teps += edges_traversed;
        
        printf("  BFS %2d: Root=%10ld, Time=%.4f s, TEPS=%.4f GTEPS\n", 
               i + 1, (long)root, t_bfs, teps);
    }
    
    double avg_bfs_time = total_bfs_time / num_bfs_roots;
    double harmonic_mean_teps = num_bfs_roots / 
        (total_bfs_time / (total_teps / 1e9));
    
    printf("\n=== Summary ===\n");
    printf("Graph construction time: %.2f seconds\n", 
           get_time() - t_start - total_bfs_time);
    printf("Average BFS time: %.4f seconds\n", avg_bfs_time);
    printf("Harmonic mean TEPS: %.4f GTEPS\n", harmonic_mean_teps);
    
    free(parent);
    free(graph->offsets);
    free(graph->neighbors);
    free(graph);
    
    return 0;
}

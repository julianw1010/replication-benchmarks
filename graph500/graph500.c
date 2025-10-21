#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

#define SCALE 27  // ~100GB for edge list
#define EDGEFACTOR 16
#define NUM_BFS_ROOTS 4

typedef struct {
    int64_t *edges;
    int64_t *offsets;
    int64_t num_vertices;
    int64_t num_edges;
} graph_t;

static inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

void generate_edges(int64_t **edges_out, int64_t *num_edges_out) {
    int64_t num_vertices = 1L << SCALE;
    int64_t num_edges = num_vertices * EDGEFACTOR;
    
    printf("Allocating %.2f GB for edge list...\n", (num_edges * 2 * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0));
    
    int64_t *edges = (int64_t *)malloc(num_edges * 2 * sizeof(int64_t));
    if (!edges) {
        fprintf(stderr, "Failed to allocate edge list\n");
        exit(1);
    }
    
    double start = get_time();
    
    #pragma omp parallel
    {
        uint64_t seed = 0x123456789ABCDEF0ULL + omp_get_thread_num();
        
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < num_edges; i++) {
            int64_t u = xorshift64(&seed) % num_vertices;
            int64_t v = xorshift64(&seed) % num_vertices;
            edges[i * 2] = u;
            edges[i * 2 + 1] = v;
        }
    }
    
    printf("Edge generation: %.2f s\n", get_time() - start);
    
    *edges_out = edges;
    *num_edges_out = num_edges;
}

void build_csr(int64_t *edges, int64_t num_edges, graph_t *graph) {
    int64_t num_vertices = 1L << SCALE;
    
    printf("Building CSR (%.2f GB)...\n", 
           (num_vertices * sizeof(int64_t) + num_edges * 2 * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0));
    
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges * 2;
    graph->offsets = (int64_t *)calloc(num_vertices + 1, sizeof(int64_t));
    graph->edges = (int64_t *)malloc(num_edges * 2 * sizeof(int64_t));
    
    if (!graph->offsets || !graph->edges) {
        fprintf(stderr, "Failed to allocate CSR\n");
        exit(1);
    }
    
    double start = get_time();
    
    // Count degrees
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t u = edges[i * 2];
        int64_t v = edges[i * 2 + 1];
        #pragma omp atomic
        graph->offsets[u + 1]++;
        #pragma omp atomic
        graph->offsets[v + 1]++;
    }
    
    // Prefix sum
    for (int64_t i = 1; i <= num_vertices; i++) {
        graph->offsets[i] += graph->offsets[i - 1];
    }
    
    // Fill edges in parallel with shared temp offsets
    int64_t *temp_offsets = (int64_t *)malloc((num_vertices + 1) * sizeof(int64_t));
    memcpy(temp_offsets, graph->offsets, (num_vertices + 1) * sizeof(int64_t));
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t u = edges[i * 2];
        int64_t v = edges[i * 2 + 1];
        
        int64_t pos_u = __sync_fetch_and_add(&temp_offsets[u], 1);
        int64_t pos_v = __sync_fetch_and_add(&temp_offsets[v], 1);
        
        graph->edges[pos_u] = v;
        graph->edges[pos_v] = u;
    }
    
    free(temp_offsets);
    
    printf("CSR build: %.2f s\n", get_time() - start);
}

int64_t parallel_bfs(graph_t *graph, int64_t root) {
    int64_t num_vertices = graph->num_vertices;
    int64_t *visited = (int64_t *)malloc(num_vertices * sizeof(int64_t));
    
    #pragma omp parallel for
    for (int64_t i = 0; i < num_vertices; i++) {
        visited[i] = -1;
    }
    
    visited[root] = 0;
    int64_t *queue = (int64_t *)malloc(num_vertices * sizeof(int64_t));
    int64_t *next_queue = (int64_t *)malloc(num_vertices * sizeof(int64_t));
    
    queue[0] = root;
    int64_t queue_size = 1;
    int64_t level = 0;
    int64_t edges_traversed = 0;
    
    while (queue_size > 0) {
        int64_t next_size = 0;
        
        #pragma omp parallel
        {
            int64_t local_next[1024];
            int64_t local_count = 0;
            
            #pragma omp for schedule(dynamic, 1024) reduction(+:edges_traversed)
            for (int64_t i = 0; i < queue_size; i++) {
                int64_t u = queue[i];
                int64_t start = graph->offsets[u];
                int64_t end = graph->offsets[u + 1];
                
                for (int64_t j = start; j < end; j++) {
                    int64_t v = graph->edges[j];
                    edges_traversed++;
                    
                    if (__sync_bool_compare_and_swap(&visited[v], -1, level + 1)) {
                        if (local_count < 1024) {
                            local_next[local_count++] = v;
                        } else {
                            int64_t pos = __sync_fetch_and_add(&next_size, local_count);
                            memcpy(&next_queue[pos], local_next, local_count * sizeof(int64_t));
                            local_next[0] = v;
                            local_count = 1;
                        }
                    }
                }
            }
            
            if (local_count > 0) {
                int64_t pos = __sync_fetch_and_add(&next_size, local_count);
                memcpy(&next_queue[pos], local_next, local_count * sizeof(int64_t));
            }
        }
        
        int64_t *temp = queue;
        queue = next_queue;
        next_queue = temp;
        queue_size = next_size;
        level++;
    }
    
    free(visited);
    free(queue);
    free(next_queue);
    
    return edges_traversed;
}

int main() {
    printf("Graph500 Benchmark - SCALE=%d, EDGEFACTOR=%d\n", SCALE, EDGEFACTOR);
    printf("Target memory: ~100GB, Threads: %d\n\n", omp_get_max_threads());
    
    int64_t *edges;
    int64_t num_edges;
    
    generate_edges(&edges, &num_edges);
    
    graph_t graph;
    build_csr(edges, num_edges, &graph);
    free(edges);
    
    printf("\nRunning BFS traversals...\n");
    
    uint64_t seed = 0xDEADBEEF;
    double total_time = 0;
    double total_teps = 0;
    
    for (int i = 0; i < NUM_BFS_ROOTS; i++) {
        int64_t root = xorshift64(&seed) % graph.num_vertices;
        
        double start = get_time();
        int64_t edges_traversed = parallel_bfs(&graph, root);
        double elapsed = get_time() - start;
        
        double teps = edges_traversed / elapsed / 1e9;
        total_time += elapsed;
        total_teps += teps;
        
        printf("BFS %2d: root=%10ld, edges=%10ld, time=%.3f s, GTEPS=%.3f\n", 
               i, root, edges_traversed, elapsed, teps);
    }
    
    printf("\nAverage: %.3f s, %.3f GTEPS\n", total_time / NUM_BFS_ROOTS, total_teps / NUM_BFS_ROOTS);
    
    free(graph.offsets);
    free(graph.edges);
    
    return 0;
}

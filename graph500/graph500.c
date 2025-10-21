#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>

#define SCALE 27
#define EDGEFACTOR 23
#define NUM_BFS_ROOTS 64

// R-MAT parameters (Graph500 spec)
#define A 0.57
#define B 0.19
#define C 0.19
#define D 0.05

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

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline double rand_double(uint64_t *state) {
    return (splitmix64(state) >> 11) * 0x1.0p-53;
}

void generate_rmat_edges(int64_t **edges_out, int64_t *num_edges_out) {
    int64_t num_vertices = 1L << SCALE;
    int64_t num_edges = num_vertices * EDGEFACTOR;
    
    double mem_gb = (num_edges * 2 * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
    printf("Generating R-MAT graph: %.2f GB edge list\n", mem_gb);
    
    int64_t *edges = malloc(num_edges * 2 * sizeof(int64_t));
    if (!edges) {
        fprintf(stderr, "Edge allocation failed\n");
        exit(1);
    }
    
    double start = get_time();
    
    #pragma omp parallel
    {
        uint64_t seed = 0x2b992ddfa23249d6ULL ^ (omp_get_thread_num() * 0x9e3779b97f4a7c15ULL);
        
        #pragma omp for schedule(static)
        for (int64_t e = 0; e < num_edges; e++) {
            int64_t u = 0, v = 0;
            
            for (int depth = 0; depth < SCALE; depth++) {
                double r = rand_double(&seed);
                int64_t u_bit, v_bit;
                
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
            
            // Permute to break symmetry
            u = (u * 0x9e3779b97f4a7c15ULL) % num_vertices;
            v = (v * 0x94d049bb133111ebULL) % num_vertices;
            
            edges[e * 2] = u;
            edges[e * 2 + 1] = v;
        }
    }
    
    printf("R-MAT generation: %.2f s\n", get_time() - start);
    
    *edges_out = edges;
    *num_edges_out = num_edges;
}

void build_csr(int64_t *edges, int64_t num_edges, graph_t *graph) {
    int64_t num_vertices = 1L << SCALE;
    
    double mem_gb = (num_vertices * sizeof(int64_t) + num_edges * 2 * sizeof(int64_t)) / (1024.0 * 1024.0 * 1024.0);
    printf("Building CSR: %.2f GB\n", mem_gb);
    
    graph->num_vertices = num_vertices;
    graph->num_edges = num_edges * 2;
    graph->offsets = calloc(num_vertices + 1, sizeof(int64_t));
    graph->edges = malloc(num_edges * 2 * sizeof(int64_t));
    
    if (!graph->offsets || !graph->edges) {
        fprintf(stderr, "CSR allocation failed\n");
        exit(1);
    }
    
    double start = get_time();
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t u = edges[i * 2];
        int64_t v = edges[i * 2 + 1];
        __sync_fetch_and_add(&graph->offsets[u + 1], 1);
        __sync_fetch_and_add(&graph->offsets[v + 1], 1);
    }
    
    for (int64_t i = 1; i <= num_vertices; i++) {
        graph->offsets[i] += graph->offsets[i - 1];
    }
    
    int64_t *temp = malloc((num_vertices + 1) * sizeof(int64_t));
    memcpy(temp, graph->offsets, (num_vertices + 1) * sizeof(int64_t));
    
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < num_edges; i++) {
        int64_t u = edges[i * 2];
        int64_t v = edges[i * 2 + 1];
        
        int64_t pos_u = __sync_fetch_and_add(&temp[u], 1);
        int64_t pos_v = __sync_fetch_and_add(&temp[v], 1);
        
        graph->edges[pos_u] = v;
        graph->edges[pos_v] = u;
    }
    
    free(temp);
    printf("CSR build: %.2f s\n", get_time() - start);
}

int64_t parallel_bfs(graph_t *graph, int64_t root) {
    int64_t n = graph->num_vertices;
    int64_t *parent = malloc(n * sizeof(int64_t));
    
    #pragma omp parallel for
    for (int64_t i = 0; i < n; i++) {
        parent[i] = -1;
    }
    
    parent[root] = root;
    int64_t *frontier = malloc(n * sizeof(int64_t));
    int64_t *next = malloc(n * sizeof(int64_t));
    
    frontier[0] = root;
    int64_t fsize = 1;
    int64_t edges_visited = 0;
    
    while (fsize > 0) {
        int64_t nsize = 0;
        
        #pragma omp parallel reduction(+:edges_visited)
        {
            int64_t local_buf[1024];
            int64_t local_cnt = 0;
            
            #pragma omp for schedule(dynamic, 64)
            for (int64_t i = 0; i < fsize; i++) {
                int64_t u = frontier[i];
                int64_t start = graph->offsets[u];
                int64_t end = graph->offsets[u + 1];
                
                for (int64_t j = start; j < end; j++) {
                    int64_t v = graph->edges[j];
                    edges_visited++;
                    
                    if (__sync_bool_compare_and_swap(&parent[v], -1, u)) {
                        if (local_cnt < 1024) {
                            local_buf[local_cnt++] = v;
                        } else {
                            int64_t pos = __sync_fetch_and_add(&nsize, local_cnt);
                            memcpy(&next[pos], local_buf, local_cnt * sizeof(int64_t));
                            local_buf[0] = v;
                            local_cnt = 1;
                        }
                    }
                }
            }
            
            if (local_cnt > 0) {
                int64_t pos = __sync_fetch_and_add(&nsize, local_cnt);
                memcpy(&next[pos], local_buf, local_cnt * sizeof(int64_t));
            }
        }
        
        int64_t *tmp = frontier;
        frontier = next;
        next = tmp;
        fsize = nsize;
    }
    
    free(parent);
    free(frontier);
    free(next);
    
    return edges_visited;
}

int main() {
    printf("=== Graph500 Benchmark ===\n");
    printf("SCALE=%d, EDGEFACTOR=%d, Threads=%d\n\n", SCALE, EDGEFACTOR, omp_get_max_threads());
    
    int64_t *edges, num_edges;
    generate_rmat_edges(&edges, &num_edges);
    
    graph_t graph;
    build_csr(edges, num_edges, &graph);
    free(edges);
    
    printf("\nBFS Traversals:\n");
    
    uint64_t seed = 0x2b992ddfa23249d6ULL;
    double total_time = 0, total_teps = 0;
    
    for (int i = 0; i < NUM_BFS_ROOTS; i++) {
        int64_t root = splitmix64(&seed) % graph.num_vertices;
        
        double start = get_time();
        int64_t edges_visited = parallel_bfs(&graph, root);
        double elapsed = get_time() - start;
        double teps = edges_visited / elapsed / 1e9;
        
        total_time += elapsed;
        total_teps += teps;
        
        printf("BFS %2d: %.3f s, %.3f GTEPS\n", i, elapsed, teps);
    }
    
    printf("\n=== Results ===\n");
    printf("Mean: %.3f s, %.3f GTEPS\n", total_time / NUM_BFS_ROOTS, total_teps / NUM_BFS_ROOTS);
    printf("Harmonic mean TEPS: %.3f GTEPS\n", NUM_BFS_ROOTS / (total_time / (total_teps * NUM_BFS_ROOTS)));
    
    free(graph.offsets);
    free(graph.edges);
    
    return 0;
}

// xsbench_scaled.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
#include <stdint.h>

// Scaled from Mitosis Multi Socket: p_factor=25M, g_factor=920K, 440GB
// Target: 100GB, ratio maintained at ~27.2
#define P_FACTOR 12500000000UL  // 100GB memory target
#define G_FACTOR 460000          // Maintains ~27K ratio (closer to Mitosis 27.2)
#define LOOKUPS 6000000000UL     // Sufficient stress test iterations

typedef struct {
    double *data;
    uint64_t size;
} GridPoint;

typedef struct {
    GridPoint *grid;
    uint64_t n_grid;
    uint64_t n_points;
} Simulation;

uint64_t hash(uint64_t x) {
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = ((x >> 16) ^ x) * 0x45d9f3b;
    x = (x >> 16) ^ x;
    return x;
}

Simulation* init_simulation(uint64_t p_factor, uint64_t g_factor) {
    Simulation *sim = malloc(sizeof(Simulation));
    sim->n_grid = g_factor;
    sim->n_points = p_factor;
    
    printf("Allocating %lu grid points...\n", sim->n_grid);
    sim->grid = malloc(sim->n_grid * sizeof(GridPoint));
    
    uint64_t total_bytes = 0;
    #pragma omp parallel for reduction(+:total_bytes)
    for(uint64_t i = 0; i < sim->n_grid; i++) {
        sim->grid[i].size = (p_factor / g_factor) + (hash(i) % 100);
        sim->grid[i].data = malloc(sim->grid[i].size * sizeof(double));
        
        for(uint64_t j = 0; j < sim->grid[i].size; j++) {
            sim->grid[i].data[j] = (double)hash(i*p_factor + j) / (double)UINT64_MAX;
        }
        total_bytes += sim->grid[i].size * sizeof(double);
    }
    
    printf("Allocated %.2f GB\n", total_bytes / 1e9);
    return sim;
}

double run_lookups(Simulation *sim, uint64_t n_lookups) {
    double result = 0.0;
    
    #pragma omp parallel reduction(+:result)
    {
        uint64_t tid = omp_get_thread_num();
        uint64_t seed = hash(tid + 12345);
        double local_sum = 0.0;
        
        #pragma omp for schedule(dynamic, 1000)
        for(uint64_t i = 0; i < n_lookups; i++) {
            seed = hash(seed + i);
            uint64_t grid_idx = seed % sim->n_grid;
            uint64_t point_idx = hash(seed) % sim->grid[grid_idx].size;
            
            local_sum += sim->grid[grid_idx].data[point_idx];
        }
        result += local_sum;
    }
    
    return result;
}

void cleanup(Simulation *sim) {
    for(uint64_t i = 0; i < sim->n_grid; i++) {
        free(sim->grid[i].data);
    }
    free(sim->grid);
    free(sim);
}

int main(int argc, char *argv[]) {
    uint64_t p_factor = P_FACTOR;
    uint64_t g_factor = G_FACTOR;
    uint64_t lookups = LOOKUPS;
    
    if(argc > 1) p_factor = atol(argv[1]);
    if(argc > 2) g_factor = atol(argv[2]);
    if(argc > 3) lookups = atol(argv[3]);
    
    printf("XSBench Scaled - Multi Socket Configuration\n");
    printf("Threads: %d (max available)\n", omp_get_max_threads());
    printf("p_factor: %lu, g_factor: %lu (ratio: %.1f)\n", 
           p_factor, g_factor, (double)p_factor/g_factor);
    printf("Target: 100GB, Lookups: %lu\n\n", lookups);
    
    Simulation *sim = init_simulation(p_factor, g_factor);
    
    printf("\nStarting benchmark...\n");
    double start = omp_get_wtime();
    double result = run_lookups(sim, lookups);
    double end = omp_get_wtime();
    
    printf("\nRuntime: %.3f seconds\n", end - start);
    printf("Lookups/sec: %.2e\n", lookups / (end - start));
    printf("Checksum: %.6e\n", result);
    
    cleanup(sim);
    return 0;
}

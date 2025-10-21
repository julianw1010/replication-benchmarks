#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define TABLE_SIZE (100ULL * 1024 * 1024 * 1024 / sizeof(Entry))  // ~100GB
#define PROBE_COUNT (4000000000ULL)  // 4B probes per thread
#define HASH_MASK (TABLE_SIZE - 1)

typedef struct {
    uint64_t key;
    uint64_t value;
} Entry;

static Entry *hash_table;

static inline uint64_t hash(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

void build_phase(uint64_t num_entries) {
    printf("Building hash table with %lu entries (%.2f GB)...\n", 
           num_entries, (num_entries * sizeof(Entry)) / (1024.0 * 1024.0 * 1024.0));
    
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < num_entries; i++) {
        uint64_t key = hash(i);
        uint64_t idx = key & HASH_MASK;
        hash_table[idx].key = key;
        hash_table[idx].value = i;
    }
    
    double elapsed = omp_get_wtime() - start;
    printf("Build completed in %.2f seconds\n", elapsed);
}

uint64_t probe_phase(uint64_t num_probes) {
    printf("Probing with %lu lookups per thread...\n", num_probes);
    
    uint64_t total_hits = 0;
    double start = omp_get_wtime();
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t local_hits = 0;
        uint64_t seed = hash(tid + 12345);
        
        #pragma omp for schedule(static) reduction(+:total_hits)
        for (uint64_t i = 0; i < num_probes; i++) {
            seed = hash(seed + i);
            uint64_t idx = seed & HASH_MASK;
            
            // Random walk to stress page tables
            for (int j = 0; j < 4; j++) {
                uint64_t next_idx = hash(hash_table[idx].value) & HASH_MASK;
                if (hash_table[next_idx].key != 0) {
                    local_hits++;
                }
                idx = next_idx;
            }
        }
        total_hits += local_hits;
    }
    
    double elapsed = omp_get_wtime() - start;
    double throughput = (num_probes * 4.0) / elapsed / 1e6;
    
    printf("Probe completed in %.2f seconds\n", elapsed);
    printf("Throughput: %.2f M probes/sec\n", throughput);
    printf("Total hits: %lu\n", total_hits);
    
    return total_hits;
}

int main(int argc, char **argv) {
    int num_threads = omp_get_max_threads();
    
    if (argc > 1) {
        num_threads = atoi(argv[1]);
        omp_set_num_threads(num_threads);
    }
    
    printf("HashJoin Benchmark - Page Table Walk Stress Test\n");
    printf("Threads: %d\n", num_threads);
    printf("Table size: %lu entries (%.2f GB)\n", 
           TABLE_SIZE, (TABLE_SIZE * sizeof(Entry)) / (1024.0 * 1024.0 * 1024.0));
    
    // Allocate hash table
    hash_table = (Entry *)malloc(TABLE_SIZE * sizeof(Entry));
    if (!hash_table) {
        fprintf(stderr, "Failed to allocate memory\n");
        return 1;
    }
    
    // Initialize to zero
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < TABLE_SIZE; i++) {
        hash_table[i].key = 0;
        hash_table[i].value = 0;
    }
    
    // Run benchmark phases
    build_phase(TABLE_SIZE / 2);
    probe_phase(PROBE_COUNT);
    
    free(hash_table);
    printf("\nBenchmark complete\n");
    
    return 0;
}

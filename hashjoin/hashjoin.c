#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define TARGET_GB 100ULL
#define BYTES_PER_ELEMENT 228ULL
#define NUM_ELEMENTS ((TARGET_GB * 1024ULL * 1024ULL * 1024ULL) / BYTES_PER_ELEMENT)
#define HASH_MASK (NUM_ELEMENTS - 1)

typedef struct __attribute__((packed)) {
    uint64_t key;
    uint64_t value;
    uint64_t hash;
    uint64_t metadata;
    char payload[196];
} Entry;

static Entry *hash_table;

static inline uint64_t hash_fn(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

void build_phase(void) {
    double start = omp_get_wtime();
    
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < NUM_ELEMENTS; i++) {
        uint64_t key = hash_fn(i);
        uint64_t idx = key & HASH_MASK;
        hash_table[idx].key = key;
        hash_table[idx].value = i;
        hash_table[idx].hash = hash_fn(key);
        hash_table[idx].metadata = i ^ key;
    }
    
    printf("Build: %.2f sec\n", omp_get_wtime() - start);
}

uint64_t probe_phase(void) {
    uint64_t total_probes = NUM_ELEMENTS * 40;
    uint64_t total_hits = 0;
    double start = omp_get_wtime();
    
    #pragma omp parallel reduction(+:total_hits)
    {
        int tid = omp_get_thread_num();
        uint64_t idx = hash_fn(tid * 0x9e3779b97f4a7c15ULL) & HASH_MASK;
        
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < total_probes; i++) {
            for (int hop = 0; hop < 8; hop++) {
                uint64_t val = hash_table[idx].value;
                uint64_t key = hash_table[idx].key;
                idx = hash_fn(val ^ key ^ hop) & HASH_MASK;
                
                if (key != 0) {
                    total_hits++;
                }
            }
        }
    }
    
    double elapsed = omp_get_wtime() - start;
    printf("Probe: %.2f sec | %.2f M probes/sec | Hits: %lu\n", 
           elapsed, (total_probes * 8.0) / elapsed / 1e6, total_hits);
    
    return total_hits;
}

int main(int argc, char **argv) {
    if (argc > 1) omp_set_num_threads(atoi(argv[1]));
    
    size_t total_bytes = NUM_ELEMENTS * sizeof(Entry);
    printf("HashJoin Multi-Socket Benchmark\n");
    printf("Threads: %d | Elements: %lu | Size: %.2fGB\n", 
           omp_get_max_threads(), NUM_ELEMENTS, 
           total_bytes / (1024.0*1024.0*1024.0));
    
    hash_table = (Entry *)malloc(total_bytes);
    if (!hash_table) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    #pragma omp parallel for
    for (uint64_t i = 0; i < NUM_ELEMENTS; i++) {
        hash_table[i].key = 0;
    }
    
    build_phase();
    probe_phase();
    
    free(hash_table);
    return 0;
}

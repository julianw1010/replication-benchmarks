#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

#define TARGET_GB 100ULL
#define BYTES_PER_ENTRY 228ULL
#define TABLE_SIZE ((TARGET_GB * 1024ULL * 1024ULL * 1024ULL) / BYTES_PER_ENTRY)
#define LOAD_FACTOR 0.75
#define NUM_ELEMENTS ((uint64_t)(TABLE_SIZE * LOAD_FACTOR))
#define MAX_PROBE 64

typedef struct __attribute__((packed)) {
    uint64_t key;
    uint64_t value;
    uint64_t hash;
    uint64_t chain_len;
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
        uint64_t key = hash_fn(i) | 1;
        uint64_t idx = hash_fn(key) % TABLE_SIZE;
        hash_table[idx].key = key;
        hash_table[idx].value = i;
        hash_table[idx].hash = hash_fn(key);
    }
    
    double elapsed = omp_get_wtime() - start;
    printf("Build: %.2f sec | %.2f M ops/sec\n",
           elapsed, NUM_ELEMENTS / elapsed / 1e6);
}

uint64_t probe_phase(void) {
    uint64_t total_probes = NUM_ELEMENTS * 3;
    uint64_t hits = 0;
    double start = omp_get_wtime();
    
    #pragma omp parallel reduction(+:hits)
    {
        uint64_t seed = hash_fn(omp_get_thread_num() * 0x123456789abcdefULL);
        
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < total_probes; i++) {
            seed = hash_fn(seed + i);
            uint64_t key = hash_fn(seed % NUM_ELEMENTS) | 1;
            uint64_t idx = hash_fn(key) % TABLE_SIZE;
            
            for (int probe = 0; probe < MAX_PROBE; probe++) {
                uint64_t pos = (idx + probe) % TABLE_SIZE;
                if (hash_table[pos].key == key) {
                    hits += hash_table[pos].value;
                    break;
                }
            }
        }
    }
    
    double elapsed = omp_get_wtime() - start;
    printf("Probe: %.2f sec | %.2f M probes/sec | Hits: %lu\n",
           elapsed, total_probes / elapsed / 1e6, hits);
    
    return hits;
}

int main(int argc, char **argv) {
    if (argc > 1) omp_set_num_threads(atoi(argv[1]));
    
    size_t total_bytes = TABLE_SIZE * sizeof(Entry);
    printf("HashJoin Multi-Socket Benchmark (Mitosis-style)\n");
    printf("Threads: %d | Elements: %lu | Size: %.2f GB\n",
           omp_get_max_threads(), NUM_ELEMENTS, total_bytes / (1024.0*1024.0*1024.0));
    
    hash_table = (Entry *)malloc(total_bytes);
    if (!hash_table) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    // Quick parallel touch to allocate pages
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < TABLE_SIZE; i += 512) {
        hash_table[i].key = 0;
    }
    
    build_phase();
    probe_phase();
    
    free(hash_table);
    return 0;
}

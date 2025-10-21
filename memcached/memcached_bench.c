#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#define KEY_SIZE 8
#define VAL_SIZE 24
#define ENTRY_SIZE (KEY_SIZE + VAL_SIZE)
#define NUM_ENTRIES (100ULL * 1024 * 1024 * 1024 / ENTRY_SIZE)

typedef struct {
    uint64_t key;
    char value[VAL_SIZE];
} __attribute__((packed)) entry_t;

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

int main(int argc, char **argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    uint64_t iterations = (argc > 2) ? strtoull(argv[2], NULL, 10) : 8500000000ULL;
    
    printf("Allocating %.2f GB...\n", (NUM_ENTRIES * ENTRY_SIZE) / (1024.0 * 1024.0 * 1024.0));
    
    entry_t *table = (entry_t*)malloc(NUM_ENTRIES * sizeof(entry_t));
    if (!table) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    #pragma omp parallel for
    for (uint64_t i = 0; i < NUM_ENTRIES; i++) {
        table[i].key = i;
    }
    
    printf("Running %d threads, %llu iterations...\n", num_threads, iterations);
    
    omp_set_num_threads(num_threads);
    
    double start = omp_get_wtime();
    uint64_t total_sum = 0;
    
    #pragma omp parallel reduction(+:total_sum)
    {
        uint64_t rng_state = omp_get_thread_num() + 1;
        uint64_t local_sum = 0;
        uint64_t iters = iterations / num_threads;
        
        for (uint64_t i = 0; i < iters; i++) {
            uint64_t idx = xorshift64(&rng_state) % NUM_ENTRIES;
            local_sum += table[idx].key;
        }
        
        total_sum += local_sum;
    }
    
    double elapsed = omp_get_wtime() - start;
    double ops_per_sec = iterations / elapsed;
    
    printf("Time: %.2f s\n", elapsed);
    printf("Throughput: %.2f M ops/s\n", ops_per_sec / 1e6);
    printf("Checksum: %llu\n", total_sum);
    
    free(table);
    return 0;
}

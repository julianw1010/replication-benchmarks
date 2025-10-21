#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// Mitosis Memcached spec: keysize=8, element=24, 100% reads
// Scaled to 100GB from original 576M elements / 363GB
#define KEY_SIZE 8
#define VAL_SIZE 24
#define NUM_BUCKETS (1ULL << 28)  // 268M buckets = 2GB
#define NUM_ENTRIES (2500000000ULL)  // 2.5B entries * 40B = 100GB

typedef struct entry {
    uint64_t key;
    char value[VAL_SIZE];
    struct entry *next;
} entry_t;

typedef struct {
    entry_t *head;
} bucket_t;

static inline uint64_t hash_func(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return *state = x;
}

static inline uint64_t zipfian_next(uint64_t *state, uint64_t n, double alpha) {
    double u = (double)xorshift64(state) / (double)UINT64_MAX;
    return (uint64_t)(n * pow(u, -1.0 / (alpha - 1.0))) % n;
}

entry_t* hash_get(bucket_t *table, uint64_t key) {
    uint64_t bucket_idx = hash_func(key) % NUM_BUCKETS;
    entry_t *curr = table[bucket_idx].head;
    
    while (curr) {
        if (curr->key == key) return curr;
        curr = curr->next;
    }
    return NULL;
}

int main(int argc, char **argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    uint64_t iterations = (argc > 2) ? strtoull(argv[2], NULL, 10) : 1000000000ULL;
    
    size_t table_size = NUM_BUCKETS * sizeof(bucket_t);
    size_t entries_size = NUM_ENTRIES * sizeof(entry_t);
    printf("Target memory: %.2f GB (%.2f GB buckets + %.2f GB entries)\n", 
           (table_size + entries_size) / (1024.0 * 1024.0 * 1024.0),
           table_size / (1024.0 * 1024.0 * 1024.0),
           entries_size / (1024.0 * 1024.0 * 1024.0));
    
    bucket_t *table = (bucket_t*)calloc(NUM_BUCKETS, sizeof(bucket_t));
    entry_t *entries = (entry_t*)malloc(NUM_ENTRIES * sizeof(entry_t));
    
    if (!table || !entries) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    printf("Populating %llu entries...\n", NUM_ENTRIES);
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < NUM_ENTRIES; i++) {
        entries[i].key = i;
        memset(entries[i].value, (char)(i & 0xFF), VAL_SIZE);
        
        uint64_t bucket_idx = hash_func(i) % NUM_BUCKETS;
        
        entry_t *old_head;
        do {
            old_head = table[bucket_idx].head;
            entries[i].next = old_head;
        } while (!__sync_bool_compare_and_swap(&table[bucket_idx].head, old_head, &entries[i]));
    }
    
    printf("Running %d threads, %llu iterations (100%% reads, Zipfian α=1.2)...\n", 
           num_threads, iterations);
    omp_set_num_threads(num_threads);
    
    double start = omp_get_wtime();
    uint64_t total_hits = 0;
    
    #pragma omp parallel reduction(+:total_hits)
    {
        uint64_t rng_state = (uint64_t)omp_get_thread_num() * 0x123456789ABCDEFULL + (uint64_t)time(NULL);
        uint64_t local_hits = 0;
        uint64_t iters = iterations / num_threads;
        
        for (uint64_t i = 0; i < iters; i++) {
            uint64_t key = zipfian_next(&rng_state, NUM_ENTRIES, 1.2);
            entry_t *result = hash_get(table, key);
            if (result) local_hits++;
        }
        
        total_hits += local_hits;
    }
    
    double elapsed = omp_get_wtime() - start;
    double ops_per_sec = iterations / elapsed;
    
    printf("Time: %.2f s\n", elapsed);
    printf("Throughput: %.2f M ops/s\n", ops_per_sec / 1e6);
    printf("Hit rate: %.2f%%\n", (total_hits * 100.0) / iterations);
    
    free(entries);
    free(table);
    return 0;
}

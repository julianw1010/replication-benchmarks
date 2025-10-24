#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

// Scaled benchmark parameters for ~100GB memory footprint
#define NUM_ELEMENTS 1342177280UL  // 1280M elements
#define KEY_SIZE 8
#define VALUE_SIZE 24
#define HASH_TABLE_SIZE (NUM_ELEMENTS * 2)  // Load factor ~0.5
#define NUM_OPERATIONS 80000000UL  // 100M operations per thread
#define CACHE_LINE_SIZE 64

// Hash table entry structure
typedef struct hash_entry {
    uint64_t key;
    char value[VALUE_SIZE];
    struct hash_entry *next;
    char padding[CACHE_LINE_SIZE - sizeof(uint64_t) - VALUE_SIZE - sizeof(void*)];
} __attribute__((aligned(CACHE_LINE_SIZE))) hash_entry_t;

// Hash table structure
typedef struct {
    hash_entry_t **buckets;
    hash_entry_t *entries;
    size_t num_entries;
    size_t num_buckets;
} hash_table_t;

// Fast hash function (FNV-1a variant)
static inline uint64_t hash_key(uint64_t key, size_t num_buckets) {
    uint64_t hash = 14695981039346656037ULL;
    hash ^= key;
    hash *= 1099511628211ULL;
    return hash % num_buckets;
}

// Random number generator (xorshift64)
static inline uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

// Initialize hash table
hash_table_t* init_hash_table() {
    printf("Initializing hash table with %lu elements...\n", NUM_ELEMENTS);
    printf("Hash table buckets: %lu\n", HASH_TABLE_SIZE);
    
    hash_table_t *ht = malloc(sizeof(hash_table_t));
    if (!ht) {
        fprintf(stderr, "Failed to allocate hash table structure\n");
        exit(1);
    }
    
    ht->num_entries = NUM_ELEMENTS;
    ht->num_buckets = HASH_TABLE_SIZE;
    
    // NUMA-aware allocation: use malloc instead of calloc
    ht->buckets = malloc(HASH_TABLE_SIZE * sizeof(hash_entry_t*));
    if (!ht->buckets) {
        fprintf(stderr, "Failed to allocate hash table buckets\n");
        exit(1);
    }
    
    // First-touch initialization of buckets in parallel for NUMA distribution
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        ht->buckets[i] = NULL;
    }
    
    // Allocate all entries
    ht->entries = malloc(NUM_ELEMENTS * sizeof(hash_entry_t));
    if (!ht->entries) {
        fprintf(stderr, "Failed to allocate hash table entries\n");
        exit(1);
    }
    
    // Initialize entries and insert into hash table
    // Use static scheduling for proper first-touch NUMA distribution
    printf("Populating hash table...\n");
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        hash_entry_t *entry = &ht->entries[i];
        entry->key = i;
        
        // Initialize value with pseudo-random data
        uint64_t seed = i;
        for (size_t j = 0; j < VALUE_SIZE / sizeof(uint64_t); j++) {
            ((uint64_t*)entry->value)[j] = xorshift64(&seed);
        }
        
        entry->next = NULL;
        
        // Insert into hash table
        size_t bucket = hash_key(i, HASH_TABLE_SIZE);
        
        // Thread-safe insertion
        hash_entry_t *old_head;
        do {
            old_head = __atomic_load_n(&ht->buckets[bucket], __ATOMIC_ACQUIRE);
            entry->next = old_head;
        } while (!__atomic_compare_exchange_n(&ht->buckets[bucket], &old_head, entry,
                                               0, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
    }
    
    printf("Hash table initialization complete\n");
    return ht;
}

// Lookup operation (100% reads)
static inline hash_entry_t* lookup(hash_table_t *ht, uint64_t key) {
    size_t bucket = hash_key(key, ht->num_buckets);
    hash_entry_t *entry = ht->buckets[bucket];
    
    while (entry) {
        if (entry->key == key) {
            return entry;
        }
        entry = entry->next;
    }
    
    return NULL;
}

// Benchmark worker
void run_benchmark(hash_table_t *ht, int thread_id, uint64_t *ops_count, double *elapsed) {
    uint64_t seed = thread_id + time(NULL);
    uint64_t local_ops = 0;
    uint64_t hits = 0;
    
    struct timeval start, end;
    gettimeofday(&start, NULL);
    
    // Perform read operations
    for (uint64_t i = 0; i < NUM_OPERATIONS; i++) {
        uint64_t key = xorshift64(&seed) % NUM_ELEMENTS;
        hash_entry_t *entry = lookup(ht, key);
        
        if (entry) {
            hits++;
            // Simulate using the value
            __asm__ __volatile__("" : : "r"(entry->value[0]) : "memory");
        }
        
        local_ops++;
    }
    
    gettimeofday(&end, NULL);
    *elapsed = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
    *ops_count = local_ops;
    
    printf("Thread %d: %lu operations, %lu hits (%.2f%%), %.2f seconds, %.2f Mops/s\n",
           thread_id, local_ops, hits, (hits * 100.0) / local_ops, *elapsed,
           local_ops / (*elapsed * 1000000.0));
}

int main(int argc, char *argv[]) {
    printf("=== Memcached Benchmark (NUMA-Aware - Scaled to 100GB) ===\n");
    printf("Parameters:\n");
    printf("  Elements: %lu (%.1fM)\n", NUM_ELEMENTS, NUM_ELEMENTS / 1048576.0);
    printf("  Key size: %d bytes\n", KEY_SIZE);
    printf("  Value size: %d bytes\n", VALUE_SIZE);
    printf("  Entry size: %lu bytes\n", sizeof(hash_entry_t));
    printf("  Estimated memory: ~%.2f GB\n", 
           (NUM_ELEMENTS * sizeof(hash_entry_t) + HASH_TABLE_SIZE * sizeof(void*)) / (1024.0 * 1024.0 * 1024.0));
    printf("  Workload: 100%% reads\n");
    printf("  Operations per thread: %lu\n", NUM_OPERATIONS);
    
    int num_threads = omp_get_max_threads();
    printf("  Threads: %d\n\n", num_threads);
    
    // Initialize hash table
    struct timeval init_start, init_end;
    gettimeofday(&init_start, NULL);
    hash_table_t *ht = init_hash_table();
    gettimeofday(&init_end, NULL);
    double init_time = (init_end.tv_sec - init_start.tv_sec) + 
                       (init_end.tv_usec - init_start.tv_usec) / 1000000.0;
    printf("Initialization time: %.2f seconds\n\n", init_time);
    
    // Run benchmark
    printf("Starting benchmark...\n");
    
    // Allocate thread-local arrays inside parallel region for NUMA awareness
    uint64_t *ops_counts = NULL;
    double *elapsed_times = NULL;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            ops_counts = malloc(num_threads * sizeof(uint64_t));
            elapsed_times = malloc(num_threads * sizeof(double));
        }
    }
    
    struct timeval bench_start, bench_end;
    gettimeofday(&bench_start, NULL);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        run_benchmark(ht, tid, &ops_counts[tid], &elapsed_times[tid]);
    }
    
    gettimeofday(&bench_end, NULL);
    double total_time = (bench_end.tv_sec - bench_start.tv_sec) + 
                        (bench_end.tv_usec - bench_start.tv_usec) / 1000000.0;
    
    // Calculate statistics
    uint64_t total_ops = 0;
    for (int i = 0; i < num_threads; i++) {
        total_ops += ops_counts[i];
    }
    
    printf("\n=== Results ===\n");
    printf("Total operations: %lu\n", total_ops);
    printf("Total time: %.2f seconds\n", total_time);
    printf("Throughput: %.2f Mops/s\n", total_ops / (total_time * 1000000.0));
    printf("Average latency: %.2f ns/op\n", (total_time * 1000000000.0) / total_ops);
    
    // Cleanup
    free(ht->buckets);
    free(ht->entries);
    free(ht);
    free(ops_counts);
    free(elapsed_times);
    
    return 0;
}

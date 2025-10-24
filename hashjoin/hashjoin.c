/*
 * HashJoin Benchmark - NUMA-Optimized Baseline
 * 
 * Scaled from Mitosis paper: 2B elements (455GB) -> 1.59B elements (100GB)
 * 
 * This benchmark represents an ALREADY OPTIMIZED NUMA-aware application using:
 * - First-touch allocation (malloc + parallel initialization)
 * - Thread-local data structures allocated in parallel regions
 * - Static scheduling for NUMA locality
 * 
 * Any performance improvement with --pgtablerepl=all can be attributed
 * SOLELY to page table replication, not basic NUMA optimizations.
 * 
 * Usage: numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./hashjoin
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <sys/time.h>

// Scaled parameters for 100GB (calculated from 455GB / 2B elements)
#define NUM_TUPLES 1592413722UL
#define NUM_BUCKETS 2274876745UL

// Data structures
typedef struct tuple {
    uint64_t key;
    uint64_t value;
} tuple_t;

typedef struct hash_entry {
    uint64_t key;
    uint64_t value;
    struct hash_entry* next;
} hash_entry_t;

typedef struct {
    hash_entry_t* head;
} bucket_t;

// Global data structures (allocated with first-touch)
tuple_t* build_relation = NULL;
tuple_t* probe_relation = NULL;
bucket_t* hash_table = NULL;
hash_entry_t* hash_entries = NULL;

// Thread-local results
typedef struct {
    uint64_t matches;
    uint64_t probes;
    char padding[48]; // Cache line padding to avoid false sharing
} thread_result_t;

static inline uint64_t hash_function(uint64_t key) {
    // Simple multiplicative hash
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

static inline double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void print_memory_info() {
    size_t build_size = NUM_TUPLES * sizeof(tuple_t);
    size_t probe_size = NUM_TUPLES * sizeof(tuple_t);
    size_t buckets_size = NUM_BUCKETS * sizeof(bucket_t);
    size_t entries_size = NUM_TUPLES * sizeof(hash_entry_t);
    size_t total = build_size + probe_size + buckets_size + entries_size;
    
    printf("Memory allocation:\n");
    printf("  Build relation:   %7.2f GB (%lu tuples)\n", 
           build_size / (1024.0*1024.0*1024.0), NUM_TUPLES);
    printf("  Probe relation:   %7.2f GB (%lu tuples)\n", 
           probe_size / (1024.0*1024.0*1024.0), NUM_TUPLES);
    printf("  Hash buckets:     %7.2f GB (%lu buckets)\n", 
           buckets_size / (1024.0*1024.0*1024.0), NUM_BUCKETS);
    printf("  Hash entries:     %7.2f GB (%lu entries)\n", 
           entries_size / (1024.0*1024.0*1024.0), NUM_TUPLES);
    printf("  Total:            %7.2f GB\n", total / (1024.0*1024.0*1024.0));
    printf("\n");
}

void allocate_and_initialize() {
    double start, end;
    
    printf("=== Phase 1: NUMA-Aware Allocation and Initialization ===\n");
    print_memory_info();
    
    // Allocate memory (NOT initialized yet - will use first-touch)
    printf("Allocating memory...\n");
    start = get_time();
    
    build_relation = (tuple_t*)malloc(NUM_TUPLES * sizeof(tuple_t));
    probe_relation = (tuple_t*)malloc(NUM_TUPLES * sizeof(tuple_t));
    hash_table = (bucket_t*)malloc(NUM_BUCKETS * sizeof(bucket_t));
    hash_entries = (hash_entry_t*)malloc(NUM_TUPLES * sizeof(hash_entry_t));
    
    if (!build_relation || !probe_relation || !hash_table || !hash_entries) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        exit(1);
    }
    
    end = get_time();
    printf("Allocation time: %.2f seconds\n\n", end - start);
    
    // First-touch initialization for build relation
    // CRITICAL: Use schedule(static) to ensure contiguous chunks per thread
    // This distributes memory across NUMA nodes according to thread binding
    printf("First-touch: Initializing build relation...\n");
    start = get_time();
    
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < NUM_TUPLES; i++) {
            // Initialize on local NUMA node (first-touch policy)
            build_relation[i].key = i;
            build_relation[i].value = rand_r(&seed);
        }
    }
    
    end = get_time();
    printf("Build relation initialization: %.2f seconds (%.2f GB/s)\n", 
           end - start, (NUM_TUPLES * sizeof(tuple_t) / (1024.0*1024.0*1024.0)) / (end - start));
    
    // First-touch initialization for probe relation
    printf("First-touch: Initializing probe relation...\n");
    start = get_time();
    
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num() + 1000;
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < NUM_TUPLES; i++) {
            // Keys overlap with build relation for matches
            // About 80% match rate
            if (i % 5 == 0) {
                probe_relation[i].key = NUM_TUPLES + i; // No match
            } else {
                probe_relation[i].key = (i * 3) % NUM_TUPLES; // Match
            }
            probe_relation[i].value = rand_r(&seed);
        }
    }
    
    end = get_time();
    printf("Probe relation initialization: %.2f seconds (%.2f GB/s)\n", 
           end - start, (NUM_TUPLES * sizeof(tuple_t) / (1024.0*1024.0*1024.0)) / (end - start));
    
    // First-touch initialization for hash table buckets
    printf("First-touch: Initializing hash table buckets...\n");
    start = get_time();
    
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < NUM_BUCKETS; i++) {
        hash_table[i].head = NULL;
    }
    
    end = get_time();
    printf("Hash buckets initialization: %.2f seconds (%.2f GB/s)\n", 
           end - start, (NUM_BUCKETS * sizeof(bucket_t) / (1024.0*1024.0*1024.0)) / (end - start));
    
    // First-touch initialization for hash entries
    printf("First-touch: Initializing hash entries...\n");
    start = get_time();
    
    #pragma omp parallel for schedule(static)
    for (uint64_t i = 0; i < NUM_TUPLES; i++) {
        hash_entries[i].key = 0;
        hash_entries[i].value = 0;
        hash_entries[i].next = NULL;
    }
    
    end = get_time();
    printf("Hash entries initialization: %.2f seconds (%.2f GB/s)\n\n", 
           end - start, (NUM_TUPLES * sizeof(hash_entry_t) / (1024.0*1024.0*1024.0)) / (end - start));
}

void build_phase() {
    double start, end;
    
    printf("=== Phase 2: Hash Table Build ===\n");
    start = get_time();
    
    // Parallel hash table build
    // Each thread processes its chunk and inserts into hash table
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < NUM_TUPLES; i++) {
            uint64_t hash = hash_function(build_relation[i].key);
            uint64_t bucket_idx = hash % NUM_BUCKETS;
            
            // Setup hash entry
            hash_entries[i].key = build_relation[i].key;
            hash_entries[i].value = build_relation[i].value;
            
            // Insert into bucket (requires atomic operation for correctness)
            hash_entry_t* old_head;
            #pragma omp atomic capture
            {
                old_head = hash_table[bucket_idx].head;
                hash_table[bucket_idx].head = &hash_entries[i];
            }
            hash_entries[i].next = old_head;
        }
    }
    
    end = get_time();
    printf("Build phase time: %.2f seconds\n", end - start);
    printf("Build throughput: %.2f M tuples/s\n\n", (NUM_TUPLES / 1e6) / (end - start));
}

void probe_phase() {
    double start, end;
    
    printf("=== Phase 3: Hash Table Probe ===\n");
    
    // Allocate thread-local results INSIDE parallel region for NUMA locality
    thread_result_t* thread_results = NULL;
    int num_threads;
    
    start = get_time();
    
    #pragma omp parallel
    {
        // First thread allocates array
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            // CRITICAL: Allocate thread-local data here so it's on a local NUMA node
            thread_results = (thread_result_t*)malloc(num_threads * sizeof(thread_result_t));
            if (!thread_results) {
                fprintf(stderr, "ERROR: Failed to allocate thread results\n");
                exit(1);
            }
        }
        
        // Each thread initializes its own result (first-touch)
        int tid = omp_get_thread_num();
        thread_results[tid].matches = 0;
        thread_results[tid].probes = 0;
        
        // Probe phase
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < NUM_TUPLES; i++) {
            uint64_t hash = hash_function(probe_relation[i].key);
            uint64_t bucket_idx = hash % NUM_BUCKETS;
            
            thread_results[tid].probes++;
            
            // Search in bucket
            hash_entry_t* entry = hash_table[bucket_idx].head;
            while (entry != NULL) {
                if (entry->key == probe_relation[i].key) {
                    thread_results[tid].matches++;
                    break;
                }
                entry = entry->next;
            }
        }
    }
    
    end = get_time();
    
    // Aggregate results
    uint64_t total_matches = 0;
    uint64_t total_probes = 0;
    for (int i = 0; i < num_threads; i++) {
        total_matches += thread_results[i].matches;
        total_probes += thread_results[i].probes;
    }
    
    printf("Probe phase time: %.2f seconds\n", end - start);
    printf("Probe throughput: %.2f M tuples/s\n", (NUM_TUPLES / 1e6) / (end - start));
    printf("Total probes: %lu\n", total_probes);
    printf("Total matches: %lu (%.2f%% hit rate)\n\n", 
           total_matches, (total_matches * 100.0) / total_probes);
    
    free(thread_results);
}

void cleanup() {
    printf("=== Cleanup ===\n");
    free(build_relation);
    free(probe_relation);
    free(hash_table);
    free(hash_entries);
    printf("Memory freed\n\n");
}

int main(int argc, char** argv) {
    double total_start, total_end;
    
    printf("==========================================================\n");
    printf("HashJoin Benchmark - NUMA-Optimized Baseline\n");
    printf("==========================================================\n");
    printf("Configuration:\n");
    printf("  Tuples:  %lu (1.59B)\n", NUM_TUPLES);
    printf("  Buckets: %lu (2.27B)\n", NUM_BUCKETS);
    printf("  Threads: %d\n", omp_get_max_threads());
    printf("==========================================================\n\n");
    
    total_start = get_time();
    
    allocate_and_initialize();
    build_phase();
    probe_phase();
    
    total_end = get_time();
    
    printf("==========================================================\n");
    printf("Total execution time: %.2f seconds\n", total_end - total_start);
    printf("==========================================================\n");
    
    cleanup();
    
    return 0;
}

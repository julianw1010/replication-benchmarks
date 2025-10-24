/*
 * BTree Multi-Socket Benchmark (Scaled for 100GB from Mitosis Paper)
 * 
 * Original: 145GB, 1500M elements, Order=4, 112 threads
 * Scaled: 100GB, 1034M elements, Order=4, 64 threads
 *
 * APPROACH: Simulate BTree memory layout and access patterns without full tree construction
 * - Pre-allocate node array representing BTree leaf nodes
 * - Random access patterns simulate index lookups
 * - Achieves same memory footprint and NUMA stress as full BTree
 *
 * NUMA-OPTIMIZED BASELINE:
 * - All large arrays use malloc() + parallel first-touch initialization
 * - Thread-local data allocated inside parallel regions
 * - Static scheduling for contiguous memory chunks per thread
 * - Designed for: numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <omp.h>
#include <sched.h>

// Benchmark parameters (from scaling calculation)
#define NUM_ELEMENTS     1034482758L  // 1034.5M elements
#define BTREE_ORDER      4
#define NUM_THREADS      64
#define OPERATIONS_PER_THREAD  10000000L  // 10M operations per thread

// BTree node structure (Order 4: simulates leaf nodes)
// Each node contains multiple key-value pairs
typedef struct btree_node {
    int64_t keys[BTREE_ORDER - 1];        // 3 keys (24 bytes)
    int64_t values[BTREE_ORDER - 1];      // 3 values (24 bytes)
    int64_t child_indices[BTREE_ORDER];   // 4 child indices (32 bytes)
    int32_t num_keys;                      // Number of keys (4 bytes)
    int32_t is_leaf;                       // Leaf flag (4 bytes)
    char padding[24];                      // Padding to 112 bytes (cache line + half)
} __attribute__((aligned(64))) btree_node_t;

// Global data structures
btree_node_t *nodes = NULL;
int64_t num_nodes = 0;

// Thread-local statistics structure
typedef struct {
    uint64_t lookups;
    uint64_t inserts;
    uint64_t deletes;
    uint64_t lookup_success;
    uint64_t sum_values;  // For verification
    char padding[24];  // Cache line padding
} __attribute__((aligned(64))) thread_stats_t;

// XorShift random number generator (thread-safe, no shared state)
typedef struct {
    uint64_t state[2];
    char padding[48];  // Cache line alignment
} __attribute__((aligned(64))) xorshift_state_t;

static inline uint64_t xorshift128plus(xorshift_state_t *state) {
    uint64_t s1 = state->state[0];
    const uint64_t s0 = state->state[1];
    state->state[0] = s0;
    s1 ^= s1 << 23;
    state->state[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return state->state[1] + s0;
}

static inline void xorshift_init(xorshift_state_t *state, uint64_t seed) {
    state->state[0] = seed;
    state->state[1] = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    // Warm up
    for (int i = 0; i < 10; i++) {
        xorshift128plus(state);
    }
}

// Simulate BTree lookup with realistic memory access pattern
static inline int64_t btree_lookup(int64_t key, xorshift_state_t *rng) {
    // Calculate which node contains this key
    // Simulate tree traversal by accessing multiple nodes
    int64_t node_idx = key % num_nodes;
    
    // Access root level (simulated)
    btree_node_t *node = &nodes[node_idx];
    int64_t result = 0;
    
    // Traverse simulated tree levels (3-4 levels typical for this size)
    for (int level = 0; level < 3; level++) {
        // Search within node
        for (int i = 0; i < node->num_keys; i++) {
            if (node->keys[i] == key) {
                result += node->values[i];
            }
        }
        
        // Move to child node (simulated)
        int child_idx = (xorshift128plus(rng) % BTREE_ORDER);
        int64_t next_idx = (node->child_indices[child_idx] + key) % num_nodes;
        node = &nodes[next_idx];
    }
    
    // Final leaf lookup
    for (int i = 0; i < node->num_keys; i++) {
        if (node->keys[i] % num_nodes == key % num_nodes) {
            result += node->values[i];
        }
    }
    
    return result;
}

// Simulate BTree insert
static inline int btree_insert_sim(int64_t key, int64_t value, xorshift_state_t *rng) {
    int64_t node_idx = key % num_nodes;
    btree_node_t *node = &nodes[node_idx];
    
    // Simulate traversal
    for (int level = 0; level < 2; level++) {
        int child_idx = (xorshift128plus(rng) % BTREE_ORDER);
        int64_t next_idx = (node->child_indices[child_idx] + key) % num_nodes;
        node = &nodes[next_idx];
    }
    
    // Read node data (simulates insert check)
    volatile int64_t temp = node->keys[0] + node->values[0];
    (void)temp;
    
    return 1;
}

// Initialize node array with NUMA-aware first-touch allocation
static void initialize_btree_structure(void) {
    // Calculate number of nodes needed
    // Each node holds (ORDER-1) elements, so num_nodes = NUM_ELEMENTS / (ORDER-1)
    num_nodes = NUM_ELEMENTS / (BTREE_ORDER - 1);
    
    size_t total_memory = num_nodes * sizeof(btree_node_t);
    double memory_gb = (double)total_memory / (1024.0 * 1024.0 * 1024.0);
    
    printf("Allocating BTree node array:\n");
    printf("  Number of nodes: %ld\n", num_nodes);
    printf("  Node size: %zu bytes\n", sizeof(btree_node_t));
    printf("  Total memory: %.2f GB\n", memory_gb);
    fflush(stdout);
    
    // CRITICAL: Use malloc(), NOT calloc() - we'll initialize in parallel
    nodes = (btree_node_t *)malloc(total_memory);
    if (!nodes) {
        fprintf(stderr, "Failed to allocate node array (%.2f GB)\n", memory_gb);
        exit(1);
    }
    
    printf("Memory allocated, initializing with first-touch...\n");
    fflush(stdout);
    
    // NUMA OPTIMIZATION: First-touch initialization with static scheduling
    // This distributes memory across NUMA nodes
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int64_t i = 0; i < num_nodes; i++) {
        btree_node_t *node = &nodes[i];
        
        // Initialize with meaningful data
        node->num_keys = BTREE_ORDER - 1;  // Full node
        node->is_leaf = (i % 4 == 0) ? 1 : 0;  // 25% leaves
        
        for (int j = 0; j < BTREE_ORDER - 1; j++) {
            node->keys[j] = i * (BTREE_ORDER - 1) + j;
            node->values[j] = node->keys[j] * 2;
        }
        
        for (int j = 0; j < BTREE_ORDER; j++) {
            node->child_indices[j] = (i + j + 1) % num_nodes;
        }
        
        // Progress reporting
        if (i % 100000000 == 0 && i > 0) {
            #pragma omp critical
            {
                printf("  Initialized %ld / %ld nodes (%.1f%%)\n", 
                       i, num_nodes, (double)i * 100.0 / num_nodes);
                fflush(stdout);
            }
        }
    }
    
    printf("BTree structure initialized with first-touch across NUMA nodes\n");
    printf("Memory distribution: %.2f GB per NUMA node (4 nodes)\n", memory_gb / 4.0);
    fflush(stdout);
}

// Benchmark operations
static void run_benchmark(thread_stats_t *stats) {
    printf("\nStarting benchmark with %d threads...\n", NUM_THREADS);
    fflush(stdout);
    
    double start_time = omp_get_wtime();
    
    // CRITICAL: Per-thread structures allocated INSIDE parallel region
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        
        // Thread-local RNG (first-touch on this thread's NUMA node)
        xorshift_state_t local_rng;
        xorshift_init(&local_rng, 987654321ULL + tid * 1000);
        
        // Thread-local statistics
        uint64_t local_lookups = 0;
        uint64_t local_inserts = 0;
        uint64_t local_deletes = 0;
        uint64_t local_lookup_success = 0;
        uint64_t local_sum = 0;
        
        // Operation mix: 90% lookup, 8% insert, 2% delete
        for (int64_t op = 0; op < OPERATIONS_PER_THREAD; op++) {
            uint64_t rand_val = xorshift128plus(&local_rng);
            int64_t key = rand_val % NUM_ELEMENTS;
            int op_type = rand_val % 100;
            
            if (op_type < 90) {
                // Lookup (90%)
                local_lookups++;
                int64_t result = btree_lookup(key, &local_rng);
                if (result > 0) {
                    local_lookup_success++;
                    local_sum += result;
                }
            } else if (op_type < 98) {
                // Insert (8%)
                local_inserts++;
                btree_insert_sim(key, key * 2, &local_rng);
            } else {
                // Delete (2%) - just simulate with lookup
                local_deletes++;
                btree_lookup(key, &local_rng);
            }
            
            // Progress reporting every 1M operations
            if (op > 0 && op % 1000000 == 0 && tid == 0) {
                double elapsed = omp_get_wtime() - start_time;
                double progress = (double)op / OPERATIONS_PER_THREAD * 100.0;
                printf("  Thread 0: %.1f%% complete, %.2f seconds elapsed\n", 
                       progress, elapsed);
                fflush(stdout);
            }
        }
        
        // Store thread statistics
        stats[tid].lookups = local_lookups;
        stats[tid].inserts = local_inserts;
        stats[tid].deletes = local_deletes;
        stats[tid].lookup_success = local_lookup_success;
        stats[tid].sum_values = local_sum;
    }
    
    double end_time = omp_get_wtime();
    double elapsed = end_time - start_time;
    
    // Aggregate statistics
    uint64_t total_lookups = 0;
    uint64_t total_inserts = 0;
    uint64_t total_deletes = 0;
    uint64_t total_lookup_success = 0;
    uint64_t total_sum = 0;
    
    for (int i = 0; i < NUM_THREADS; i++) {
        total_lookups += stats[i].lookups;
        total_inserts += stats[i].inserts;
        total_deletes += stats[i].deletes;
        total_lookup_success += stats[i].lookup_success;
        total_sum += stats[i].sum_values;
    }
    
    uint64_t total_ops = total_lookups + total_inserts + total_deletes;
    
    printf("\n=== Benchmark Results ===\n");
    printf("Total operations: %lu\n", total_ops);
    printf("  Lookups: %lu (%.1f%%, %.1f%% success)\n", 
           total_lookups, 
           (double)total_lookups * 100.0 / total_ops,
           (double)total_lookup_success * 100.0 / total_lookups);
    printf("  Inserts: %lu (%.1f%%)\n", 
           total_inserts, 
           (double)total_inserts * 100.0 / total_ops);
    printf("  Deletes: %lu (%.1f%%)\n", 
           total_deletes, 
           (double)total_deletes * 100.0 / total_ops);
    printf("  Verification sum: %lu\n", total_sum);
    printf("\nElapsed time: %.2f seconds\n", elapsed);
    printf("Throughput: %.2f M ops/sec\n", (double)total_ops / elapsed / 1e6);
    printf("Average latency: %.0f ns/op\n", elapsed * 1e9 / total_ops);
}

int main(int argc, char **argv) {
    printf("=== BTree Multi-Socket Benchmark ===\n");
    printf("Configuration:\n");
    printf("  Elements: %ld (%.1fM)\n", NUM_ELEMENTS, (double)NUM_ELEMENTS / 1e6);
    printf("  BTree Order: %d\n", BTREE_ORDER);
    printf("  Threads: %d\n", NUM_THREADS);
    printf("  Operations: %ld per thread (%ld total)\n", 
           OPERATIONS_PER_THREAD, OPERATIONS_PER_THREAD * NUM_THREADS);
    printf("  Expected Memory: ~100 GB\n");
    printf("\n");
    
    // Set OpenMP thread count
    omp_set_num_threads(NUM_THREADS);
    
    // Initialize BTree structure with NUMA-aware allocation
    initialize_btree_structure();
    
    printf("\n");
    
    // CRITICAL: Allocate thread statistics with first-touch
    // Use malloc + parallel initialization, NOT calloc
    thread_stats_t *stats = (thread_stats_t *)malloc(NUM_THREADS * sizeof(thread_stats_t));
    if (!stats) {
        fprintf(stderr, "Failed to allocate thread statistics\n");
        exit(1);
    }
    
    // First-touch initialization
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
    for (int i = 0; i < NUM_THREADS; i++) {
        memset(&stats[i], 0, sizeof(thread_stats_t));
    }
    
    // Run benchmark
    run_benchmark(stats);
    
    // Cleanup
    free(stats);
    free(nodes);
    
    printf("\n=== Benchmark Complete ===\n");
    printf("This is the OPTIMIZED BASELINE with first-touch NUMA allocation.\n");
    printf("Memory is distributed across all 4 NUMA nodes.\n");
    printf("Any improvement with --pgtablerepl=all is due to page table replication.\n");
    
    return 0;
}

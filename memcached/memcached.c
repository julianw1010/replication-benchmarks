/*
 * NUMA-Optimized Memcached Benchmark
 * Simulates Mitosis paper workload: 100% reads, keysize=8, valuesize=24
 * Scaled to 300GB from original 363GB
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include <numa.h>
#include <numaif.h>
#include <sys/time.h>
#include <sched.h>
#include <unistd.h>
#include <pthread.h>
#include <assert.h>

// Scaled parameters (from 576M elements @ 363GB to 300GB)
#define NUM_ELEMENTS 476033057L
#define HASH_TABLE_SIZE 714049585L  // 1.5x elements for good performance

// Memcached parameters
#define KEY_SIZE 8
#define VALUE_SIZE 24
#define CACHE_LINE_SIZE 64

// Structure sizes aligned to cache lines to prevent false sharing
typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) {
    uint64_t key;                    // 8 bytes
    char value[VALUE_SIZE];           // 24 bytes
    uint32_t flags;                   // 4 bytes
    uint32_t expiry;                  // 4 bytes
    uint64_t cas;                     // 8 bytes (compare-and-swap)
    struct memcached_item *h_next;   // 8 bytes (hash chain)
    struct memcached_item *lru_next; // 8 bytes
    struct memcached_item *lru_prev; // 8 bytes
    char padding[128 - 72];           // Pad to 128 bytes (2 cache lines)
} memcached_item;

typedef struct __attribute__((aligned(CACHE_LINE_SIZE))) {
    memcached_item **buckets;        // Hash table buckets
    memcached_item *items;            // Item storage
    uint64_t *access_pattern;         // Random access pattern for reads
    size_t num_elements;
    size_t hash_size;
    
    // Statistics per thread (NUMA-local)
    uint64_t *thread_hits;
    uint64_t *thread_misses;
    uint64_t *thread_ops;
    
    // Additional structures to reach target memory footprint
    uint64_t *auxiliary_data1;       // Simulates connection buffers
    uint64_t *auxiliary_data2;       // Simulates slab allocator metadata
    uint64_t *auxiliary_data3;       // Simulates statistics/monitoring
    size_t aux_elements_per_array;   // Track actual allocation size
} memcached_state;

// Fast hash function for 8-byte keys
static inline uint64_t hash_key(uint64_t key, size_t hash_size) {
    // MurmurHash3 finalizer for good distribution
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key % hash_size;
}

// Initialize items with NUMA-aware first-touch
static void init_items_numa(memcached_state *state) {
    printf("Initializing %ld items with NUMA-aware first-touch...\n", NUM_ELEMENTS);
    
    // CRITICAL: Use static scheduling for contiguous chunks per thread
    // This ensures each thread initializes a contiguous region of memory
    // on its local NUMA node
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        // First-touch: memory is allocated on the NUMA node of the accessing thread
        state->items[i].key = i;
        
        // Initialize value with some pattern
        for (int j = 0; j < VALUE_SIZE; j++) {
            state->items[i].value[j] = (char)((i + j) & 0xFF);
        }
        
        state->items[i].flags = 0;
        state->items[i].expiry = 0;
        state->items[i].cas = i;
        state->items[i].h_next = NULL;
        state->items[i].lru_next = NULL;
        state->items[i].lru_prev = NULL;
    }
}

// Initialize hash table with NUMA-aware first-touch
static void init_hash_table_numa(memcached_state *state) {
    printf("Initializing hash table with %ld buckets...\n", HASH_TABLE_SIZE);
    
    // First-touch initialization for hash table
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
        state->buckets[i] = NULL;
    }
    
    // Insert items into hash table
    // For a read-only benchmark, we can do simple serial insertion
    // In production, this would use proper chaining for collisions
    printf("Inserting items into hash table...\n");
    
    size_t inserted = 0;
    size_t collisions = 0;
    
    // Simple linear probing for collision resolution
    for (size_t i = 0; i < NUM_ELEMENTS; i++) {
        uint64_t hash_idx = hash_key(state->items[i].key, HASH_TABLE_SIZE);
        uint64_t original_idx = hash_idx;
        
        // Linear probe to find empty slot
        while (state->buckets[hash_idx] != NULL) {
            collisions++;
            hash_idx = (hash_idx + 1) % HASH_TABLE_SIZE;
            
            // Prevent infinite loop in case table is full
            if (hash_idx == original_idx) {
                printf("Warning: Hash table full at item %ld\n", i);
                break;
            }
        }
        
        if (state->buckets[hash_idx] == NULL) {
            state->buckets[hash_idx] = &state->items[i];
            inserted++;
        }
        
        // Progress indicator every 10M items
        if ((i & 0xFFFFFF) == 0 && i > 0) {
            printf("  Inserted %ldM items (%.1f%%)...\n", 
                   i / 1000000, (double)i / NUM_ELEMENTS * 100);
        }
    }
    
    printf("Insertion complete: %ld items inserted, %ld collisions (%.2f%% fill rate)\n", 
           inserted, collisions, (double)inserted / HASH_TABLE_SIZE * 100);
}

// Initialize auxiliary data structures for memory footprint
static void init_auxiliary_data_numa(memcached_state *state) {
    printf("Initializing auxiliary data structures...\n");
    
    size_t aux_per_structure = state->aux_elements_per_array;
    
    if (aux_per_structure == 0) {
        printf("  Skipping auxiliary data initialization (allocation failed)\n");
        return;
    }
    
    size_t bytes_per_gb = 1024L * 1024L * 1024L;
    printf("  Initializing %.1f GB per auxiliary structure (%.1f GB total)\n",
           (aux_per_structure * sizeof(uint64_t)) / (double)bytes_per_gb,
           (aux_per_structure * 3 * sizeof(uint64_t)) / (double)bytes_per_gb);
    
    // Initialize auxiliary data 1 (simulates connection buffers)
    if (state->auxiliary_data1) {
        printf("  Initializing auxiliary_data1...\n");
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < aux_per_structure; i++) {
            state->auxiliary_data1[i] = i;
        }
    }
    
    // Initialize auxiliary data 2 (simulates slab metadata)
    if (state->auxiliary_data2) {
        printf("  Initializing auxiliary_data2...\n");
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < aux_per_structure; i++) {
            state->auxiliary_data2[i] = i * 2;
        }
    }
    
    // Initialize auxiliary data 3 (simulates statistics)
    if (state->auxiliary_data3) {
        printf("  Initializing auxiliary_data3...\n");
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < aux_per_structure; i++) {
            state->auxiliary_data3[i] = i * 3;
        }
    }
}

// Initialize random access pattern
static void init_access_pattern(memcached_state *state) {
    printf("Generating random access pattern...\n");
    
    // Each thread initializes its portion of the access pattern
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        size_t chunk_size = NUM_ELEMENTS / num_threads;
        size_t start = tid * chunk_size;
        size_t end = (tid == num_threads - 1) ? NUM_ELEMENTS : start + chunk_size;
        
        unsigned int seed = tid + time(NULL);
        
        for (size_t i = start; i < end; i++) {
            state->access_pattern[i] = rand_r(&seed) % NUM_ELEMENTS;
        }
    }
}

// Allocate all memory with NUMA awareness
static memcached_state* allocate_state_numa() {
    printf("Allocating memory structures...\n");
    
    memcached_state *state = malloc(sizeof(memcached_state));
    if (!state) {
        perror("Failed to allocate state");
        exit(1);
    }
    
    // CRITICAL: Use malloc (not calloc) for NUMA-aware first-touch allocation
    
    // Allocate items array
    state->items = (memcached_item *)malloc(NUM_ELEMENTS * sizeof(memcached_item));
    if (!state->items) {
        perror("Failed to allocate items");
        exit(1);
    }
    
    // Allocate hash table
    state->buckets = (memcached_item **)malloc(HASH_TABLE_SIZE * sizeof(memcached_item *));
    if (!state->buckets) {
        perror("Failed to allocate hash table");
        exit(1);
    }
    
    // Allocate access pattern
    state->access_pattern = (uint64_t *)malloc(NUM_ELEMENTS * sizeof(uint64_t));
    if (!state->access_pattern) {
        perror("Failed to allocate access pattern");
        exit(1);
    }
    
    // Allocate auxiliary data to reach 300GB footprint
    size_t target_aux_gb = 233;
    size_t bytes_per_gb = 1024L * 1024L * 1024L;
    size_t total_aux_bytes = target_aux_gb * bytes_per_gb;
    size_t aux_per_structure = total_aux_bytes / (3 * sizeof(uint64_t));
    
    printf("Allocating auxiliary data: 3 x %.1f GB = %.1f GB\n",
           (aux_per_structure * sizeof(uint64_t)) / (double)bytes_per_gb,
           (aux_per_structure * 3 * sizeof(uint64_t)) / (double)bytes_per_gb);
    
    state->auxiliary_data1 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
    state->auxiliary_data2 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
    state->auxiliary_data3 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
    
    if (!state->auxiliary_data1 || !state->auxiliary_data2 || !state->auxiliary_data3) {
        printf("Warning: Could not allocate full auxiliary data\n");
        printf("  auxiliary_data1: %s\n", state->auxiliary_data1 ? "OK" : "FAILED");
        printf("  auxiliary_data2: %s\n", state->auxiliary_data2 ? "OK" : "FAILED");
        printf("  auxiliary_data3: %s\n", state->auxiliary_data3 ? "OK" : "FAILED");
        
        // Free any partial allocations
        if (state->auxiliary_data1) { free(state->auxiliary_data1); state->auxiliary_data1 = NULL; }
        if (state->auxiliary_data2) { free(state->auxiliary_data2); state->auxiliary_data2 = NULL; }
        if (state->auxiliary_data3) { free(state->auxiliary_data3); state->auxiliary_data3 = NULL; }
        
        // Try smaller allocation
        printf("Trying smaller auxiliary allocation (50GB per structure)...\n");
        aux_per_structure = (50L * bytes_per_gb) / sizeof(uint64_t);  // 50GB per structure
        state->auxiliary_data1 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
        state->auxiliary_data2 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
        state->auxiliary_data3 = (uint64_t *)malloc(aux_per_structure * sizeof(uint64_t));
        
        if (!state->auxiliary_data1 || !state->auxiliary_data2 || !state->auxiliary_data3) {
            printf("Warning: Even smaller allocation failed. Running with minimal memory.\n");
            aux_per_structure = 0;
        }
    }
    
    state->aux_elements_per_array = aux_per_structure;
    
    // Allocate thread-local statistics (NUMA-aware)
    int num_threads = omp_get_max_threads();
    state->thread_hits = (uint64_t *)malloc(num_threads * sizeof(uint64_t) * 8);  // *8 for cache line padding
    state->thread_misses = (uint64_t *)malloc(num_threads * sizeof(uint64_t) * 8);
    state->thread_ops = (uint64_t *)malloc(num_threads * sizeof(uint64_t) * 8);
    
    state->num_elements = NUM_ELEMENTS;
    state->hash_size = HASH_TABLE_SIZE;
    
    return state;
}

// Benchmark function - 100% reads
static void benchmark_reads(memcached_state *state, int duration_seconds) {
    printf("Running benchmark (100%% reads) for %d seconds...\n", duration_seconds);
    
    struct timeval start, current;
    gettimeofday(&start, NULL);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        uint64_t local_hits = 0;
        uint64_t local_misses = 0;
        uint64_t local_ops = 0;
        
        // Get CPU and NUMA node information
        int cpu = sched_getcpu();
        int node = numa_node_of_cpu(cpu);
        
        #pragma omp single
        {
            printf("Running with %d threads\n", num_threads);
        }
        
        #pragma omp barrier
        
        // Only thread 0 reports its NUMA node as example
        if (tid == 0) {
            printf("Thread %d on CPU %d, NUMA node %d\n", tid, cpu, node);
        }
        
        // Each thread works on its portion of the access pattern
        size_t chunk_size = NUM_ELEMENTS / num_threads;
        size_t start_idx = tid * chunk_size;
        size_t end_idx = (tid == num_threads - 1) ? NUM_ELEMENTS : start_idx + chunk_size;
        
        struct timeval thread_current;
        gettimeofday(&thread_current, NULL);
        double elapsed = 0;
        
        while (elapsed < duration_seconds) {
            // Perform reads using the random access pattern
            for (size_t i = start_idx; i < end_idx && elapsed < duration_seconds; i++) {
                uint64_t item_idx = state->access_pattern[i];
                uint64_t key = state->items[item_idx].key;
                uint64_t hash_idx = hash_key(key, HASH_TABLE_SIZE);
                uint64_t original_idx = hash_idx;
                
                // Linear probe to find the item
                memcached_item *item = state->buckets[hash_idx];
                while (item != NULL && item->key != key) {
                    hash_idx = (hash_idx + 1) % HASH_TABLE_SIZE;
                    if (hash_idx == original_idx) {
                        // Wrapped around, item not found
                        item = NULL;
                        break;
                    }
                    item = state->buckets[hash_idx];
                }
                
                if (item && item->key == key) {
                    // Hit - read the value
                    volatile char first_byte = item->value[0];
                    (void)first_byte;  // Prevent optimization
                    local_hits++;
                } else {
                    local_misses++;
                }
                
                local_ops++;
                
                // Check time periodically
                if ((local_ops & 0xFFFF) == 0) {
                    gettimeofday(&thread_current, NULL);
                    elapsed = (thread_current.tv_sec - start.tv_sec) + 
                             (thread_current.tv_usec - start.tv_usec) / 1000000.0;
                }
            }
            
            // Wrap around if we finish the access pattern
            if (elapsed < duration_seconds) {
                start_idx = 0;
                end_idx = chunk_size;
            }
        }
        
        // Store thread-local statistics
        state->thread_hits[tid * 8] = local_hits;
        state->thread_misses[tid * 8] = local_misses;
        state->thread_ops[tid * 8] = local_ops;
    }
    
    // Aggregate statistics
    uint64_t total_hits = 0, total_misses = 0, total_ops = 0;
    int num_threads = omp_get_max_threads();
    
    for (int i = 0; i < num_threads; i++) {
        total_hits += state->thread_hits[i * 8];
        total_misses += state->thread_misses[i * 8];
        total_ops += state->thread_ops[i * 8];
    }
    
    gettimeofday(&current, NULL);
    double elapsed = (current.tv_sec - start.tv_sec) + 
                    (current.tv_usec - start.tv_usec) / 1000000.0;
    
    printf("\nBenchmark Results:\n");
    printf("==================\n");
    printf("Duration: %.2f seconds\n", elapsed);
    printf("Total operations: %lu\n", total_ops);
    printf("Throughput: %.2f ops/sec\n", total_ops / elapsed);
    printf("Throughput: %.2f Million ops/sec\n", (total_ops / elapsed) / 1000000.0);
    printf("Hit rate: %.2f%%\n", (double)total_hits / total_ops * 100);
}

// Check NUMA memory distribution
static void check_numa_distribution(memcached_state *state) {
    printf("\nChecking NUMA memory distribution...\n");
    
    // Re-initialize NUMA if needed
    if (numa_available() == -1) {
        printf("Warning: NUMA library not properly initialized, trying to reinitialize...\n");
        numa_set_localalloc();
    }
    
    int num_nodes = numa_num_configured_nodes();
    if (num_nodes <= 0) {
        printf("Unable to detect NUMA nodes for distribution check\n");
        return;
    }
    
    printf("System has %d NUMA nodes\n", num_nodes);
    
    // Sample memory pages to check distribution
    size_t sample_size = 1000;
    size_t sample_interval = NUM_ELEMENTS / sample_size;
    
    int *node_count = calloc(num_nodes, sizeof(int));
    
    for (size_t i = 0; i < sample_size; i++) {
        void *addr = &state->items[i * sample_interval];
        int node = -1;
        
        if (get_mempolicy(&node, NULL, 0, addr, MPOL_F_NODE | MPOL_F_ADDR) == 0) {
            if (node >= 0 && node < num_nodes) {
                node_count[node]++;
            }
        }
    }
    
    printf("Memory distribution (sampled):\n");
    for (int i = 0; i < num_nodes; i++) {
        printf("  Node %d: %d pages (%.1f%%)\n", 
               i, node_count[i], (double)node_count[i] / sample_size * 100);
    }
    
    free(node_count);
}

int main(int argc, char *argv[]) {
    int duration = 10;  // Default 10 second benchmark
    
    if (argc > 1) {
        duration = atoi(argv[1]);
    }
    
    // Check if NUMA is available
    if (numa_available() == -1) {
        printf("Warning: NUMA not available on this system\n");
    } else {
        printf("NUMA nodes: %d\n", numa_num_configured_nodes());
        printf("NUMA CPUs per node: %d\n", numa_num_configured_cpus() / numa_num_configured_nodes());
    }
    
    // Set number of threads to system thread count
    int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
    omp_set_num_threads(num_threads);
    printf("Using %d OpenMP threads\n", num_threads);
    
    // Allocate state structure
    memcached_state *state = allocate_state_numa();
    
    // Initialize with NUMA-aware first-touch
    init_items_numa(state);
    init_hash_table_numa(state);
    init_auxiliary_data_numa(state);
    init_access_pattern(state);
    
    // Check NUMA distribution
    check_numa_distribution(state);
    
    // Print memory usage estimate
    size_t items_size = NUM_ELEMENTS * sizeof(memcached_item);
    size_t hash_size = HASH_TABLE_SIZE * sizeof(memcached_item *);
    size_t pattern_size = NUM_ELEMENTS * sizeof(uint64_t);
    
    // Calculate actual auxiliary data size
    size_t aux_size = state->aux_elements_per_array * 3 * sizeof(uint64_t);
    size_t total_size = items_size + hash_size + pattern_size + aux_size;
    
    printf("\nMemory usage:\n");
    printf("  Items: %.2f GB\n", items_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Hash table: %.2f GB\n", hash_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Access pattern: %.2f GB\n", pattern_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Auxiliary data: %.2f GB\n", aux_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Total: %.2f GB\n", total_size / (1024.0 * 1024.0 * 1024.0));
    
    // Run benchmark
    printf("\nWarming up...\n");
    benchmark_reads(state, 2);  // 2 second warmup
    
    printf("\nRunning main benchmark...\n");
    benchmark_reads(state, duration);
    
    // Cleanup
    free(state->items);
    free(state->buckets);
    free(state->access_pattern);
    if (state->auxiliary_data1) free(state->auxiliary_data1);
    if (state->auxiliary_data2) free(state->auxiliary_data2);
    if (state->auxiliary_data3) free(state->auxiliary_data3);
    free(state->thread_hits);
    free(state->thread_misses);
    free(state->thread_ops);
    free(state);
    
    return 0;
}

// btree_bench.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <numa.h>

#define ORDER 4
#define MAX_KEYS (2 * ORDER - 1)
#define ELEMENT_SIZE 16
#define TARGET_GB 100ULL

typedef struct {
    uint64_t key;
    uint64_t value;
} Element;

typedef struct BTreeNode {
    Element elements[MAX_KEYS];
    struct BTreeNode* children[MAX_KEYS + 1];
    int num_keys;
    int is_leaf;
} BTreeNode;

static inline uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

int main(int argc, char** argv) {
    size_t node_size = sizeof(BTreeNode);
    uint64_t num_nodes = (TARGET_GB * 1024 * 1024 * 1024) / node_size;
    uint64_t total_elements = num_nodes * MAX_KEYS;
    
    int num_threads = omp_get_max_threads();
    uint64_t lookups_per_thread = 47000000ULL;
    
    if (argc > 1) lookups_per_thread = strtoull(argv[1], NULL, 10);
    
    double size_gb = (num_nodes * node_size) / (1024.0*1024.0*1024.0);
    
    printf("=== BTree Multi-Socket Benchmark ===\n");
    printf("Order: %d, Element Size: %d bytes\n", ORDER, ELEMENT_SIZE);
    printf("Threads: %d, Nodes: %lu (%.1fM elements)\n", 
           num_threads, num_nodes, total_elements/1e6);
    printf("Memory: %.1f GB\n", size_gb);
    
    if (numa_available() >= 0) {
        numa_set_interleave_mask(numa_all_nodes_ptr);
    }
    
    printf("\nAllocating...\n");
    double start = omp_get_wtime();
    
    BTreeNode* nodes = malloc(num_nodes * sizeof(BTreeNode));
    if (!nodes) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }
    
    #pragma omp parallel
    {
        uint64_t seed = omp_get_thread_num() + 1;
        
        #pragma omp for schedule(static)
        for (uint64_t i = 0; i < num_nodes; i++) {
            nodes[i].num_keys = MAX_KEYS;
            nodes[i].is_leaf = (i >= num_nodes / 2);
            
            for (int j = 0; j < MAX_KEYS; j++) {
                nodes[i].elements[j].key = xorshift64(&seed);
                nodes[i].elements[j].value = i;
            }
            
            if (!nodes[i].is_leaf) {
                for (int j = 0; j <= MAX_KEYS; j++) {
                    uint64_t child_idx = (i * 8 + j) % num_nodes;
                    nodes[i].children[j] = &nodes[child_idx];
                }
            }
        }
    }
    
    double init_time = omp_get_wtime() - start;
    printf("Init: %.2fs\n", init_time);
    
    printf("\nRunning index lookups (%luM total)...\n", 
           lookups_per_thread * num_threads / 1000000);
    start = omp_get_wtime();
    
    uint64_t total_sum = 0;
    
    #pragma omp parallel reduction(+:total_sum)
    {
        uint64_t seed = omp_get_thread_num() + 12345;
        uint64_t local_sum = 0;
        
        #pragma omp for schedule(dynamic, 1000)
        for (uint64_t i = 0; i < lookups_per_thread * num_threads; i++) {
            uint64_t idx = xorshift64(&seed) % num_nodes;
            BTreeNode* node = &nodes[idx];
            
            for (int depth = 0; depth < 4 && !node->is_leaf; depth++) {
                int child_idx = xorshift64(&seed) % (MAX_KEYS + 1);
                node = node->children[child_idx];
            }
            
            int key_idx = xorshift64(&seed) % node->num_keys;
            local_sum += node->elements[key_idx].value;
        }
        
        total_sum += local_sum;
    }
    
    double lookup_time = omp_get_wtime() - start;
    uint64_t total_lookups = lookups_per_thread * num_threads;
    
    printf("\n=== Results ===\n");
    printf("Lookup time: %.2fs\n", lookup_time);
    printf("Throughput: %.2fM lookups/sec\n", total_lookups / lookup_time / 1e6);
    printf("Checksum: %lu\n", total_sum);
    
    free(nodes);
    return 0;
}

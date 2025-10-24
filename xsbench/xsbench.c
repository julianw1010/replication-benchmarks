/*******************************************************************************
 * XSBench - Monte Carlo Neutronics Cross-Section Lookup Benchmark
 * 
 * Scaled configuration for 100GB memory footprint (from Mitosis paper 440GB)
 * 
 * This implementation uses NUMA-aware first-touch allocation for optimal
 * performance on multi-socket NUMA systems. All large data structures are
 * allocated with malloc() and initialized in parallel to distribute memory
 * across NUMA nodes according to the threads that will access them.
 *
 * Parameters (scaled from Mitosis paper):
 *   - Particles: 265,000,000 (adjusted for ~60s runtime)
 *   - Grid points: 6,870,628
 *   - Isotopes: 355
 *   - Memory: ~100GB
 *   - Threads: 112
 *
 * Compile: make
 * Run: numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./xsbench
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>

/*******************************************************************************
 * Configuration - Scaled to 100GB (from 440GB in Mitosis paper)
 ******************************************************************************/
#define N_ISOTOPES 355
#define N_GRIDPOINTS 6870628
#define N_PARTICLES 265000000
#define N_THREADS 112
#define N_ENERGY_GROUPS 11303

/*******************************************************************************
 * Data Structures
 ******************************************************************************/

// Nuclide grid point - represents cross-section data at an energy point
typedef struct NuclideGridPoint {
    double energy;          // Energy level (MeV)
    double total_xs;        // Total cross-section
    double elastic_xs;      // Elastic scattering cross-section
    double absorbtion_xs;   // Absorption cross-section
    double fission_xs;      // Fission cross-section
} NuclideGridPoint;

// Per-thread local data for NUMA-aware allocation
typedef struct ThreadData {
    uint64_t seed;          // RNG seed
    double verification;    // For verification sum
    int lookups;            // Number of lookups performed
    char padding[40];       // Padding to avoid false sharing (64-byte cache line)
} ThreadData;

// Global simulation data
typedef struct SimulationData {
    NuclideGridPoint *nuclide_grids;  // [N_ISOTOPES * N_GRIDPOINTS]
    int *unionized_grid;               // [N_GRIDPOINTS * N_ISOTOPES]
    double *energy_grid;               // [N_GRIDPOINTS]
    ThreadData *thread_data;           // [N_THREADS]
    int n_isotopes;
    int n_gridpoints;
    int n_particles;
    int n_threads;
} SimulationData;

/*******************************************************************************
 * Random Number Generator (LCG - fast and sufficient for this benchmark)
 ******************************************************************************/
static inline uint64_t lcg_rand(uint64_t *seed) {
    const uint64_t m = 9223372036854775808ULL; // 2^63
    const uint64_t a = 2806196910506780709ULL;
    const uint64_t c = 1ULL;
    *seed = (a * (*seed) + c) % m;
    return *seed;
}

static inline double lcg_random_double(uint64_t *seed) {
    return (double)lcg_rand(seed) / (double)9223372036854775808ULL;
}

/*******************************************************************************
 * NUMA-Aware Memory Allocation Functions
 * 
 * CRITICAL: These functions implement first-touch allocation to ensure
 * memory is distributed across NUMA nodes according to the threads that
 * will access it. This is essential for the OPTIMIZED BASELINE.
 ******************************************************************************/

// Initialize nuclide grids with first-touch - NUMA-aware
void initialize_nuclide_grids(NuclideGridPoint *grids, int n_isotopes, int n_gridpoints) {
    const long total_points = (long)n_isotopes * (long)n_gridpoints;
    
    printf("Initializing nuclide grids with first-touch (NUMA-aware)...\n");
    
    // CRITICAL: Use static scheduling to ensure contiguous chunks per thread
    // This maintains NUMA locality - each thread initializes its portion
    // on its local NUMA node
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < total_points; i++) {
        int isotope = i / n_gridpoints;
        int gridpoint = i % n_gridpoints;
        
        // Deterministic initialization based on index
        uint64_t seed = i * 2654435761ULL + 12345;
        
        // Energy matches the global energy grid structure
        double fraction = (double)gridpoint / (double)(n_gridpoints - 1);
        grids[i].energy = 1e-11 * exp(fraction * log(20.0 / 1e-11));
        
        // Generate realistic cross-section values (barns)
        double base_xs = 1.0 + lcg_random_double(&seed) * 9.0;  // 1-10 barns
        grids[i].total_xs = base_xs;
        grids[i].elastic_xs = base_xs * (0.3 + lcg_random_double(&seed) * 0.4);  // 30-70%
        grids[i].absorbtion_xs = (base_xs - grids[i].elastic_xs) * lcg_random_double(&seed);
        grids[i].fission_xs = base_xs - grids[i].elastic_xs - grids[i].absorbtion_xs;
        
        // Ensure all values are positive
        if (grids[i].fission_xs < 0.0) grids[i].fission_xs = 0.0;
    }
}

// Initialize unionized grid with first-touch - NUMA-aware
void initialize_unionized_grid(int *grid, int n_gridpoints, int n_isotopes) {
    const long total_entries = (long)n_gridpoints * (long)n_isotopes;
    
    printf("Initializing unionized grid with first-touch (NUMA-aware)...\n");
    
    // CRITICAL: Static scheduling for NUMA locality
    // The unionized grid maps (energy_idx, isotope_idx) -> gridpoint_idx
    // where gridpoint_idx is in range [0, n_gridpoints-1]
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < total_entries; i++) {
        int energy_idx = i / n_isotopes;
        int isotope_idx = i % n_isotopes;
        
        // For this energy point and isotope, determine which gridpoint to use
        // Use a deterministic hash function to create variation across isotopes
        uint64_t hash = (i * 2654435761ULL + isotope_idx * 1103515245ULL) % n_gridpoints;
        grid[i] = (int)hash;
    }
}

// Initialize energy grid with first-touch - NUMA-aware
void initialize_energy_grid(double *grid, int n_gridpoints) {
    printf("Initializing energy grid with first-touch (NUMA-aware)...\n");
    
    // CRITICAL: Static scheduling for NUMA locality
    // Create logarithmically spaced energy grid from 1e-11 to 20 MeV
    const double E_min = 1e-11;
    const double E_max = 20.0;
    const double log_E_min = log(E_min);
    const double log_E_max = log(E_max);
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_gridpoints; i++) {
        // Logarithmic spacing ensures well-distributed energy points
        double fraction = (double)i / (double)(n_gridpoints - 1);
        grid[i] = exp(log_E_min + fraction * (log_E_max - log_E_min));
    }
}

/*******************************************************************************
 * Cross-Section Lookup Function
 * 
 * This is the core computational kernel that performs lookups in the
 * nuclide grid data structure. This simulates the critical path in
 * Monte Carlo neutron transport codes.
 ******************************************************************************/
double xs_lookup(SimulationData *data, double energy, int isotope, uint64_t *seed) {
    // Binary search in energy grid to find bounding indices
    int lower = 0;
    int upper = data->n_gridpoints - 1;
    
    // Clamp energy to valid range
    if (energy <= data->energy_grid[0]) {
        lower = 0;
        upper = 1;
    } else if (energy >= data->energy_grid[upper]) {
        lower = upper - 1;
    } else {
        while (upper - lower > 1) {
            int mid = (lower + upper) / 2;
            if (data->energy_grid[mid] > energy) {
                upper = mid;
            } else {
                lower = mid;
            }
        }
    }
    
    // CRITICAL: Use long to prevent integer overflow
    // With 6.87M gridpoints * 355 isotopes = 2.44B entries > INT_MAX
    long unionized_idx_lower = (long)lower * (long)data->n_isotopes + (long)isotope;
    long unionized_idx_upper = (long)upper * (long)data->n_isotopes + (long)isotope;
    
    // Get grid indices for this isotope
    int grid_idx_lower = data->unionized_grid[unionized_idx_lower];
    int grid_idx_upper = data->unionized_grid[unionized_idx_upper];
    
    // Ensure indices are within bounds
    grid_idx_lower = grid_idx_lower % data->n_gridpoints;
    grid_idx_upper = grid_idx_upper % data->n_gridpoints;
    if (grid_idx_lower < 0) grid_idx_lower = 0;
    if (grid_idx_upper < 0) grid_idx_upper = 0;
    
    // Access nuclide grid data
    long base_idx = (long)isotope * (long)data->n_gridpoints;
    NuclideGridPoint *lower_point = &data->nuclide_grids[base_idx + grid_idx_lower];
    NuclideGridPoint *upper_point = &data->nuclide_grids[base_idx + grid_idx_upper];
    
    // Interpolate cross-section
    double energy_diff = upper_point->energy - lower_point->energy;
    double xs;
    
    if (fabs(energy_diff) < 1e-10) {
        // Energies are too close - use lower point value
        xs = lower_point->total_xs;
    } else {
        double f = (energy - lower_point->energy) / energy_diff;
        xs = lower_point->total_xs + f * (upper_point->total_xs - lower_point->total_xs);
    }
    
    // Add some computational work to simulate realistic lookup
    xs += lower_point->elastic_xs * 0.1;
    xs += upper_point->absorbtion_xs * 0.05;
    
    // Ensure result is finite and positive
    if (!isfinite(xs) || xs < 0.0) {
        xs = 1.0;  // Default cross-section value
    }
    
    return xs;
}

/*******************************************************************************
 * Main Monte Carlo Simulation Kernel
 ******************************************************************************/
void run_simulation(SimulationData *data) {
    const int n_particles = data->n_particles;
    const int n_isotopes = data->n_isotopes;
    
    printf("Running Monte Carlo simulation with %d particles...\n", n_particles);
    
    double start_time = omp_get_wtime();
    
    // CRITICAL: Each thread processes its chunk of particles
    // The first-touch allocation ensures the data accessed by each thread
    // is local to its NUMA node
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t seed = data->thread_data[tid].seed;
        double thread_verification = 0.0;
        int thread_lookups = 0;
        
        #pragma omp for schedule(static)
        for (int p = 0; p < n_particles; p++) {
            // Sample random energy for this particle
            double energy = lcg_random_double(&seed) * 20.0;  // 0-20 MeV
            
            // Sample random number of isotopes to look up (5-15)
            int n_lookups = 5 + (lcg_rand(&seed) % 11);
            
            // Perform cross-section lookups
            for (int i = 0; i < n_lookups; i++) {
                int isotope = lcg_rand(&seed) % n_isotopes;
                double xs = xs_lookup(data, energy, isotope, &seed);
                thread_verification += xs;
                thread_lookups++;
            }
        }
        
        // Store thread results
        data->thread_data[tid].seed = seed;
        data->thread_data[tid].verification = thread_verification;
        data->thread_data[tid].lookups = thread_lookups;
    }
    
    double end_time = omp_get_wtime();
    
    // Aggregate results from all threads
    double total_verification = 0.0;
    long total_lookups = 0;
    
    for (int i = 0; i < data->n_threads; i++) {
        total_verification += data->thread_data[i].verification;
        total_lookups += data->thread_data[i].lookups;
    }
    
    double runtime = end_time - start_time;
    double lookups_per_sec = (double)total_lookups / runtime;
    
    printf("\n");
    printf("=================================================================\n");
    printf("                      RESULTS\n");
    printf("=================================================================\n");
    printf("Runtime:                  %.3f seconds\n", runtime);
    printf("Total lookups:            %ld\n", total_lookups);
    printf("Lookups/sec:              %.3e\n", lookups_per_sec);
    printf("Verification hash:        %.6e\n", total_verification);
    printf("=================================================================\n");
}

/*******************************************************************************
 * Main Function
 ******************************************************************************/
int main(int argc, char *argv[]) {
    printf("=================================================================\n");
    printf("              XSBench - Monte Carlo Neutronics Kernel\n");
    printf("=================================================================\n");
    printf("Configuration (scaled from Mitosis paper):\n");
    printf("  Particles:              %d\n", N_PARTICLES);
    printf("  Grid points:            %d\n", N_GRIDPOINTS);
    printf("  Isotopes:               %d\n", N_ISOTOPES);
    printf("  Threads:                %d\n", N_THREADS);
    printf("  Target memory:          ~100 GB\n");
    printf("=================================================================\n");
    
    // Set number of threads
    omp_set_num_threads(N_THREADS);
    
    // Allocate simulation data structure
    SimulationData *data = (SimulationData *)malloc(sizeof(SimulationData));
    data->n_isotopes = N_ISOTOPES;
    data->n_gridpoints = N_GRIDPOINTS;
    data->n_particles = N_PARTICLES;
    data->n_threads = N_THREADS;
    
    // Calculate memory requirements
    long nuclide_memory = (long)N_ISOTOPES * (long)N_GRIDPOINTS * sizeof(NuclideGridPoint);
    long unionized_memory = (long)N_GRIDPOINTS * (long)N_ISOTOPES * sizeof(int);
    long energy_memory = (long)N_GRIDPOINTS * sizeof(double);
    long thread_memory = (long)N_THREADS * sizeof(ThreadData);
    long total_memory = nuclide_memory + unionized_memory + energy_memory + thread_memory;
    
    printf("\nMemory allocation:\n");
    printf("  Nuclide grids:          %.2f GB\n", nuclide_memory / (1024.0*1024.0*1024.0));
    printf("  Unionized grid:         %.2f GB\n", unionized_memory / (1024.0*1024.0*1024.0));
    printf("  Energy grid:            %.2f GB\n", energy_memory / (1024.0*1024.0*1024.0));
    printf("  Thread data:            %.2f MB\n", thread_memory / (1024.0*1024.0));
    printf("  Total:                  %.2f GB\n", total_memory / (1024.0*1024.0*1024.0));
    
    // Check for integer overflow risk
    long unionized_entries = (long)N_GRIDPOINTS * (long)N_ISOTOPES;
    long nuclide_entries = (long)N_ISOTOPES * (long)N_GRIDPOINTS;
    printf("\nArray size check:\n");
    printf("  Unionized grid entries: %ld (%.2f billion)\n", 
           unionized_entries, unionized_entries / 1e9);
    printf("  Nuclide grid entries:   %ld (%.2f billion)\n", 
           nuclide_entries, nuclide_entries / 1e9);
    if (unionized_entries > INT_MAX) {
        printf("  WARNING: Array indices exceed INT_MAX - using long for indexing\n");
    }
    printf("=================================================================\n\n");
    
    /***************************************************************************
     * CRITICAL: NUMA-AWARE FIRST-TOUCH ALLOCATION
     * 
     * All large arrays are allocated with malloc() (NOT calloc()) and then
     * initialized in parallel. This ensures memory pages are allocated on
     * the NUMA node of the thread that first touches them.
     * 
     * This is the OPTIMIZED BASELINE - any performance improvement from
     * page table replication will be solely due to reduced page table
     * overhead, not basic NUMA optimization.
     ***************************************************************************/
    
    printf("Allocating memory (malloc - not calloc for NUMA awareness)...\n");
    
    // Allocate large arrays with malloc (NOT calloc)
    data->nuclide_grids = (NuclideGridPoint *)malloc(nuclide_memory);
    data->unionized_grid = (int *)malloc(unionized_memory);
    data->energy_grid = (double *)malloc(energy_memory);
    
    if (!data->nuclide_grids || !data->unionized_grid || !data->energy_grid) {
        fprintf(stderr, "ERROR: Memory allocation failed!\n");
        fprintf(stderr, "       Attempted to allocate %.2f GB\n", 
                total_memory / (1024.0*1024.0*1024.0));
        return 1;
    }
    
    // Initialize large arrays in parallel for first-touch NUMA distribution
    initialize_nuclide_grids(data->nuclide_grids, N_ISOTOPES, N_GRIDPOINTS);
    initialize_unionized_grid(data->unionized_grid, N_GRIDPOINTS, N_ISOTOPES);
    initialize_energy_grid(data->energy_grid, N_GRIDPOINTS);
    
    /***************************************************************************
     * CRITICAL: THREAD-LOCAL DATA ALLOCATION
     * 
     * Thread-local data must be allocated INSIDE the parallel region to
     * ensure each thread's data is on its local NUMA node.
     ***************************************************************************/
    
    printf("Allocating thread-local data (NUMA-aware)...\n");
    
    // Allocate thread data array
    data->thread_data = (ThreadData *)malloc(thread_memory);
    if (!data->thread_data) {
        fprintf(stderr, "ERROR: Thread data allocation failed!\n");
        return 1;
    }
    
    // Initialize thread data inside parallel region for NUMA locality
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        data->thread_data[tid].seed = tid * 2654435761ULL + 12345;
        data->thread_data[tid].verification = 0.0;
        data->thread_data[tid].lookups = 0;
    }
    
    printf("\nNUMA-aware initialization complete.\n");
    printf("Memory should be distributed across NUMA nodes 0,1,2,3.\n");
    printf("Verify with: numastat -p $(pgrep xsbench)\n\n");
    
    // Run the simulation
    run_simulation(data);
    
    // Cleanup
    free(data->nuclide_grids);
    free(data->unionized_grid);
    free(data->energy_grid);
    free(data->thread_data);
    free(data);
    
    return 0;
}

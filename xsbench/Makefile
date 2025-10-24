# Makefile for XSBench - Monte Carlo Neutronics Benchmark
# Optimized for NUMA-aware multi-socket systems

CC = gcc
CFLAGS = -O3 -march=native -mtune=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

# Additional optimization flags for best performance
CFLAGS += -funroll-loops -ffast-math -ftree-vectorize
CFLAGS += -fno-strict-aliasing

# Debug flags (uncomment for debugging)
# CFLAGS += -g -O0 -DDEBUG

TARGET = xsbench
SOURCE = xsbench.c

.PHONY: all clean run run-numactl help

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)
	@echo ""
	@echo "==================================================================="
	@echo "Build complete: $(TARGET)"
	@echo "==================================================================="
	@echo ""
	@echo "Run with NUMA awareness (recommended):"
	@echo "  numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET)"
	@echo ""
	@echo "Or simply:"
	@echo "  make run-numactl"
	@echo ""
	@echo "To verify NUMA distribution (in another terminal):"
	@echo "  numastat -p \$$(pgrep xsbench)"
	@echo "==================================================================="

run: $(TARGET)
	./$(TARGET)

run-numactl: $(TARGET)
	@echo "Running with NUMA node binding..."
	@echo "Nodes: 0,1,2,3 (CPU and memory)"
	@echo ""
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET)

clean:
	rm -f $(TARGET) *.o

help:
	@echo "XSBench Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make              - Build the benchmark"
	@echo "  make run          - Run without NUMA binding"
	@echo "  make run-numactl  - Run with NUMA binding (recommended)"
	@echo "  make clean        - Remove built files"
	@echo "  make help         - Show this help"
	@echo ""
	@echo "Configuration:"
	@echo "  Memory footprint: ~100GB"
	@echo "  Threads: 112"
	@echo "  Particles: 25,000,000"
	@echo "  Grid points: 209,090"
	@echo ""
	@echo "NUMA Testing:"
	@echo "  1. Run: make run-numactl"
	@echo "  2. In another terminal: watch -n 1 'numastat -p \$$(pgrep xsbench)'"
	@echo "  3. Verify memory is distributed across nodes 0,1,2,3"
	@echo ""
	@echo "Page Table Replication Testing:"
	@echo "  Baseline (no replication):"
	@echo "    numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./xsbench"
	@echo ""
	@echo "  With page table replication (requires kernel support):"
	@echo "    numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 --pgtablerepl=all ./xsbench"

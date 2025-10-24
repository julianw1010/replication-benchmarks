# Makefile for BTree Multi-Socket Benchmark
# Optimized for NUMA systems with page table replication testing

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

# Target binary
TARGET = btree

# Source file
SRC = btree.c

# Default target
all: $(TARGET)

# Build benchmark
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"
	@echo ""
	@echo "Usage examples:"
	@echo "  Basic run (uses all NUMA nodes):"
	@echo "    numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET)"
	@echo ""
	@echo "  With page table replication (kernel 5.x+):"
	@echo "    numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 --pgtablerepl=all ./$(TARGET)"
	@echo ""
	@echo "  Custom thread count (via OMP_NUM_THREADS):"
	@echo "    OMP_NUM_THREADS=64 numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Rebuild from scratch
rebuild: clean all

# Check NUMA topology
check-numa:
	@echo "NUMA topology check:"
	@numactl --hardware 2>/dev/null || echo "numactl not found - install numactl package"

.PHONY: all clean rebuild check-numa

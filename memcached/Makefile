# Makefile for NUMA-optimized Memcached benchmark

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lnuma -lpthread -lm

TARGET = memcached
SOURCE = memcached.c

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCE) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET)

run-short: $(TARGET)
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 ./$(TARGET) 5

# Run with page table replication (for testing improvement)
run-pgtablerepl: $(TARGET)
	numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 --pgtablerepl=all ./$(TARGET)

# Check NUMA statistics
numa-stats:
	numastat -m

.PHONY: all clean run run-short run-pgtablerepl numa-stats

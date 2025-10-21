CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp -lm

TARGET = memcached_bench

all: $(TARGET)

$(TARGET): memcached_bench.c
	$(CC) $(CFLAGS) -o $(TARGET) memcached_bench.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

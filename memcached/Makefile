CC = gcc
CFLAGS = -O3 -fopenmp -Wall
LDFLAGS = -lm

TARGET = memcached_bench
SRC = memcached_bench.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
TARGET = memcached

.PHONY: all clean run

all: $(TARGET)

$(TARGET): memcached.c
	$(CC) $(CFLAGS) -o $@ $^ -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

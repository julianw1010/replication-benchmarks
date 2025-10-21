# Makefile
CC = gcc
CFLAGS = -O3 -fopenmp -march=native -Wall
LDFLAGS = -fopenmp -lm

TARGET = xsbench

all: $(TARGET)

$(TARGET): xsbench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

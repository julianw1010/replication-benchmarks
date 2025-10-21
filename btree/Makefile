# Makefile
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -DNDEBUG
LDFLAGS = -fopenmp -lnuma -lm

TARGET = btree
SRC = btree.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

run-100gb: $(TARGET)
	./$(TARGET) 1034000000 100000000

run-custom: $(TARGET)
	@echo "Usage: make run-custom ELEMENTS=N LOOKUPS=M"
	./$(TARGET) $(ELEMENTS) $(LOOKUPS)

.PHONY: all clean run run-100gb run-custom

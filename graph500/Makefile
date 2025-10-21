CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

TARGET = graph500
SRC = graph500.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	OMP_NUM_THREADS=$$(nproc) ./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

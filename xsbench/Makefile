CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp -lm

TARGET = xsbench
SRC = xsbench.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

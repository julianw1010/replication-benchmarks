# Makefile
CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp -lm

TARGET = canneal
SRC = canneal.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

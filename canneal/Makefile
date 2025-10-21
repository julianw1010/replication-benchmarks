CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -lm -fopenmp

TARGET = canneal
SRC = canneal.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

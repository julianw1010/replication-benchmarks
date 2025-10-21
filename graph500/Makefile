CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -lm

TARGET = graph500
SRC = graph500.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: all run clean

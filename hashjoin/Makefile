CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
TARGET = hashjoin

all: $(TARGET)

$(TARGET): hashjoin.c
	$(CC) $(CFLAGS) -o $(TARGET) hashjoin.c

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

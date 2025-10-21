CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp
TARGET = hashjoin

.PHONY: all clean run

all: $(TARGET)

$(TARGET): hashjoin.c
	$(CC) $(CFLAGS) -o $(TARGET) hashjoin.c $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

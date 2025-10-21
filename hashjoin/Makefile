CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
TARGET = hashjoin
SRC = hashjoin.c

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

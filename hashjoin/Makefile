CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -lm -fopenmp

TARGET = hashjoin
SRC = hashjoin.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

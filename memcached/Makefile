CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp

TARGET = memcached
SRC = memcached.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -lm -fopenmp

TARGET = graph500
SRC = graph500.c

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

clean:
	rm -f $(TARGET)

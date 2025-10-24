.PHONY: all clean

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp -lm

TARGET = canneal

all: $(TARGET)

$(TARGET): canneal.c
	$(CC) $(CFLAGS) -o $(TARGET) canneal.c $(LDFLAGS)

clean:
	rm -f $(TARGET)

CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Wall
LDFLAGS = -fopenmp -lnuma

TARGET = btree
SRC = btree.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run

include ../Makefile.config

CC=g++
CFLAGS=-Wall -O3 -std=c++11 -I/opt/homebrew/opt/google-sparsehash/include

OBJS=version.o buzhash.o
TARGETS=hashgen hashdup onion
all: $(TARGETS)

$(TARGETS): $(OBJS)

%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< 

%: %.c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

install: $(TARGETS)
	mkdir -p $(INSTALL_BIN) && install -m 755 $(TARGETS) $(INSTALL_BIN)

clean:
	rm -f $(OBJS) $(TARGETS)

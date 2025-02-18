all: print-this

print_this: print-this-src.c Makefile
	gcc -O0 -ggdb3 -std=c99 -static print-this-src.c -o print-this


clean:
	rm -rf print-this

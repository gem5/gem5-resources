CC ?= gcc
CXX ?= g++

all: matrix-multiply-omp

matrix-multiply-omp: matrix-omp.cpp matrix-omp.h matrix-omp-init.cpp
	$(CXX) -g -O3 -fopenmp -o matrix-omp matrix-omp-init.cpp matrix-omp.cpp -static

clean:
	rm -f ./matrix-multiply-omp ./*.o ./*.S

.PHONY: clean

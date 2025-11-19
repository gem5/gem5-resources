# See LICENSE.txt for license details.

CXX_FLAGS += -std=c++11 -O3 -Wall -DM5ANNOTATION
PAR_FLAG = -fopenmp

# Link with the m5 library
M5_LIB = -L/usr/local/lib -lm5

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq (,$(findstring sunCC,$(CXX)))
	CXX_FLAGS = -std=c++11 -xO3 -m64 -xtarget=native
	PAR_FLAG = -xopenmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

KERNELS = bc bfs cc cc_sv pr pr_spmv sssp tc
SUITE = $(KERNELS) converter

# Define output directory for compiled binaries
OUTDIR := bin
$(shell mkdir -p $(OUTDIR))

.PHONY: all
all: $(addprefix $(OUTDIR)/, $(SUITE))

# Compilation rule to place binaries in bin/
$(OUTDIR)/% : src/%.cc src/*.h
	$(CXX) $(CXX_FLAGS) $< $(M5_LIB) -o $@ -no-pie

# Testing
include test/test.mk

# Benchmark Automation
include benchmark/bench.mk

.PHONY: clean
clean:
	rm -f $(OUTDIR)/* test/out/*

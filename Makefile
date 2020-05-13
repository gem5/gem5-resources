source-dir          := src
output-dir          := output

# RISCV-Tests Parameters
riscv-tests-dir     := $(source-dir)/riscv-tests
riscv-benchmark-dir := $(riscv-tests-dir)/benchmarks
riscv-benchmarks    := $(riscv-benchmark-dir)/dhrystone.riscv \
                       $(riscv-benchmark-dir)/median.riscv \
                       $(riscv-benchmark-dir)/mm.riscv \
                       $(riscv-benchmark-dir)/mt-matmul.riscv \
                       $(riscv-benchmark-dir)/mt-vvadd.riscv \
                       $(riscv-benchmark-dir)/multiply.riscv \
                       $(riscv-benchmark-dir)/pmp.riscv \
                       $(riscv-benchmark-dir)/qsort.riscv \
                       $(riscv-benchmark-dir)/rsort.riscv \
                       $(riscv-benchmark-dir)/spmv.riscv \
                       $(riscv-benchmark-dir)/towers.riscv \
                       $(riscv-benchmark-dir)/vvadd.riscv
riscv-dist-dir      := $(output-dir)/test-progs/riscv-tests

# Insttest Parameters
insttest-src        := $(source-dir)/insttest
insttest-bin        := $(insttest-src)/bin
insttest-execs      := $(insttest-bin)/insttest-rv64a \
	                   $(insttest-bin)/insttest-rv64c \
					   $(insttest-bin)/insttest-rv64d \
					   $(insttest-bin)/insttest-rv64f \
					   $(insttest-bin)/insttest-rv64i \
					   $(insttest-bin)/insttest-rv64m
insttest-dist-dir   := $(output-dir)/test-progs/insttest/bin/riscv/linux

# PThreads Parameters
pthreads-dir        := $(source-dir)/pthreads
pthreads-x86-make   := Makefile.x86
pthreads-a32-make   := Makefile.aarch32
pthreads-a64-make   := Makefile.aarch64
pthreads-ri5-make   := Makefile.riscv
pthreads-x86-dir    := $(pthreads-dir)/bin.x86
pthreads-a32-dir    := $(pthreads-dir)/bin.aarch32
pthreads-a64-dir    := $(pthreads-dir)/bin.aarch64
pthreads-ri5-dir    := $(pthreads-dir)/bin.riscv64
pthreads-out        := $(output-dir)/test-progs/pthread
pthreads-x86-out    := $(pthreads-out)/x86
pthreads-a32-out    := $(pthreads-out)/aarch32
pthreads-a64-out    := $(pthreads-out)/aarch64
pthreads-ri5-out    := $(pthreads-out)/riscv64

# Square Parameters
square-dir			:= $(source-dir)/square
square-bin			:= $(square-dir)/bin
square-bench		:= $(square-bin)/square.o
square-out			:= $(output-dir)/test-progs/square

# Asmtest Parameters
asmtest-src         := $(source-dir)/asmtest
asmtest-bin         := $(asmtest-src)/bin
asmtest-dist-dir    := $(output-dir)/test-progs/asmtest/bin

# RISCV-Tests
.PHONY: riscv-tests
riscv-tests: $(riscv-benchmarks)
	-mkdir -p $(riscv-dist-dir)
	cp $(riscv-benchmarks) $(riscv-dist-dir)/

$(riscv-benchmarks):
	cd $(riscv-tests-dir) && autoconf && ./configure --prefix=/opt/riscv/target
	make -C "$(riscv-tests-dir)"

.PHONY: clean-riscv-tests
clean-riscv-tests:
	-rm -r $(riscv-dist-dir)
	-make -C $(riscv-tests-dir) clean

# Insttests
.PHONY: insttests
insttests: $(insttest-execs)
	-mkdir -p $(insttest-dist-dir)
	cp $(insttest-execs) $(insttest-dist-dir)/

$(insttest-execs):
	make -C $(insttest-src)

.PHONY: clean-insttests
clean-insttests:
	-rm -r $(insttest-dist-dir)
	-make -C $(insttest-src) clean

# PThreads
.PHONY: pthreads
pthreads: pthreads-x86 pthreads-aarch32 pthreads-aarch64 pthreads-riscv64

.PHONY: pthreads-x86
pthreads-x86: $(pthreads-x86-out)

$(pthreads-x86-out): $(pthreads-x86-dir) $(pthreads-out)
	cp -r $(pthreads-x86-dir) $(pthreads-x86-out)

$(pthreads-x86-dir): $(pthreads-dir) $(pthreads-dir)/$(pthreads-x86-make)
	cd $(pthreads-dir) && make -f $(pthreads-x86-make)

.PHONY: clean-pthreads-x86
clean-pthreads-x86:
	-rm -rf $(pthreads-x86-out)
	-cd $(pthreads-dir) && make -f $(pthreads-x86-make) clean

.PHONY: pthreads-aarch32
pthreads-aarch32: $(pthreads-a32-out)

$(pthreads-a32-out): $(pthreads-a32-dir) $(pthreads-out)
	cp -r $(pthreads-a32-dir) $(pthreads-a32-out)

$(pthreads-a32-dir): $(pthreads-dir) $(pthreads-dir)/$(pthreads-a32-make)
	cd $(pthreads-dir) && make -f $(pthreads-a32-make)

.PHONY: clean-pthreads-aarch32
clean-pthreads-aarch32:
	-rm -rf $(pthreads-a32-out)
	-cd $(pthreads-dir) && make -f $(pthreads-a32-make) clean

.PHONY: pthreads-aarch64
pthreads-aarch64: $(pthreads-a64-out)

$(pthreads-a64-out): $(pthreads-a64-dir) $(pthreads-out)
	cp -r $(pthreads-a64-dir) $(pthreads-a64-out)

$(pthreads-a64-dir): $(pthreads-dir) $(pthreads-dir)/$(pthreads-a64-make)
	cd $(pthreads-dir) && make -f $(pthreads-a64-make)

.PHONY: clean-pthreads-aarch64
clean-pthreads-aarch64:
	-rm -rf $(pthreads-x86-out)
	-cd $(pthreads-dir) && make -f $(pthreads-a64-make) clean

.PHONY: pthreads-riscv64
pthreads-riscv64: $(pthreads-ri5-out)

$(pthreads-ri5-out): $(pthreads-ri5-dir) $(pthreads-out)
	cp -r $(pthreads-ri5-dir) $(pthreads-ri5-out)

$(pthreads-ri5-dir): $(pthreads-dir) $(pthreads-dir)/$(pthreads-ri5-make)
	cd $(pthreads-dir) && make -f $(pthreads-ri5-make)

.PHONY: clean-pthreads-riscv64
clean-pthreads-riscv64:
	-rm -rf $(pthreads-ri5-out)
	-cd $(pthreads-dir) && make -f $(pthreads-ri5-make) clean

$(pthreads-out):
	-mkdir -p $(pthreads-out)

.PHONY: clean-pthreads
clean-pthreads: clean-pthreads-x86 \
				clean-pthreads-aarch32 \
				clean-pthreads-aarch64 \
				clean-pthreads-riscv64
	-rm -rf $(pthreads-out)

# Square
.PHONY: square
square: $(square-out)
	-cp $(square-bench) $(square-out)/

$(square-out): $(square-bin)
	mkdir -p $(square-out)

$(square-bin):
	make -C $(square-dir) gfx8-apu

.PHONY: clean-square
clean-square:
	-rm -r $(square-out)
	-make -C $(square-dir) clean

# Asmtest
.PHONY: asmtests
asmtests: $(asmtest-dist-dir)

$(asmtest-dist-dir): $(asmtest-bin)
	-mkdir -p $(asmtest-dist-dir)
	cp $(asmtest-bin)/* $(asmtest-dist-dir)/

$(asmtest-bin): $(asmtest-src)/Makefile
	make -C $(asmtest-src)

.PHONY: clean-asmtests
clean-asmtests:
	-make -C $(asmtest-src) clean
	-rm -rf $(asmtest-dist-dir)

# Global
.PHONY: all
all: riscv-tests insttests pthreads square asmtests

.PHONY: clean
clean: clean-riscv-tests clean-insttests clean-pthreads clean-square \
	   clean-asmtests
	-rm -r $(output-dir)

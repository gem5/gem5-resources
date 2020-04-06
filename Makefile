source-dir          := src

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
riscv-dist-dir      := dist/current/test-progs/riscv-tests

# Insttest Parameters
insttest-src        := $(source-dir)/insttest
insttest-bin        := $(insttest-src)/bin
insttest-execs      := $(insttest-bin)/insttest-rv64a \
	                   $(insttest-bin)/insttest-rv64c \
					   $(insttest-bin)/insttest-rv64d \
					   $(insttest-bin)/insttest-rv64f \
					   $(insttest-bin)/insttest-rv64i \
					   $(insttest-bin)/insttest-rv64m
insttest-dist-dir   := dist/current/test-progs/insttest/bin/riscv/linux

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
	make -C $(riscv-tests-dir) clean

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
	make -C $(insttest-src) clean

# Global
.PHONY: all
all: riscv-tests insttests

.PHONY: clean
clean: clean-riscv-tests clean-insttests
	-rm -r dist

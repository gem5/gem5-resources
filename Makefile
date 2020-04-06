riscv-tests-dir     := src/riscv-tests
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

.PHONY: riscv-tests
riscv-tests: $(riscv-benchmarks)
	mkdir -p dist/current/test-progs/riscv-tests
	cp $(riscv-benchmarks) dist/current/test-progs/riscv-tests/

.PHONY: clean-riscv-tests
clean-riscv-tests:
	rm -rf dist/current/test-progs/riscv-tests/*
	make -C $(riscv-tests-dir) clean

$(riscv-benchmarks):
	cd $(riscv-tests-dir) && autoconf && ./configure --prefix=/opt/riscv/target
	make -C "$(riscv-tests-dir)"

.PHONY: clean
clean: clean-riscv-tests

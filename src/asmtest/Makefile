#=======================================================================
# Makefile for riscv-tests/isa
#-----------------------------------------------------------------------

XLEN ?= 64

src_dir 	:= ./isa
bin_dir 	:= ./bin
dump_dir 	:= ./dump
bench_dir 	:= ./benchmarks

include $(src_dir)/rv64ui/Makefrag
include $(src_dir)/rv64uc/Makefrag
include $(src_dir)/rv64um/Makefrag
include $(src_dir)/rv64ua/Makefrag
include $(src_dir)/rv64uf/Makefrag
include $(src_dir)/rv64ud/Makefrag
include $(src_dir)/rv64uzfh/Makefrag
include $(src_dir)/rv64si/Makefrag
include $(src_dir)/rv64ssvnapot/Makefrag
include $(src_dir)/rv64mi/Makefrag
include $(src_dir)/rv64mzicbo/Makefrag
include $(src_dir)/rv64uamt/Makefrag
include $(src_dir)/rv64samt/Makefrag
include $(src_dir)/rv64ub/Makefrag
include $(src_dir)/rv32ui/Makefrag
include $(src_dir)/rv32uc/Makefrag
include $(src_dir)/rv32um/Makefrag
include $(src_dir)/rv32ua/Makefrag
include $(src_dir)/rv32uf/Makefrag
include $(src_dir)/rv32ud/Makefrag
include $(src_dir)/rv32uzfh/Makefrag
include $(src_dir)/rv32si/Makefrag
include $(src_dir)/rv32mi/Makefrag
include $(src_dir)/rv32uamt/Makefrag
include $(src_dir)/rv32samt/Makefrag
include $(src_dir)/rv32ub/Makefrag

include $(bench_dir)/Makefile

default: all

#--------------------------------------------------------------------
# Build rules
#--------------------------------------------------------------------

RISCV_PREFIX ?= riscv$(XLEN)-unknown-elf-
RISCV_GCC ?= $(RISCV_PREFIX)gcc
RISCV_GCC_OPTS ?= -static -mcmodel=medany -fvisibility=hidden -nostdlib -nostartfiles
RISCV_OBJDUMP ?= $(RISCV_PREFIX)objdump --disassemble-all --disassemble-zeroes --section=.text --section=.text.startup --section=.text.init --section=.data
RISCV_SIM ?= spike --isa=rv$(XLEN)gcv

vpath %.S $(src_dir)

#------------------------------------------------------------
# Build assembly tests

%.dump: %
	mkdir -p $(dump_dir)
	$(RISCV_OBJDUMP) $(bin_dir)/$< > $(dump_dir)/$@

%.out: %
	$(RISCV_SIM) --isa=rv64gc_zfh_zicboz_svnapot_zicntr --misaligned $< 2> $@

%.out32: %
	$(RISCV_SIM) --isa=rv32gc_zfh_zicboz_svnapot_zicntr --misaligned $< 2> $@

define compile_template

$$($(1)_p_tests): $(1)-p-%: $(1)/%.S
	mkdir -p $(bin_dir)
	$$(RISCV_GCC) $(2) $$(RISCV_GCC_OPTS) -I$(src_dir)/../env/p -I$(src_dir)/macros/scalar -T$(src_dir)/../env/p/link.ld $$< -o $(bin_dir)/$$@
$(1)_p_env_tests += $$($(1)_p_tests)

$$($(1)_ps_tests): $(1)-ps-%: $(1)/%.S
	mkdir -p $(bin_dir)
	$$(RISCV_GCC) $(2) $$(RISCV_GCC_OPTS) -I$(src_dir)/../env/ps -I$(src_dir)/macros/scalar -I$(src_dir)/macros/mt -T$(src_dir)/../env/ps/link.ld $$< -o $(bin_dir)/$$@
$(1)_ps_env_tests += $$($(1)_ps_tests)

$$($(1)_v_tests): $(1)-v-%: $(1)/%.S
	mkdir -p $(bin_dir)
	$$(RISCV_GCC) $(2) $$(RISCV_GCC_OPTS) -DENTROPY=0x$$(shell echo \$$@ | md5sum | cut -c 1-7) -std=gnu99 -O2 -I$(src_dir)/../env/v -I$(src_dir)/macros/scalar -T$(src_dir)/../env/v/link.ld $(src_dir)/../env/v/entry.S $(src_dir)/../env/v/*.c $$< -o $(bin_dir)/$$@
$(1)_v_env_tests += $$($(1)_v_tests)

$(1)_tests_dump = $$(addsuffix .dump, $$($(1)_tests))

$(1): $$($(1)_tests_dump)

.PHONY: $(1)

p_env_tests 	+= $$($(1)_p_env_tests)
v_env_tests 	+= $$($(1)_v_env_tests)
ps_env_tests 	+= $$($(1)_ps_env_tests)

endef

$(eval $(call compile_template,rv32ui,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32uc,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32um,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32ua,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32uf,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32ud,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32uzfh,-march=rv32g_zfh -mabi=ilp32))
$(eval $(call compile_template,rv32si,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32mi,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32uamt,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32samt,-march=rv32g -mabi=ilp32))
$(eval $(call compile_template,rv32ub,-march=rv32g_zba_zbb_zbc_zbs -mabi=ilp32))
$(eval $(call compile_template,rv64ui,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64uc,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64um,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64ua,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64uf,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64ud,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64uzfh,-march=rv64g_zfh -mabi=lp64))
$(eval $(call compile_template,rv64mzicbo,-march=rv64g_zicboz -mabi=lp64))
$(eval $(call compile_template,rv64si,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64ssvnapot,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64mi,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64uamt,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64samt,-march=rv64g -mabi=lp64))
$(eval $(call compile_template,rv64ub,-march=rv64g_zba_zbb_zbc_zbs -mabi=lp64))

p_env_tests_dump = $(addsuffix .dump, $(p_env_tests))
ps_env_tests_dump = $(addsuffix .dump, $(ps_env_tests))
v_env_tests_dump = $(addsuffix .dump, $(v_env_tests))

#------------------------------------------------------------
# Targets

all: 	p ps v
# build tests with p environment
p:		$(p_env_tests_dump)
# build tests with ps environment
ps: 	$(ps_env_tests_dump)
# build tests with v environment
v:		$(v_env_tests_dump)

#------------------------------------------------------------
# Clean up

clean:
	-rm -rf $(bin_dir) $(dump_dir)

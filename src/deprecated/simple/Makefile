# Copyright (c) 2020 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Input files for each build type. We whitelist barmetal ones,
# and blacklist
BARE_INS = \
  m5_exit \
  m5_checkpoint
USER_INS = $(basename $(foreach BASE_DIR, . $(ISA), $(foreach IN_EXT, \
  $(IN_EXT_ASM) $(IN_EXT_C) $(IN_EXT_CXX), $(wildcard $(BASE_DIR)/*$(IN_EXT)))))

# Input values.
BOOTLOADER_SRC = bootloader/$(ISA)$(IN_EXT_ASM)
BOOTLOADER_OBJ = $(OUT_DIR)/bootloader$(OBJ_EXT)
CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
GEM5_ROOT = $(CURDIR)/../../../gem5
M5OP_OBJ = $(GEM5_ROOT)/util/m5/build/$(M5_ISA)/out/libm5.a
# Common to USER and BARE.
CFLAGS = $(CCFLAGS) -std=c11
# Common to C and C++.
CCFLAGS = \
  -I $(GEM5_ROOT)/include/ \
  -I $(GEM5_ROOT)/util/m5/src/ \
  -O3 \
  -Wall \
  -Werror \
  -Wextra \
  -fno-pie \
  -ggdb3 \
  -no-pie \
  -pedantic \
  -static \
  $(CCFLAGS_ISA)
# -Wl,--whole-archive allows C++ threads to work with -static.
# https://stackoverflow.com/questions/35116327/when-g-static-link-pthread-cause-segmentation-fault-why
LDFLAGS_USER = \
  -lrt \
  -pthread \
  -Wl,--whole-archive \
  -lpthread \
  -Wl,--no-whole-archive
CCFLAGS_BARE = $(CFLAGS) \
  -T link.ld \
  -Wl,--section-start=.text=0x80000000 \
  -fno-asynchronous-unwind-tables \
  -fno-unwind-tables \
  -nostartfiles \
  -nostdlib
CFLAGS_USER = $(CFLAGS)
CXXFLAGS = $(CCFLAGS) -std=c++11
CXXFLAGS_BARE = $(CXXFLAGS)
CXXFLAGS_USER = $(CXXFLAGS)
IN_EXT_ASM = .S
IN_EXT_C = .c
IN_EXT_CXX = .cpp
OUT_DIR_BASE = out
OUT_DIR = $(OUT_DIR_BASE)/$(ISA)
OUT_BARE_DIR = $(OUT_DIR)/bare
OUT_SQUASHFS_DIR = $(OUT_DIR)/squashfs
OUT_SQUASHFS_EXT = .squashfs
OUT_USER_DIR = $(OUT_DIR)/user
OBJ_EXT = .o
OUT_EXT = .out

ifeq ($(ISA),)
ISA = $(shell uname -m)
endif

ifeq ($(ISA),x86_64)
CCFLAGS_ISA = -msse3
M5_ISA = x86
PREFIX =
else
ifeq ($(ISA),aarch64)
CCFLAGS_ISA = -Xassembler -march=all
M5_ISA = arm64
# Input files just for this ISA.
ISA_BARE_INS = semihost_exit
# Exclude baremetal only builds.
USER_INS := $(filter-out $(ISA)/semihost_exit, $(USER_INS))
# Commented out for now because otherwise adds an
# empty entry to the target list.
#ISA_USER_INS =
PREFIX = aarch64-linux-gnu-
else
M5_ISA = $(ISA)
ifeq ($(ISA),arm)
PREFIX = arm-linux-gnueabihf-
else
ifeq ($(ISA),riscv)
PREFIX = riscv64-linux-gnu-
# riscv does not have an util/m5/m5op_
M5OP_OBJ =
USER_INS := $(filter-out ./m5_checkpoint ./m5_exit, $(USER_INS))
else
ifeq ($(ISA),sparc64)
PREFIX = sparc64-linux-
# sparc64 does not have an util/m5/m5op_
M5OP_OBJ =
# At present sparc64 only compiles the pthread binaries. Other binaries have
# not been tested.
USER_INS := $(filter-out ./m5_checkpoint \
                         ./m5_exit \
                         ./chdir_print \
                         ./hello \
                         ./hellp_cpp \
                         ./m5_exit_addr \
                         ./page_access_wrap \
                         ./stack_print \
                         ./std_thread, $(USER_INS))
else
$(error unknown isa $(ISA))
endif
endif
endif
endif
endif

# Calculated values.
BARE_OUTS = $(addprefix $(OUT_BARE_DIR)/, $(addsuffix $(OUT_EXT), $(BARE_INS) $(addprefix $(ISA)/,$(ISA_BARE_INS))))
SQUASHFS_OUTS = $(addprefix $(OUT_USER_DIR)/, $(addsuffix $(OUT_EXT), $(USER_INS)))
USER_OUTS = $(addprefix $(OUT_USER_DIR)/, $(addsuffix $(OUT_EXT), $(USER_INS)))

.PHONY: all bare clean mkdir user

user: mkdir-user $(USER_OUTS)

bare: mkdir-bare $(BARE_OUTS)

all: bare user

user-show:
	@echo $(USER_OUTS)

$(OUT_BARE_DIR)/%$(OUT_EXT): %$(IN_EXT_ASM) $(M5OP_OBJ)
	$(CC) $(CCFLAGS_BARE) -o '$@' $^

$(OUT_USER_DIR)/%$(OUT_EXT): %$(IN_EXT_ASM) $(M5OP_OBJ)
	$(CC) $(CFLAGS_USER) -o '$@' $^

$(OUT_BARE_DIR)/%$(OUT_EXT): %$(IN_EXT_C) $(M5OP_OBJ) $(BOOTLOADER_OBJ)
	$(CC) $(CCFLAGS_BARE) -o '$@' $^

$(OUT_USER_DIR)/%$(OUT_EXT): %$(IN_EXT_C) $(M5OP_OBJ)
	$(CC) $(CFLAGS_USER) -o '$@' $^ $(LDFLAGS_USER)

$(OUT_SQUASHFS_DIR)/%$(OUT_SQUASHFS_EXT): $(OUT_USER_DIR)/%$(OUT_EXT)
	@# Create dev so that the kernel can automount CONFIG_DEVTMPFS=y.
	mkdir -p '$(basename $@)/sbin' '$(basename $@)/dev'
	cp '$<' '$(basename $@)/sbin/init'
	@# Existing squashfs files don't get updated otherwise.
	rm -f '$@'
	mksquashfs '$(basename $@)' '$@'

$(OUT_BARE_DIR)/%$(OUT_EXT): %$(IN_EXT_CXX) $(M5OP_OBJ) $(BOOTLOADER_OBJ)
	$(CXX) $(CXXFLAGS_BARE) -o '$@' $^

$(OUT_USER_DIR)/%$(OUT_EXT): %$(IN_EXT_CXX) $(M5OP_OBJ)
	$(CXX) $(CXXFLAGS_USER) -o '$@' $^ $(LDFLAGS_USER)

$(BOOTLOADER_OBJ): $(BOOTLOADER_SRC)
	$(CC) $(CFLAGS_USER) -c -o '$@' '$<'

$(M5OP_OBJ):
	cd $(GEM5_ROOT) && scons -C util/m5 build/$(M5_ISA)/out/m5

clean:
	rm -rf '$(OUT_DIR_BASE)'

mkdir-bare:
	mkdir -p '$(OUT_BARE_DIR)/$(ISA)'

mkdir-user:
	mkdir -p '$(OUT_USER_DIR)/$(ISA)'

EXECUTABLE := allSyncPrims-1kernel
HIP_PATH ?= /opt/rocm/hip

SRC_DIR := src
SRC := $(wildcard $(SRC_DIR)/*.hip.cpp)

BIN_DIR := bin

all: release

release: $(SRC) | $(BIN_DIR)
	$(HIP_PATH)/bin/hipcc -DGFX9 --amdgpu-target=gfx900,gfx902 $(SRC) -o $(BIN_DIR)/$(EXECUTABLE)

# gfx8 has a different number of bits it uses for sleeps, so compile accordingly
release-gfx8: $(SRC) | $(BIN_DIR)
	$(HIP_PATH)/bin/hipcc --amdgpu-target=gfx803,gfx801 $(SRC) -o $(BIN_DIR)/$(EXECUTABLE)

debug: $(SRC) | $(BIN_DIR)
	$(HIP_PATH)/bin/hipcc -DDEBUG -g -O0 $(SRC) -o $(BIN_DIR)/$(EXECUTABLE).debug

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -f $(BIN_DIR)

BIN_DIR ?= ./bin
HIP_PATH ?= /opt/rocm/hip

all: $(BIN_DIR)
	$(HIP_PATH)/bin/hipcc src/lulesh.hip.cc -o $(BIN_DIR)/lulesh --amdgpu-target=gfx801,gfx803,gfx900,gfx902

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

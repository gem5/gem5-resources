BIN_DIR?= ./bin

all: $(BIN_DIR)
	hipcc src/lulesh.hip.cc -o $(BIN_DIR)/lulesh --amdgpu-target=gfx801

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

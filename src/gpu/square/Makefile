HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin

square: $(BIN_DIR)/square

$(BIN_DIR)/square: square.cpp $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx900,gfx902 $(CXXFLAGS) square.cpp -o $(BIN_DIR)/square

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: square clean

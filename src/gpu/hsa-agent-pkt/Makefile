HIP_PATH?= /opt/rocm/hip
HSA_PATH?= /opt/rocm/hsa
HIPCC=$(HIP_PATH)/bin/hipcc

BIN_DIR?= ./bin

gfx8-apu: $(BIN_DIR)/square.o

$(BIN_DIR)/square.o: square.cpp HSA_Interface.cpp $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx801 $(CXXFLAGS) -I$(HSA_PATH) square.cpp HSA_Interface.cpp -o $(BIN_DIR)/square.o

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(BIN_DIR)

.PHONY: gfx8-apu clean

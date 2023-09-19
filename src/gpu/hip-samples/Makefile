HIP_PATH?= /opt/rocm/hip
HIPCC=$(HIP_PATH)/bin/hipcc

SRC_DIR := src
BIN_DIR := bin

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
EXECUTABLES := $(basename $(notdir $(SOURCES)))

.PHONY: all clean

all: $(EXECUTABLES)

$(EXECUTABLES): %: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(HIPCC) --amdgpu-target=gfx801,gfx803,gfx900,gfx902 $< -o $(BIN_DIR)/$@

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BIN_DIR)

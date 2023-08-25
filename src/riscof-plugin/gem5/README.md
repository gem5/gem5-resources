### Riscof Plugin for GEM5 Simulator

- Using Riscof for GEM5 Simulator

- The GEM5 implements RISC-V ISA instructions. It can be used as a target for
  running tests using riscof.

- Config file entry
```
# Name of the Plugin. All entries are case sensitive. Including the name.
[gem5]

#path to the directory where the plugin is present. (required)
pluginpath=<path-to-plugin>

#path to the ISA config file. (required)
ispec=<path-to-isa-config>

#path to the Platform config file. (required)
pspec=<path-to-platform-config>

# gem5 repository path. (required)
gem5path=<path-to-gem5-repository>

# python config script for gem5 DUT. (required)
configpath=<path-to-config-for-gem5>

# arguments for config script. (required)
configargs=<arguments-for-gem5-config>

#number of jobs to spawn in parallel (optional)
jobs=1

#the make command to use while executing the make file. Any of bmake,cmake,pmake,make etc. (optional)
#Default is make
make=make

# build_opts for building gem5, available are listed in <gem5-repo>/build_opts (optional)
build_opts=RISCV

# require build gem5 binary in gem5 repository (optional)
required_build=False

# using docker to build and run gem5 binary (required)
docker=False

# the docker image for gem5 (required if docker=True)
image=gcr.io/gem5-test/ubuntu-22.04_all-dependencies:latest
```

- Export command
```
pwd=pwd;export PYTHONPATH=$pwd:$PYTHONPATH;
```

- `gem5_isa.yaml`

```yaml
hart_ids: [0]
hart0:
  ISA: RV64IMCZicsr_Zifencei # String represents of ISA would like to verify.
  physical_addr_sz: 39 # Size of the physical address
  User_Spec_Version: '2.3' # Version of ISA spec
  supported_xlen: [64] # List of supported xlen on the target
```

See: https://riscv-config.readthedocs.io/en/stable/yaml-specs.html#isa-yaml-spec

- gem5_platform.yaml

See: https://riscv-config.readthedocs.io/en/stable/yaml-specs.html#platform-yaml-spec

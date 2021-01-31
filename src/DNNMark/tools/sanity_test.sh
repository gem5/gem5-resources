#! /bin/bash

# Setup directories
WORK_DIR="$(pwd)"/..
CONFIG_DIR=${WORK_DIR}/config_example/
BENCHMARK_DIR=${WORK_DIR}/build/benchmarks/

BENCHMARK_LIST="$(ls ${BENCHMARK_DIR} | grep test*)"
CONFIG_LIST="$(ls ${CONFIG_DIR})"
PROFILER=nvprof

for bm in ${BENCHMARK_LIST[@]}
do
  EXE="$(find ${BENCHMARK_DIR}${bm} -executable -type f)"
  echo $bm
  trimed_bm="$(echo $bm | cut -d "_" -f2)"
  if [ ${trimed_bm} == "fwd" ] || [ ${trimed_bm} == "bwd" ]; then
    trimed_bm="$(echo $bm | cut -d "_" -f3)"
  fi
  for config in ${CONFIG_LIST[@]}
  do
    if [[ $config == *"$trimed_bm"* ]]; then
      echo "Configure file: " $config
      echo "${EXE} -config ${config} -debuginfo 1 -warmup 1"
      ${EXE} -config ${CONFIG_DIR}${config} -debuginfo 1 -warmup 1
    fi
  done
done



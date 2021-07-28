#! /bin/sh

if [ $# -ne 1 ]
then
  echo "[Error] The setup script requires one additional parameter specifying whether CUDA or HCC is used"
  echo "Options: [CUDA, HIP]"
  exit
fi

OPTION=$1

BUILD_DIR=build
if [ ! -d ${BUILD_DIR} ]; then
  mkdir ${BUILD_DIR}
fi
cd ${BUILD_DIR}

if [ ${OPTION} = "CUDA" ]
then
  CUDNN_PATH=${HOME}/cudnn
  cmake -DCUDA_ENABLE=ON -DCUDNN_ROOT=${CUDNN_PATH} ..
elif [ ${OPTION} = "HIP" ]
then
  MIOPEN_PATH=/opt/rocm/miopen
  ROCBLAS_PATH=/opt/rocm/rocblas
  CXX=/opt/rocm/bin/hipcc cmake \
    -DHCC_ENABLE=ON \
    -DMIOPEN_ROOT=${MIOPEN_PATH} \
    -DROCBLAS_ROOT=${ROCBLAS_PATH} \
    -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/lib/cmake/AMDDeviceLibs/;/opt/rocm/lib/cmake/amd_comgr/" \
    ..
fi

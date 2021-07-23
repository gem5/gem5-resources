# CMake version related
cmake_minimum_required(VERSION 2.8.9)
if (${CMAKE_VERSION} VERSION_GREATER 3.0)
  cmake_policy(SET CMP0046 OLD)
endif()

# Set project
project(dnnmark)

# Set path of sub-directory with other cmake file
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Set
option (CUDA_ENABLE "Only run benchmark on Nvidia device" OFF)
option (HCC_ENABLE "Only run benchmark on AMD device" OFF)

# Locate CUDA package
find_package(CUDA)

# Detect GPU device
if(CUDA_FOUND AND ${CUDA_ENABLE})
  try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/tools/has_cuda_gpu.c
    CMAKE_FLAGS
      -DINCLUDE_DIRECTORIES:STRING=${CUDA_TOOLKIT_INCLUDE}
      -DLINK_LIBRARIES:STRING=${CUDA_CUDART_LIBRARY}
    COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
    RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR)

  # Display number of GPUs found
  message("${RUN_OUTPUT_VAR}")

  # COMPILE_RESULT_VAR is TRUE when compile succeeds
  # RUN_RESULT_VAR is zero when a GPU is found
  if(COMPILE_RESULT_VAR AND NOT RUN_RESULT_VAR)
    set(CUDA_HAVE_GPU TRUE)
  else()
    set(CUDA_HAVE_GPU FALSE)
  endif()
endif()

# Detect HIPCC
find_program(HIPCC_FOUND hipcc)

option (double-test "Make data type double" OFF)
option (enable-cudnnv6 "Enable cuDNN version 6" OFF)

# Build NICE library only with CUDA
if (CUDA_FOUND AND ${CUDA_ENABLE})

  # Print messge to indicate whether GPU device exists
  if (CUDA_HAVE_GPU)
    message(STATUS "GPU device exists")
    add_definitions(-DNVIDIA_CUDNN)
  else()
    message(WARNING "GPU device does not exist")
  endif()

  #include the built-in findCUDA cmake scripts
  include(FindCUDA)

  # Stop if CUDNN_ROOT is not defined
  if (NOT DEFINED CUDNN_ROOT)
    message(FATAL_ERROR "Path to CuDNN is not defined")
  endif()

  if (EXISTS ${CUDNN_ROOT})
    message(STATUS "CuDNN library root path: " ${CUDNN_ROOT})
    include_directories(${CUDNN_ROOT}/include)
  else()
    message(FATAL_ERROR "Path to CuDNN library not exists")
  endif()

  # Enable double data type
  if (double-test)
    add_definitions(-DDOUBLE_TEST)
  endif()

  # Add deinition of C++11 stardard
  add_definitions(-std=c++11)

  # Enable cuDNN v6
  if (enable-cudnnv6)
    add_definitions(-DCUDNNV6)
  endif()

  # Set path of DNNMark include files
  set(DNNMARK_INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
  include_directories(${DNNMARK_INCLUDES})

  # Set path of DNNMark layer files
  set(DNNMARK_LAYERS ${DNNMARK_INCLUDES}/layers)
  include_directories(${DNNMARK_LAYERS})

  # Set source files
  file(GLOB_RECURSE DNNMARK_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} core/src/*.cc)
  message(STATUS "DNNMark Source files: " ${DNNMARK_SOURCES})

  # Set NVCC flags
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_50,code=sm_50)

  # Find cuda libraries
  find_library(CUDNN_LIBRARY cudnn ${CUDNN_ROOT}/lib64)
  find_library(CUDA_BLAS_LIBRARY cublas HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  find_library(CUDA_RAND_LIBRARY curand HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

  # Find glog library
  find_library(GLOG_LIBRARY glog)

  # Add NICE library together with CUDA
  include_directories(${CUDA_INCLUDE_DIR})
  cuda_add_library(${PROJECT_NAME} SHARED ${DNNMARK_SOURCES})
  add_dependencies(${PROJECT_NAME} ${CUDNN_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_BLAS_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_RAND_LIBRARY})
  add_dependencies(${PROJECT_NAME} ${CUDA_LIBRARIES})
  add_dependencies(${PROJECT_NAME} ${GLOG_LIBRARY})
  target_link_libraries(${PROJECT_NAME}
                        ${CUDNN_LIBRARY}
                        ${CUDA_BLAS_LIBRARY}
                        ${CUDA_RAND_LIBRARY}
                        ${CUDA_LIBRARIES}
                        ${GLOG_LIBRARY}
                        m)

elseif(HIPCC_FOUND AND ${HCC_ENABLE})

  # Cover the include and linkage requirement here
  execute_process(COMMAND hcc-config  --cxxflags
                  OUTPUT_VARIABLE HCC_COMPILE_FLAGS)
  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${HCC_COMPILE_FLAGS}")

  execute_process(COMMAND hcc-config  --ldflags
                  OUTPUT_VARIABLE HCC_LINKER_FLAGS)
  set(LINKER_FLAGS "${LINKER_FLAGS} ${HCC_LINKER_FLAGS}")

  set(ROCM_ROOT /opt/rocm)
  if (EXISTS ${MIOPEN_ROOT})
    message(STATUS "MIOpen root path: " ${MIOPEN_ROOT})
    include_directories(${MIOPEN_ROOT}/include)
    include_directories(${MIOPEN_ROOT}/src/include)
    include_directories(${MIOPEN_ROOT}/build/include)
    include_directories(${ROCM_ROOT}/hip/include)
  else()
    message(FATAL_ERROR "Path to MIOpen library not exists")
  endif()

  if (EXISTS ${ROCBLAS_ROOT})
    message(STATUS "ROCBLAS root path: " ${ROCBLAS_ROOT})
    include_directories(${ROCBLAS_ROOT}/include)
  else()
    message(FATAL_ERROR "Path to ROCBLAS library not exists")
  endif()

  # Locate HIP to this project
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_ROOT}/hcc ${ROCM_ROOT}/hip)
  find_package(hip REQUIRED PATHS ${ROCM_ROOT})

  # Add a guard for AMD MIOpen
  add_definitions(-DAMD_MIOPEN)

  # Add deinition of C++11 stardard
  add_definitions(-std=c++11 -Wno-attributes)

  # Enable double data type
  if (double-test)
    add_definitions(-DDOUBLE_TEST)
  endif()

  # Set path of DNNMark include files
  set(DNNMARK_INCLUDES ${CMAKE_SOURCE_DIR}/core/include)
  include_directories(${DNNMARK_INCLUDES})

  # Set path of DNNMark layer files
  set(DNNMARK_LAYERS ${DNNMARK_INCLUDES}/layers)
  include_directories(${DNNMARK_LAYERS})

  # Set source files
  file(GLOB_RECURSE DNNMARK_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} core/src/*.cc)
  message(STATUS "DNNMark Source files: " ${DNNMARK_SOURCES})

  # Find rocBLAS libraries
  find_library(ROCBLAS_LIBRARY rocblas ${ROCBLAS_ROOT}/lib)

  # Find MIOpen libraries
  find_library(MIOPEN_LIBRARY MIOpen ${MIOPEN_ROOT}/lib)

  message(${ROCBLAS_LIBRARY} ${MIOPEN_LIBRARY})

  # Find glog libraries
  find_library(GLOG_LIBRARY glog)

  # Add DNNMark library
  add_library(${PROJECT_NAME} SHARED ${DNNMARK_SOURCES})
  add_dependencies(${PROJECT_NAME} ${MIOPEN_LIBRARY})
  target_link_libraries(${PROJECT_NAME}
                        ${ROCBLAS_LIBRARY}
                        ${MIOPEN_LIBRARY}
                        ${GLOG_LIBRARY}
                        m)
  set_target_properties(${PROJECT_NAME} PROPERTIES
      COMPILE_FLAGS ${COMPILER_FLAGS}
      LINK_FLAGS ${LINKER_FLAGS})

else()

  message(FATAL_ERROR "CUDA or HCC not exists")

endif()

# Benchmarking
option (enable-benchmarking "Compile all benchmarks of DNNMark" ON)

if (enable-benchmarking)
  # Include the cmake file of gflag
  find_library(GFLAGS_LIBRARY gflags)

  # Invoke cmake files of each benchmark
  set(BENCHMARKS benchmarks)
  add_subdirectory(${BENCHMARKS})

endif()

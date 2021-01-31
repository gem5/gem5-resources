# Include the DNNMark include path
include_directories(${DNNMARK_INCLUDES})

# Obtain the name of the directory(also part of the benchmark name)
get_filename_component(BENCHMARK_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Set the benchmark name
set(BENCHMARK_NAME ${PROJECT_NAME}_${BENCHMARK_DIR})

# Add source file
file(GLOB_RECURSE BENCHMARK_SRC_FILE RELATIVE
     ${CMAKE_CURRENT_SOURCE_DIR}
     *.cc)

# Print out the source file information
message(STATUS ${BENCHMARK_NAME} ": Benchmark source files: " ${BENCHMARK_SRC_FILE})

# Add benchmark executable
add_executable(${BENCHMARK_NAME} ${COMMON_SRC} ${BENCHMARK_SRC_FILE})

# Add dependencies of current benchmark
add_dependencies(${BENCHMARK_NAME} ${PROJECT_NAME})
add_dependencies(${BENCHMARK_NAME} ${GFLAGS_LIBRARY})

# Link libraries
target_link_libraries(${BENCHMARK_NAME}
    ${PROJECT_NAME}
    ${GFLAGS_LIBRARY})


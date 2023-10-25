FROM ghcr.io/gem5/gcn-gpu:latest
RUN apt-get update && apt-get -y install libopenmpi-dev libomp-dev

ENV HCC_AMDGPU_TARGET="gfx801,gfx803,gfx900"

ENV HIPCC_BIN=/opt/rocm/bin
ENV MPI_INCLUDE=/usr/lib/x86_64-linux-gnu/openmpi/include

ENV OPT="-O3 -g -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO"
ENV OMP="-I/usr/lib/llvm-10/include/openmp -L/usr/lib/llvm-10/lib -fopenmp"

ENV HIPCC_FLAGS="-v -ffast_math -DINLINE_FORCE -I${MPI_INCLUDE}"
ENV HIPCC_FLAGS="-v -I${MPI_INCLUDE} -I/opt/rocm/hip/include"

ENV HACC_PLATFORM="hip"
ENV HACC_OBJDIR="${HACC_PLATFORM}"

ENV HACC_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_CC="${HIPCC_BIN}/hipcc -x c -Xclang -std=c99"

ENV HACC_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_CXX="${HIPCC_BIN}/hipcc -Xclang"

ENV HACC_LDFLAGS="-lm -lrt"

# USE_SERIAL_COSMO must be set to avoid building the code with MPI, which isn't
# supported on the GPU model in gem5.
ENV USE_SERIAL_COSMO="1"
ENV HACC_NUM_CUDA_DEV="1"
ENV HACC_MPI_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_MPI_CC="${HIPCC_BIN}/hipcc -x c -Xclang -std=c99 -Xclang -pthread"

ENV HACC_MPI_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_MPI_CXX="${HIPCC_BIN}/hipcc -Xclang -pthread"
ENV HACC_MPI_LD="${HIPCC_BIN}/hipcc -Xclang -pthread"

ENV HACC_MPI_LDFLAGS="-lm -lrt"

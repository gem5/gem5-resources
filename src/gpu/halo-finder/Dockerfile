FROM gcr.io/gem5-test/gcn-gpu
RUN apt-get update && apt-get -y install libopenmpi-dev libomp-dev

ENV HIPCC_BIN=/opt/rocm/bin
ENV MPI_INCLUDE=/usr/lib/openmpi/include

ENV OPT="-O3 -g -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO"
ENV OMP="-fopenmp"

ENV HIPCC_FLAGS="-v -ffast_math -DINLINE_FORCE -I${MPI_INCLUDE}"
ENV HIPCC_FLAGS="-v -I${MPI_INCLUDE} -I/opt/rocm/hip/include -I/opt/rocm/hcc-1.0/include"

ENV HACC_PLATFORM="hip"
ENV HACC_OBJDIR="${HACC_PLATFORM}"

ENV HACC_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_CC="${HIPCC_BIN}/hcc -x c -Xclang -std=c99"

ENV HACC_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_CXX="${HIPCC_BIN}/hipcc -Xclang"

ENV HACC_LDFLAGS="-lm -lrt"

# USE_SERIAL_COSMO must be set to avoid building the code with MPI, which isn't
# supported on the GPU model in gem5.
ENV USE_SERIAL_COSMO="1"
ENV HACC_NUM_CUDA_DEV="1"
ENV HACC_MPI_CFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_MPI_CC="${HIPCC_BIN}/hcc -x c -Xclang -std=c99 -Xclang -pthread"

ENV HACC_MPI_CXXFLAGS="$OPT $OMP $HIPCC_FLAGS"
ENV HACC_MPI_CXX="${HIPCC_BIN}/hipcc -Xclang -pthread"
ENV HACC_MPI_LD="${HIPCC_BIN}/hipcc -Xclang -pthread"

ENV HACC_MPI_LDFLAGS="-lm -lrt"

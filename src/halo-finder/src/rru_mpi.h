#ifndef RRU_MPI_H
#define RRU_MPI_H

//#ifndef USE_VTK_COSMO
// Needed for some versions of MPI which define these
//#undef SEEK_SET
//#undef SEEK_CUR
//#undef SEEK_END
//#endif

#ifndef MPICH_IGNORE_CXX_SEEK
#define MPICH_IGNORE_CXX_SEEK
#endif
#include <mpi.h>

#endif

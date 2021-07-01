#ifndef DISTRIBUTION_H
#define DISTRIBUTION_H

#include "complex-type.h"
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

///
// descriptor for a process grid
//   cart     Cartesian MPI communicator
//   nproc[]  dimensions of process grid
//   period[] periods of process grid
//   self[]   coordinate of this process in the process grid
//   n[]      local grid dimensions
///
typedef struct {
    MPI_Comm cart;
    int nproc[3];
    int period[3];
    int self[3];
    int n[3];
} process_topology_t;


///
// descriptor for data distribution
//   debug               toggle debug output
//   n[3]                (global) grid dimensions
//   padding[3]          padding applied to (local) arrays
//   process_topology_1  1-d process topology
//   process_topology_2  2-d process topology
//   process_topology_3  3-d process topology
///
typedef struct {
    bool debug;
    int n[3];
    int padding[3];
    process_topology_t process_topology_1;
    process_topology_t process_topology_2_z;
    process_topology_t process_topology_2_y;
    process_topology_t process_topology_2_x;
    process_topology_t process_topology_3;
    complex_t *d2_chunk;
    complex_t *d3_chunk;
} distribution_t;


///
// create 1-, 2- and 3-d cartesian data distributions
//   comm   MPI Communicator
//   d      distribution descriptor
//   n      (global) grid dimensions (3 element array)
//   debug  debugging output
///
void distribution_init(MPI_Comm comm, const int n[], const int n_padded[], distribution_t *d, bool debug);


///
// create 1-, 2- and 3-d cartesian data distributions with explicitly
// provided dimension lists
//   comm       MPI Communicator
//   n          (global) grid dimensions (3 element array)
//   n_padded   padded grid dimensions (3 element array)
//   nproc_1d   1d process grid (3 element array: x, 1, 1)
//   nproc_2d   1d process grid (3 element array: x, y, 1)
//   nproc_3d   3d process grid (3 element array: x, y, z)
//   d          distribution descriptor
//   debug      debugging output
///
void distribution_init_explicit(MPI_Comm comm,
                                const int n[],
                                const int n_padded[],
                                int nproc_1d[],
                                int nproc_2d_x[],
                                int nproc_2d_y[],
                                int nproc_2d_z[],
                                int nproc_3d[],
                                distribution_t *d,
                                bool debug);

///
// clean up the data distribution
//   d    distribution descriptor
///
void distribution_fini(distribution_t *d);


///
// assert that the data and processor grids are commensurate
//   d    distribution descriptor
///
void distribution_assert_commensurate(distribution_t *d);


///
// redistribute a 1-d to a 3-d data distribution
//   a    input
//   b    ouput
//   d    distribution descriptor
///
void distribution_1_to_3(const complex_t *a,
                         complex_t *b,
                         distribution_t *d);

///
// redistribute a 3-d to a 1-d data distribution
//   a    input
//   b    ouput
//   d    distribution descriptor
///
void distribution_3_to_1(const complex_t *a,
                         complex_t *b,
                         distribution_t *d);

///
// redistribute a 2-d to a 3-d data distribution
//   a    input
//   b    ouput
//   d    distribution descriptor
///
void distribution_2_to_3(const complex_t *a,
                         complex_t *b,
                         distribution_t *d, int dim_z);

///
// redistribute a 3-d to a 2-d data distribution
//   a    input
//   b    ouput
//   d    distribution descriptor
///
void distribution_3_to_2(const complex_t *a,
                         complex_t *b,
                         distribution_t *d, int dim_z);


///
// Some accessor functions
///
static inline int distribution_get_nproc_1d(distribution_t *d, int direction)
{
    return d->process_topology_1.nproc[direction];
}

static inline int distribution_get_nproc_2d_x(distribution_t *d, int direction)
{
    return d->process_topology_2_x.nproc[direction];
}
static inline int distribution_get_nproc_2d_y(distribution_t *d, int direction)
{
    return d->process_topology_2_y.nproc[direction];
}
static inline int distribution_get_nproc_2d_z(distribution_t *d, int direction)
{
    return d->process_topology_2_z.nproc[direction];
}

static inline int distribution_get_nproc_3d(distribution_t *d, int direction)
{
    return d->process_topology_3.nproc[direction];
}

static inline int distribution_get_self_1d(distribution_t *d, int direction)
{
    return d->process_topology_1.self[direction];
}

static inline int distribution_get_self_2d_x(distribution_t *d, int direction)
{
    return d->process_topology_2_x.self[direction];
}
static inline int distribution_get_self_2d_y(distribution_t *d, int direction)
{
    return d->process_topology_2_y.self[direction];
}
static inline int distribution_get_self_2d_z(distribution_t *d, int direction)
{
    return d->process_topology_2_z.self[direction];
}
static inline int distribution_get_self_3d(distribution_t *d, int direction)
{
    return d->process_topology_3.self[direction];
}

void Coord_x_pencils(int myrank, int coord[], distribution_t *d);
void Rank_x_pencils(int * myrank, int coord[], distribution_t *d);
void Coord_y_pencils(int myrank, int coord[], distribution_t *d);
void Rank_y_pencils(int * myrank, int coord[], distribution_t *d);
void Coord_z_pencils(int myrank, int coord[], distribution_t *d);
void Rank_z_pencils(int * myrank, int coord[], distribution_t *d);

#ifdef __cplusplus
}
#endif

#endif

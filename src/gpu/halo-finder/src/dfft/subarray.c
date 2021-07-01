// This is an implementation of MPI subarray's in terms of MPI struct
// used to work around a potential bug in Open MPI's implementation.

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int MPI_Type_create_subarray(int ndims,
                             int array_of_sizes[],
                             int array_of_subsizes[],
                             int array_of_starts[],
                             int order,
                             MPI_Datatype oldtype,
                             MPI_Datatype *newtype)
{
    MPI_Datatype *t;
    MPI_Aint lb;
    MPI_Aint extent;

    t = malloc((ndims + 1) * sizeof(MPI_Datatype));
    if (!t) {
        perror("out of memory");
    }
    MPI_Type_get_extent(oldtype, &lb, &extent);
    MPI_Type_dup(oldtype, &t[ndims]);
    for (int i = ndims - 1; i >= 0; --i) {
        int blocklength[3];
        MPI_Aint displacement[3];
        MPI_Datatype type[3];

        blocklength[0] = 1;
        displacement[0] = 0;
        type[0] = MPI_LB;

        blocklength[1] = array_of_subsizes[i];
        displacement[1] = extent * array_of_starts[i];
        type[1] = t[i + 1];

        blocklength[2] = 1;
        displacement[2] = extent * array_of_sizes[i];
        type[2] = MPI_UB;

        MPI_Type_create_struct(3,
                               blocklength,
                               displacement,
                               type,
                               &t[i]);
        extent *= array_of_sizes[i];
    }
    MPI_Type_dup(t[0], newtype);
    MPI_Type_commit(newtype);
    for (int i = 0; i < (ndims + 1); ++i) {
        MPI_Type_free(&t[i]);
    }
    free(t);

    return 0;
}

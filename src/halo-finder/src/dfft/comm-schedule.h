///
// A communication schedule is a list of sends and recvs to execute.
//
// See test-comm-schedule.c for an example of use.
///

#ifndef COMM_SCHEDULE_H
#define COMM_SCHEDULE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

///
// eunumeration of directions (send or recv) in a communication schedule
///
enum { COMM_SCHEDULE_SEND = 1, COMM_SCHEDULE_RECV = 2 };

///
// descriptor for a communication schedule
//   next       descriptor for the next item in the schedule
//   addr       the base address for this item
//   type       MPI datatype for this item
//   comm       MPI communicator for this item
//   peer       the rank of my peer for this item
//   direction  send or recv
//   count      number of MPI datatypes to send for this item
///
typedef struct comm_schedule_t comm_schedule_t;
struct comm_schedule_t {
    comm_schedule_t *next;
    void *addr;
    MPI_Datatype type;
    MPI_Comm comm;
    MPI_Request req;
    int peer;
    int direction;
    int count;
};

///
// append a new item to a communication schedule
///
comm_schedule_t *comm_schedule_append(comm_schedule_t *schedule,
                                      MPI_Comm comm,
                                      int peer,
                                      int direction,
                                      void *addr,
                                      MPI_Datatype type,
                                      int count);

///
// prepend a new item to a communication schedule
///
comm_schedule_t *comm_schedule_prepend(comm_schedule_t *schedule,
                                       MPI_Comm comm,
                                       int peer,
                                       int direction,
                                       void *addr,
                                       MPI_Datatype type,
                                       int count);

///
// start execution of a communication schedule
///
void comm_schedule_start(comm_schedule_t *schedule);

///
// progress an already started communication schedule
///
void comm_schedule_progress(comm_schedule_t *schedule);

///
// wait for completion of a communication schedule
///
void comm_schedule_wait(comm_schedule_t *schedule);

///
// execute a communication schedule
///
void comm_schedule_execute(comm_schedule_t *schedule);

///
// delete a communication schedule
///
void comm_schedule_free(comm_schedule_t *schedule);

#ifdef __cplusplus
}
#endif

#endif

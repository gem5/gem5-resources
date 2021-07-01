///
// A communication schedule is a list of sends and recvs to execute.
//
// See test-comm-schedule.c for an example of use.
///

#ifndef ACTIVE_SCHEDULE_H
#define ACTIVE_SCHEDULE_H

#ifdef __cplusplus
#define EXTERN_C_BEGIN extern "C" {
#define EXTERN_C_END }
#else
#define EXTERN_C_BEGIN
#define EXTERN_C_END
#endif

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

EXTERN_C_BEGIN

///
// Function pointer
///
typedef void (*active_function_pointer_t)(void *);

///
// eunumeration of directions (send or recv) in a communication schedule
///
enum { ACTIVE_SCHEDULE_SEND = 1, ACTIVE_SCHEDULE_RECV = 2 };

///
// descriptor for a communication schedule
//   next       descriptor for the next item in the schedule
//   addr       the base address for this item
//   type       MPI datatype for this item
//   comm       MPI communicator for this item
//   peer       the rank of my peer for this item
//   direction  send or recv
//   count      number of MPI datatypes to send for this item
//   pre_func   a function to call before the send/recv is started
//   pre_data   pointer to data for pre_func
//   post_func  a function to call after the send/recv completes
//   post_data  pointer to data for post_func
///
typedef struct active_schedule_t active_schedule_t;
struct active_schedule_t {
    active_schedule_t *next;
    void *addr;
    MPI_Datatype type;
    MPI_Comm comm;
    MPI_Request req;
    int peer;
    int direction;
    int count;
    active_function_pointer_t pre_function;
    void *pre_data;
    active_function_pointer_t post_function;
    void *post_data;
};


///
// append a new item to a communication schedule
///
active_schedule_t *active_schedule_append(active_schedule_t *schedule,
                                          MPI_Comm comm,
                                          int peer,
                                          int direction,
                                          void *addr,
                                          MPI_Datatype type,
                                          int count,
                                          active_function_pointer_t pre_function,
                                          void *pre_data,
                                          active_function_pointer_t post_function,
                                          void *post_data);

///
// prepend a new item to a communication schedule
///
active_schedule_t *active_schedule_prepend(active_schedule_t *schedule,
                                           MPI_Comm comm,
                                           int peer,
                                           int direction,
                                           void *addr,
                                           MPI_Datatype type,
                                           int count,
                                           active_function_pointer_t pre_function,
                                           void *pre_data,
                                           active_function_pointer_t post_function,
                                           void *post_data);

///
// start execution of a communication schedule
///
void active_schedule_start(active_schedule_t *schedule);

///
// progress an already started communication schedule
///
void active_schedule_progress(active_schedule_t *schedule);

///
// wait for completion of a communication schedule
///
void active_schedule_wait(active_schedule_t *schedule);

///
// execute a communication schedule
///
void active_schedule_execute(active_schedule_t *schedule, int depth);

///
// delete a communication schedule
///
void active_schedule_free(active_schedule_t *schedule);

EXTERN_C_END

#endif

#include <stdbool.h>

#include "comm-schedule.h"

#ifdef DEBUG
static bool debug = true;
#else
static bool debug = false;
#endif


static void print_schedule(comm_schedule_t *schedule)
{
    int global_self;
    int self;
    MPI_Aint lb;
    MPI_Aint extent;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &global_self);
    MPI_Comm_rank(schedule->comm, &self);
    MPI_Type_get_extent(schedule->type, &lb, &extent);

    fprintf(stderr,
            "%d: schedule=%p, comm=%p, "
            "type=%p(lb=%ld, extent=%ld), count=%d, %d %s %d\n",
            global_self,
            schedule,
            (void *) schedule->comm,
            (void *) schedule->type,
            (long) lb,
            (long) extent,
            schedule->count,
            self,
            schedule->direction == COMM_SCHEDULE_SEND ? " -> " : " <- ",
            schedule->peer);
}            


comm_schedule_t *comm_schedule_append(comm_schedule_t *schedule,
                                      MPI_Comm comm,
                                      int peer,
                                      int direction,
                                      void *addr,
                                      MPI_Datatype type,
                                      int count)
{
    comm_schedule_t *new_schedule;

    new_schedule = (comm_schedule_t *) malloc(sizeof(comm_schedule_t));
    if (!new_schedule) {
        perror("out of memory");
    }
    new_schedule->next = NULL;
    new_schedule->comm = comm;
    new_schedule->peer = peer;
    new_schedule->direction = direction;
    new_schedule->addr = addr;
    MPI_Type_dup(type, &new_schedule->type);
    new_schedule->count = count;
    new_schedule->req = MPI_REQUEST_NULL;

    if (schedule == NULL) {
        schedule = new_schedule;
    } else {
        comm_schedule_t *p = schedule;
        while (p->next) {
            p = p->next;
        }
        p->next = new_schedule;
    }

    return schedule;
}


comm_schedule_t *comm_schedule_prepend(comm_schedule_t *schedule,
                                       MPI_Comm comm,
                                       int peer,
                                       int direction,
                                       void *addr,
                                       MPI_Datatype type,
                                       int count)
{
    comm_schedule_t *new_schedule;

    new_schedule = (comm_schedule_t *) malloc(sizeof(comm_schedule_t));
    if (!new_schedule) {
        perror("out of memory");
    }
    new_schedule->next = schedule;
    new_schedule->comm = comm;
    new_schedule->peer = peer;
    new_schedule->direction = direction;
    new_schedule->addr = addr;
    MPI_Type_dup(type, &new_schedule->type);
    new_schedule->count = count;
    new_schedule->req = MPI_REQUEST_NULL;
    return new_schedule;
}


void comm_schedule_free(comm_schedule_t *schedule)
{
    while (schedule) {
        comm_schedule_t *s = schedule->next;
        MPI_Type_free(&schedule->type);
        free(schedule);
        schedule = s;
    }
}


void comm_schedule_start(comm_schedule_t *schedule)
{
    for (comm_schedule_t *s = schedule; s; s = s->next) {
        if (debug) {
            print_schedule(s);
        }
        if (s->direction == COMM_SCHEDULE_SEND) {
            MPI_Isend(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
        } else if (s->direction == COMM_SCHEDULE_RECV) {
            MPI_Irecv(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
        } else {
            perror("unknown direction in communication schedule");
        }
    }
}


void comm_schedule_progress(comm_schedule_t *schedule)
{
    for (comm_schedule_t *s = schedule; s; s = s->next) {
        int flag;
        MPI_Test(&s->req, &flag, MPI_STATUS_IGNORE);
    }
}


void comm_schedule_wait(comm_schedule_t *schedule)
{
    MPI_Request array_of_requests[256];
    int i;
    
    i = 0;
    for (comm_schedule_t *s = schedule; s; s = s->next) {
        array_of_requests[i] = s -> req;
        i = i + 1;
    }

    MPI_Waitall( i, array_of_requests, MPI_STATUSES_IGNORE );

    //for (comm_schedule_t *s = schedule; s; s = s->next) {
    //    MPI_Wait(&s->req, MPI_STATUS_IGNORE);
    //}
}


void comm_schedule_execute(comm_schedule_t *schedule)
{
    comm_schedule_start(schedule);
    comm_schedule_wait(schedule);
}

#include <stdbool.h>

#include "active-schedule.h"

#ifdef DEBUG
static bool debug = true;
#else
static bool debug = false;
#endif

static void print_schedule(active_schedule_t *schedule)
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
            schedule->direction == ACTIVE_SCHEDULE_SEND ? " -> " : " <- ",
            schedule->peer);
}            


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
                                           void *post_data)
{
    active_schedule_t *new_schedule;

    new_schedule = (active_schedule_t *) malloc(sizeof(active_schedule_t));
    if (!new_schedule) {
        perror("out of memory");
    }
    new_schedule->next = schedule;
    new_schedule->comm = comm;
    new_schedule->peer = peer;
    new_schedule->direction = direction;
    new_schedule->addr = addr;
    MPI_Type_dup(type, &new_schedule->type);
    new_schedule->pre_function = pre_function;
    new_schedule->pre_data = pre_data;
    new_schedule->post_function = post_function;
    new_schedule->post_data = post_data;
    new_schedule->count = count;
    new_schedule->req = MPI_REQUEST_NULL;
    return new_schedule;
}


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
                                          void *post_data)
{
    active_schedule_t *new_schedule;

    new_schedule = (active_schedule_t *) malloc(sizeof(active_schedule_t));
    if (!new_schedule) {
        perror("out of memory");
    }
    new_schedule->next = NULL;
    new_schedule->comm = comm;
    new_schedule->peer = peer;
    new_schedule->direction = direction;
    new_schedule->addr = addr;
    MPI_Type_dup(type, &new_schedule->type);
    new_schedule->pre_function = pre_function;
    new_schedule->pre_data = pre_data;
    new_schedule->post_function = post_function;
    new_schedule->post_data = post_data;
    new_schedule->count = count;
    new_schedule->req = MPI_REQUEST_NULL;

    if (schedule == NULL) {
        schedule = new_schedule;
    } else {
        active_schedule_t *p = schedule;
        while (p->next) {
            p = p->next;
        }
        p->next = new_schedule;
    }

    return schedule;
}


void active_schedule_free(active_schedule_t *schedule)
{
    while (schedule) {
        active_schedule_t *s = schedule->next;
        MPI_Type_free(&schedule->type);
        free(schedule);
        schedule = s;
    }
}


void active_schedule_start(active_schedule_t *schedule)
{
    for (active_schedule_t *s = schedule; s; s = s->next) {
        if (debug) {
            print_schedule(s);
        }
        if (s->pre_function) {
            s->pre_function(s->pre_data);
        }
        if (s->direction == ACTIVE_SCHEDULE_SEND) {
            MPI_Isend(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
        } else if (s->direction == ACTIVE_SCHEDULE_RECV) {
            MPI_Irecv(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
        } else {
            perror("unknown direction in communication schedule");
        }
    }
}


void active_schedule_progress(active_schedule_t *schedule)
{
    for (active_schedule_t *s = schedule; s; s = s->next) {
        int flag;
        MPI_Test(&s->req, &flag, MPI_STATUS_IGNORE);
    }
}


void active_schedule_wait(active_schedule_t *schedule)
{
    for (active_schedule_t *s = schedule; s; s = s->next) {
        MPI_Wait(&s->req, MPI_STATUS_IGNORE);
        if (s->post_function) {
            s->post_function(s->post_data);
        }
    }
}


void active_schedule_execute(active_schedule_t *schedule, int depth)
{

    active_schedule_t *s = schedule;
    active_schedule_t *t = schedule;
    
    while (t) {
        if (s) {
            if (s->pre_function) {
                s->pre_function(s->pre_data);
            }
            if (s->direction == ACTIVE_SCHEDULE_SEND) {
                MPI_Isend(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
            } else if (s->direction == ACTIVE_SCHEDULE_RECV) {
                MPI_Irecv(s->addr, s->count, s->type, s->peer, 0, s->comm, &s->req);
            } else {
                perror("unknown direction in communication schedule");
            }
            s = s->next;
        }

        if (depth == 0) {
            MPI_Wait(&t->req, MPI_STATUS_IGNORE);
            if (t->post_function) {
                t->post_function(t->post_data);
            }
            t = t->next;
        } else {
            --depth;
        }
    }
}

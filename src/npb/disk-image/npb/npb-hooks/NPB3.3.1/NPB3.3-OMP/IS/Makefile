SHELL=/bin/sh
BENCHMARK=is
BENCHMARKU=IS

include ../config/make.def

include ../sys/make.common

OBJS = is.o \
       ${COMMON}/c_print_results.o \
       ${COMMON}/c_timers.o \
       ${COMMON}/c_wtime.o
ifeq (${HOOKS}, 1)
        OBJS += ${COMMON}/hooks.o ${COMMON}/m5op_x86.o ${COMMON}/m5_mmap.o
endif


${PROGRAM}: config ${OBJS}
	${CLINK} ${CLINKFLAGS} -no-pie -o ${PROGRAM} ${OBJS} ${C_LIB}

.c.o:
ifeq (${HOOKS}, 1)
	${CCOMPILE} -DHOOKS $<
else
	${CCOMPILE} $<
endif

is.o:             is.c  npbparams.h


clean:
	- rm -f *.o *~ mputil*
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi

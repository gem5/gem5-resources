SHELL=/bin/sh
BENCHMARK=ep
BENCHMARKU=EP

include ../config/make.def

OBJS = ep.o ${COMMON}/print_results.o ${COMMON}/${RAND}.o \
       ${COMMON}/timers.o ${COMMON}/wtime.o
ifeq (${HOOKS}, 1)
        OBJS += ${COMMON}/hooks.o ${COMMON}/m5op_x86.o ${COMMON}/m5_mmap.o
endif

include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${FLINK} ${FLINKFLAGS} -no-pie -o ${PROGRAM} ${OBJS} ${F_LIB}


ep.o:		ep.f npbparams.h
ifeq (${HOOKS}, 1)
		${FCOMPILE} -DHOOKS ep.f
else
                ${FCOMPILE} ep.f
endif

clean:
	- rm -f *.o *~
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi

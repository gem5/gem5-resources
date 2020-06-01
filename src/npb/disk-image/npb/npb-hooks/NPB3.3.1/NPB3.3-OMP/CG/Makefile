SHELL=/bin/sh
BENCHMARK=cg
BENCHMARKU=CG

include ../config/make.def

OBJS = cg.o ${COMMON}/print_results.o  \
       ${COMMON}/${RAND}.o ${COMMON}/timers.o ${COMMON}/wtime.o
ifeq (${HOOKS}, 1)
        OBJS += ${COMMON}/hooks.o ${COMMON}/m5op_x86.o ${COMMON}/m5_mmap.o
endif


include ../sys/make.common

${PROGRAM}: config ${OBJS}
	${FLINK} ${FLINKFLAGS} -no-pie -o ${PROGRAM} ${OBJS} ${F_LIB}

cg.o:		cg.f  globals.h npbparams.h
ifeq (${HOOKS}, 1)
	        ${FCOMPILE} -DHOOKS cg.f
else
	        ${FCOMPILE} cg.f
endif

clean:
	- rm -f *.o *~
	- rm -f npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi

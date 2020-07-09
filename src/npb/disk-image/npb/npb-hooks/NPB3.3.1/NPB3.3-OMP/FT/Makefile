SHELL=/bin/sh
BENCHMARK=ft
BENCHMARKU=FT

include ../config/make.def

include ../sys/make.common

OBJS = ft.o ${COMMON}/${RAND}.o ${COMMON}/print_results.o \
       ${COMMON}/timers.o ${COMMON}/wtime.o
ifeq (${HOOKS}, 1)
        OBJS += ${COMMON}/hooks.o ${COMMON}/m5op_x86.o ${COMMON}/m5_mmap.o
endif

${PROGRAM}: config ${OBJS}
	${FLINK} ${FLINKFLAGS} -no-pie -o ${PROGRAM} ${OBJS} ${F_LIB}



.f.o:
ifeq (${HOOKS}, 1)
	${FCOMPILE} -DHOOKS $<
else
	${FCOMPILE} $<
endif

ft.o:             ft.f  global.h npbparams.h

clean:
	- rm -f *.o *~ mputil*
	- rm -f ft npbparams.h core
	- if [ -d rii_files ]; then rm -r rii_files; fi

SHELL	:= /bin/sh

OBJDIR := ${HACC_OBJDIR}

SOURCES += distribution.c
SOURCES += comm-schedule.c
SOURCES += dims.c
#SOURCES += subarray.c
OBJLIST = $(SOURCES:.c=.o)
OBJECTS := $(addprefix $(OBJDIR)/,$(OBJLIST))

TARGETS	+= $(OBJDIR)/libdfft.a

all: $(TARGETS)

tests	+= $(OBJDIR)/test-comm-schedule
tests	+= $(OBJDIR)/test-distribution
tests	+= $(OBJDIR)/test-distribution-2-to-3
tests	+= $(OBJDIR)/test-dfft
tests	+= $(OBJDIR)/test-solver-class
tests	+= $(OBJDIR)/test-solver-bench
tests	+= $(OBJDIR)/test-qpm
tests   += $(OBJDIR)/test-mpi-init
tests-fftw3 += $(OBJDIR)/test-distribution-fft
tests-fftw3 += $(OBJDIR)/test-delta-function
tests-fftw3 += $(OBJDIR)/test-timing
tests-fftw3 += $(OBJDIR)/test-solver
tests-fftw3 += $(OBJDIR)/test-zarija
tests-fftw3 += $(OBJDIR)/test-fftw-guru

fftw	:= $(FFTW_HOME)

CFLAGS += ${HACC_MPI_CFLAGS}
CXXFLAGS += ${HACC_MPI_CXXFLAGS}
LDFLAGS += ${HACC_MPI_LDFLAGS}

CFLAGS	+= $(WARN)
CFLAGS	+= -DUSE_SLAB_WORKAROUND=1
CFLAGS	+= -I$(fftw)/include -I../halo_finder

CXXFLAGS += $(WARN)
CXXFLAGS += -I$(fftw)/include -I../halo_finder

LDFLAGS	+= -L$(fftw)/lib

ifeq "${FFTW_MAJOR_VERSION}" "2"
  CFLAGS   += -DFFTW2=1
  CXXFLAGS += -DFFTW2=1
  LDLIBS   += -lfftw_mpi -lfftw
endif

ifeq "${FFTW_MAJOR_VERSION}" "3"
  CFLAGS	+= -DFFTW3=1
  CXXFLAGS	+= -DFFTW3=1
ifeq ("${FFTW_WRAPPER}", "essl")
  CFLAGS += -DESSL_FFTW=1
  CXXFLAGS += -DESSL_FFTW=1
  LDLIBS += -lfftw3_esslbg -lesslsmpbg
ifeq ("${FFTW_THREADING}", "omp")
  CFLAGS	+= -DFFTW3_THREADS=1
  CXXFLAGS	+= -DFFTW3_THREADS=1
  LDLIBS	+= -lfftw3_mpi -lfftw3_omp -lfftw3
else
  LDLIBS	+= -lfftw3_mpi -lfftw3
endif
endif
include pencil.mk
  CFLAGS	+= ${DFFT_PEN_CFLAGS}
  CXXFLAGS	+= ${DFFT_PEN_CXXFLAGS}
endif

#$(OBJECTS): | $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	${HACC_MPI_CC} ${CFLAGS} -c -o $@ $<

$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	${HACC_MPI_CXX} ${CXXFLAGS} -c -o $@ $<

clean::
	$(RM) $(TARGETS) $(tests) $(tests-fftw3) plot.in *.o *~ bigchunk-local.c
	$(RM) -rf $(OBJDIR)

check: tests
	for t in $(tests) ; do \
	    echo '++++' $$t '++++' ; \
	    mpirun -np 8 $$t ; \
	    echo '----' $$t '----' ; \
	done
	make -C testdata/qpm check-data
	make -C testdata/qpm clean

check-fftw3: tests-fftw3

tags:
	find .. -type f | egrep '(c|cc|cpp|cxx|h|hpp|hxx)$$' | xargs /usr/bin/etags

tests: $(tests)

ifeq "${FFTW_MAJOR_VERSION}" "3"
tests-fftw3: $(tests-fftw3)
else
tests-fftw3:
endif


$(OBJDIR)/libdfft.a: $(OBJDIR)/libdfft.a($(OBJECTS))
	ranlib $@

$(OBJDIR)/test-%: $(OBJDIR)/test-%.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CC} $(CFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

bigchunk-local.c: ../halo_finder/bigchunk.c
	cp ../halo_finder/bigchunk.c bigchunk-local.c

$(OBJDIR)/test-solver-class.o: solver.hpp
$(OBJDIR)/test-solver-class: $(OBJDIR)/test-solver-class.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CXX} $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJDIR)/test-solver-bench.o: solver.hpp
$(OBJDIR)/test-solver-bench: $(OBJDIR)/test-solver-bench.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CXX} $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJDIR)/test-qpm.o: solver.hpp
$(OBJDIR)/test-qpm: $(OBJDIR)/test-qpm.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CXX} $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJDIR)/test-zarija.o: solver.hpp
$(OBJDIR)/test-zarija: $(OBJDIR)/test-zarija.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CXX} $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

$(OBJDIR)/test-dfft.o: dfft.hpp distribution.hpp distribution.h
$(OBJDIR)/test-dfft: $(OBJDIR)/test-dfft.o $(OBJDIR)/bigchunk-local.o $(OBJDIR)/libdfft.a
	${HACC_MPI_CXX} $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

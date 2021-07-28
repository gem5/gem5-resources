OBJDIR = ${HACC_OBJDIR}
#SOBJDIR = ${HACC_OBJDIR}_serial


PROGS    = \
	$(OBJDIR)/ForceTreeTest


LIBS	= \
	$(OBJDIR)/libpartition.a \
	$(OBJDIR)/libparticle.a \
	$(OBJDIR)/libBHForceTree.a \
	$(OBJDIR)/libhalotime.a \
	$(OBJDIR)/libbigchunk.a


# Probably need to fix dependencies on some of the header files.
HDR= \
	RCBForceTree.h \
	RCOForceTree.h \
	BHForceTree.h \
	CosmoHalo.h \
	Definition.h \
	ForceLaw.h \
	ForceTree.h \
	InitialExchange.h \
	Message.h \
	Partition.h \
	ParticleDistribute.h \
	ParticleExchange.h \
	Timer.h \
	Timings.h \
	cudaUtil.h


all: ${LIBS}
libs: ${LIBS}
progs: ${PROGS}

include include.mk
#include ../include.mk
include dfft/include.mk
CXXFLAGS += ${DFFT_CXXFLAGS}

CFLAGS += ${HACC_MPI_CFLAGS}
CXXFLAGS += ${HACC_MPI_CXXFLAGS}
LDFLAGS += ${HACC_MPI_LDFLAGS}

#CXXFLAGS += ${MPI_COMPILE_FLAGS} 
CXXFLAGS += ${HF_TYPE_FLAGS}
CXXFLAGS += ${HF_WARNING}
#CXXFLAGS += -g
#CXXFLAGS += -DDEBUG
CXXFLAGS += -I.
#LDFLAGS  = ${MPI_LD_FLAGS} -lmpi_cxx

CFLAGS += -Idfft
CXXFLAGS += -Idfft


$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	${HACC_MPI_CC} ${CFLAGS} -c -o $@ $<

$(OBJDIR)/%.o: %.cxx | $(OBJDIR)
	${HACC_MPI_CXX} ${CXXFLAGS} -c -o $@ $<

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	${HACC_MPI_CXX} ${CFLAGS} -c -o $@ $<


#$(SOBJDIR):
#	mkdir -p $(SOBJDIR)
#
#$(SOBJDIR)/%.o: %.c | $(SOBJDIR)
#	${HACC_CC} -c -o $@ $<
#
#$(SOBJDIR)/%.o: %.cxx | $(SOBJDIR)
#	${HACC_CXX} -c -o $@ $<


#$(OBJDIR)/ForceLaw.o: ForceLaw.h


dims-local.c: dfft/dims.c
	cp -f dfft/dims.c dims-local.c


$(OBJDIR)/ForceTreeTest: $(OBJDIR)/ForceTreeTest.o $(OBJDIR)/libparticle.a $(OBJDIR)/libBHForceTree.a $(OBJDIR)/libpartition.a $(OBJDIR)/libhalotime.a $(OBJDIR)/libbigchunk.a $(OBJDIR)/dims-local.o
	${HACC_MPI_CXX} -o $@ $^ ${CXXFLAGS} ${LDFLAGS} -DUSE_SERIAL_COSMO=1

$(OBJDIR)/libpartition.a: $(OBJDIR)/libpartition.a($(OBJDIR)/Partition.o)
	ranlib $@


PARTICLE_SOURCES += ParticleDistribute.cxx
PARTICLE_SOURCES += ParticleExchange.cxx
PARTICLE_SOURCES += InitialExchange.cxx
PARTICLE_SOURCES += Message.cxx
PARTICLE_OBJLIST = $(PARTICLE_SOURCES:.cxx=.o)
PARTICLE_OBJECTS := $(addprefix $(OBJDIR)/,$(PARTICLE_OBJLIST))
$(OBJDIR)/libparticle.a: $(OBJDIR)/libparticle.a($(PARTICLE_OBJECTS))
	ranlib $@


HT_SOURCES += Timings.cxx
HT_SOURCES += Timer.cxx
HT_OBJLIST = $(HT_SOURCES:.cxx=.o)
HT_OBJECTS := $(addprefix $(OBJDIR)/,$(HT_OBJLIST))
$(OBJDIR)/libhalotime.a: $(OBJDIR)/libhalotime.a($(HT_OBJECTS))
	ranlib $@


FORCE_SOURCES += ForceLaw.cxx
FORCE_SOURCES += BHForceTree.cxx
FORCE_SOURCES += RCOForceTree.cxx
FORCE_SOURCES += RCBForceTree.cxx
FORCE_SOURCES_C += BGQStep16.c
FORCE_SOURCES_C += BGQCM.c
FORCE_OBJLIST = $(FORCE_SOURCES:.cxx=.o)
FORCE_OBJLIST_C = $(FORCE_SOURCES_C:.c=.o)
FORCE_OBJECTS := $(addprefix $(OBJDIR)/,$(FORCE_OBJLIST) $(FORCE_OBJLIST_C))
$(OBJDIR)/libBHForceTree.a: $(OBJDIR)/libBHForceTree.a($(FORCE_OBJECTS) $(FORCE_OBJLIST_C))
	ranlib $@

$(OBJDIR)/libbigchunk.a: $(OBJDIR)/libbigchunk.a($(OBJDIR)/bigchunk.o)
	ranlib $@

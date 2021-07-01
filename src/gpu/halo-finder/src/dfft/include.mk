DFFT := dfft
DFFT_HEADERS := ${DFFT}/distribution.h
DFFT_HEADERS += ${DFFT}/allocator.hpp
DFFT_HEADERS += ${DFFT}/distribution.hpp
DFFT_HEADERS += ${DFFT}/solver.hpp
DFFT_CXXFLAGS := -I${DFFT}
#DFFT_LDFLAGS := -L${DFFT}/${HACC_OBJDIR} -ldfft
DFFT_LDFLAGS := -L${DFFT}/${HACC_OBJDIR}
include ${DFFT}/pencil.mk
DFFT_CXXFLAGS += ${DFFT_PEN_CXXFLAGS}

# Common definitions for all Makefiles
CC = mpicc -Wall -pedantic -I$(MPICH)
# CC = clang -Wall -pedantic
INC = $(ROOT)/include
CIVL = /opt/sw/CIVL-trunk_5891/lib/dev/civl/abc/include
PETSC = $(ROOT)/original/inc
SRC = $(ROOT)/src
VEC = $(SRC)/vec
TEST = $(ROOT)/test
MPICH = /usr/include/x86_64-linux-gnu/mpich
LDFLAGS += $(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
NP = 5

# Common definitions for all Makefiles
#CC = mpicc -Wall -pedantic -I$(MPICH)
CC = clang -Wall -pedantic
INC = $(ROOT)/scaffolding/include
CIVL = /opt/sw/CIVL-trunk_5891/lib/dev/civl/abc/include
PETSC = $(ROOT)/original/inc
SRC = $(ROOT)/scaffolding/src
TEST = $(ROOT)/scaffolding/test
MPICH = /usr/include/x86_64-linux-gnu/mpich
LDFLAGS += $(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
NP = 5
#TYPE = -DVecNorm_Seq=VecNorm_Seq_spec
#TYPE = 
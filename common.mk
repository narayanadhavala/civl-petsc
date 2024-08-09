# Common definitions for all Makefiles
CC = mpicc -Wall -pedantic -I$(MPICH) -g
INC = $(ROOT)/scaffolding/include
CIVL = /opt/sw/CIVL-trunk_5891/lib/dev/civl/abc/include
SRC = $(ROOT)/scaffolding/src
MPICH = /usr/include/x86_64-linux-gnu/mpich
LDFLAGS += $(PETSC_DIR)/$(PETSC_ARCH)/lib -lpetsc
NP = 5
#TYPE = -DVecNorm_Seq=VecNorm_Seq_spec -DVecCopy_Seq=VecCopy_Seq_spec
#TYPE = 
ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify

#all: clear VecNormAvailable.o VecNormAvailable_test.exec

test: VecNormAvailable_test.exec
	mpirun -np $(NP) ./VecNormAvailable_test.exec

VecNormAvailable_test.exec: VecNormAvailable_test.c VecNormAvailable.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(CC) $(TYPE) -DVecNormAvailable=VecNormAvailable_spec -I$(INC) -o VecNormAvailable_test.exec VecNormAvailable_test.c VecNormAvailable.o\
		$(VEC)/petscvec.c -lm

VecNormAvailable.o: VecNormAvailable.c $(INC)/petscvec.h $(PETSC)
	$(CC) $(TYPE) -DVecNormAvailable=VecNormAvailable_spec -I$(INC) -I$(PETSC) -c VecNormAvailable.c

verify: VecNormAvailable_driver.cvl $(VEC)/petscvec.c VecNormAvailable.c $(INC)/petscvec.h
	civl verify $(TYPE) -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecNormAvailable=VecNormAvailable_spec -sysIncludePath=$(INC)  VecNormAvailable_driver.cvl	$(VEC)/petscvec.c\
	  VecNormAvailable.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: all verify clean test

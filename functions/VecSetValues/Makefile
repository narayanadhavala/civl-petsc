ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecSetValues_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=5 -inputN_MAX=5 -inputN=5 -sysIncludePath=$(INC) VecSetValues_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecSetValues_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=5 -inputN_MAX=5 -inputN=5 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecSetValues_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecSetValues_driver.cvl $(VEC)/petscvec.c VecSetValues.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=5 -inputB=2 -DVecSetValues=VecSetValues_spec -sysIncludePath=$(INC) VecSetValues_driver.cvl $(VEC)/petscvec.c VecSetValues.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecSetValues_driver.cvl $(VEC)/petscvec.c VecSetValues.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -inputB=2  -DVecSetValues=VecSetValues_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecSetValues_driver.cvl $(VEC)/petscvec.c VecSetValues.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

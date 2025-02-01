ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecMin_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -sysIncludePath=$(INC) VecMin_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecMin_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecMin_driver.cvl $(VEC)/petscvec.c VecMin.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputN_MIN=2 -inputN_MAX=2 -DVecMin=VecMin_spec -sysIncludePath=$(INC) VecMin_driver.cvl $(VEC)/petscvec.c VecMin.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecMin_driver.cvl $(VEC)/petscvec.c VecMin.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputN_MIN=2 -inputN_MAX=2 -DVecMin=VecMin_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_driver.cvl $(VEC)/petscvec.c VecMin.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

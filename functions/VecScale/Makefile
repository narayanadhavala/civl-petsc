ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecScale_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -sysIncludePath=$(INC) VecScale_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecScale_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecScale_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecScale_driver.cvl $(VEC)/petscvec.c VecScale.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputN_MIN=2 -inputN_MAX=2 -DVecScale=VecScale_spec -sysIncludePath=$(INC) VecScale_driver.cvl $(VEC)/petscvec.c VecScale.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecScale_driver.cvl $(VEC)/petscvec.c VecScale.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputN_MIN=2 -inputN_MAX=2 -DVecScale=VecScale_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecScale_driver.cvl $(VEC)/petscvec.c VecScale.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

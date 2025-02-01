ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecDotRealPart_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -sysIncludePath=$(INC) VecDotRealPart_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecDotRealPart_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecDotRealPart_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=20 -DVecDotRealPart=VecDotRealPart_spec -sysIncludePath=$(INC) VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -inputN_MIN=5 -inputN_MAX=5 -DVecDotRealPart=VecDotRealPart_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

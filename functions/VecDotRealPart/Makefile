ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecDotRealPart_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -sysIncludePath=$(INC) VecDotRealPart_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecDotRealPart_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecDotRealPart_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=4 -inputN_MAX=4)\
	 -DVecDotRealPart=VecDotRealPart_spec -sysIncludePath=$(INC) VecDotRealPart_driver.cvl $(VEC)/petscvec.c\
	  VecDotRealPart.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecDotRealPart_driver.cvl $(VEC)/petscvec.c VecDotRealPart.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=4 -inputN_MAX=4)\
	 -DVecDotRealPart=VecDotRealPart_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecDotRealPart_driver.cvl\
	  $(VEC)/petscvec.c VecDotRealPart.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecMaxPointwiseDivide_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=5 -sysIncludePath=$(INC)\
	 VecMaxPointwiseDivide_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecMaxPointwiseDivide_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=5 -DUSE_COMPLEX -DCIVL_COMPLEX\
	 -sysIncludePath=$(INC) VecMaxPointwiseDivide_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecMaxPointwiseDivide_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=3)\
	 -DVecMaxPointwiseDivide=VecMaxPointwiseDivide_spec -sysIncludePath=$(INC) VecMaxPointwiseDivide_driver.cvl\
	  $(VEC)/petscvec.c VecMaxPointwiseDivide.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

verify_complex: VecMaxPointwiseDivide_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMaxPointwiseDivide=VecMaxPointwiseDivide_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC)\
	  VecMaxPointwiseDivide_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

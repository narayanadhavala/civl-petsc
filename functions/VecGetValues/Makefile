ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecGetValues_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=5 -inputN_MAX=5 -inputN=5 -sysIncludePath=$(INC) VecGetValues_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecGetValues_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=5 -inputN_MAX=5 -inputN=5 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecGetValues_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecGetValues_driver.cvl $(VEC)/petscvec.c VecGetValues.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=3 -inputN_MAX=3 -inputB=2)\
	 -DVecGetValues=VecGetValues_spec -sysIncludePath=$(INC) VecGetValues_driver.cvl $(VEC)/petscvec.c\
	  VecGetValues.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecGetValues_driver.cvl $(VEC)/petscvec.c VecGetValues.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=3 -inputN_MAX=3 -inputB=2)\
	 -DVecGetValues=VecGetValues_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecGetValues_driver.cvl $(VEC)/petscvec.c\
	  VecGetValues.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

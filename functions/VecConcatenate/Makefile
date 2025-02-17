ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec
DEBUG = show -debug=true

all: verify_real verify_complex

test_real: VecConcatenate_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -DVecConcatenate=VecConcatenate_spec -sysIncludePath=$(INC)\
	 VecConcatenate_test.cvl $(ROOT)/functions/VecConcatenate/VecConcatenate.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl\
	  $(VEC)/civlcomplex.cvl

test_complex: VecConcatenate_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -DUSE_COMPLEX -DCIVL_COMPLEX -DVecConcatenate=VecConcatenate_spec\
	 -sysIncludePath=$(INC) VecConcatenate_test.cvl $(ROOT)/functions/VecConcatenate/VecConcatenate.c $(VEC)/petscvec.c\
	  $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecConcatenate_driver.cvl $(VEC)/petscvec.c VecConcatenate.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=2 -inputB=2)\
	 -DVecConcatenate=VecConcatenate_spec -sysIncludePath=$(INC) VecConcatenate_driver.cvl $(VEC)/petscvec.c VecConcatenate.c\
	  $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecConcatenate_driver.cvl $(VEC)/petscvec.c VecConcatenate.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS), -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=2 -inputB=2)\
	 -DVecConcatenate=VecConcatenate_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecConcatenate_driver.cvl\
	  $(VEC)/petscvec.c VecConcatenate.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

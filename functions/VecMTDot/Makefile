ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecMTDot_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -sysIncludePath=$(INC) VecMTDot_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecMTDot_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMTDot_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecMTDot_driver.cvl $(VEC)/petscvec.c VecMTDot.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMTDot=VecMTDot_spec -sysIncludePath=$(INC) VecMTDot_driver.cvl $(VEC)/petscvec.c\
	  VecMTDot.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecMTDot_driver.cvl $(VEC)/petscvec.c VecMTDot.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMTDot=VecMTDot_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMTDot_driver.cvl $(VEC)/petscvec.c\
	  VecMTDot.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

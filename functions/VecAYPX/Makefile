ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecAYPX_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DVecAYPX=VecAYPX_spec -sysIncludePath=$(INC) VecAYPX_test.cvl $(ROOT)/functions/VecAYPX/VecAYPX.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecAYPX_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DVecAYPX=VecAYPX_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAYPX_test.cvl $(ROOT)/functions/VecAYPX/VecAYPX.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecAYPX_driver.cvl $(VEC)/petscvec.c VecAYPX.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecAYPX=VecAYPX_spec -sysIncludePath=$(INC) VecAYPX_driver.cvl $(VEC)/petscvec.c\
	  VecAYPX.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecAYPX_driver.cvl $(VEC)/petscvec.c VecAYPX.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecAYPX=VecAYPX_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAYPX_driver.cvl $(VEC)/petscvec.c\
	  VecAYPX.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

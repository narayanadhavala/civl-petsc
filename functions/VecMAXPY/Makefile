ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecMAXPY_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DVecMAXPY=VecMAXPY_spec -sysIncludePath=$(INC) VecMAXPY_test.cvl $(ROOT)/functions/VecMAXPY/VecMAXPY.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecMAXPY_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DVecMAXPY=VecMAXPY_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMAXPY_test.cvl $(ROOT)/functions/VecMAXPY/VecMAXPY.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecMAXPY_driver.cvl $(VEC)/petscvec.c VecMAXPY.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMAXPY=VecMAXPY_spec -sysIncludePath=$(INC) VecMAXPY_driver.cvl $(VEC)/petscvec.c\
	  VecMAXPY.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecMAXPY_driver.cvl $(VEC)/petscvec.c VecMAXPY.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMAXPY=VecMAXPY_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMAXPY_driver.cvl $(VEC)/petscvec.c\
	  VecMAXPY.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

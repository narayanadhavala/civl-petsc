ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecMin_MPI_test.cvl $(VEC)/petscvec.c VecMin_MPI.c
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -DVecMin_MPI=VecMin_MPI_spec -DUSE_VEC_MIN -inputn=10 -sysIncludePath=$(INC) VecMin_MPI_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecMin_MPI.c

test_complex: VecMin_MPI_test.cvl $(VEC)/petscvec.c VecMin_MPI.c
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -DVecMin_MPI=VecMin_MPI_spec -DUSE_VEC_MIN -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_MPI_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecMin_MPI.c

verify_real: VecMin_MPI_driver.cvl $(VEC)/petscvec.c VecMin_MPI.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=4)\
	 -DVecMin_MPI=VecMin_MPI_spec -DUSE_VEC_MIN -sysIncludePath=$(INC) VecMin_MPI_driver.cvl $(VEC)/petscvec.c VecMin_MPI.c\
	  $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecMin_MPI_driver.cvl $(VEC)/petscvec.c VecMin_MPI.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=4)\
	 -DVecMin_MPI=VecMin_MPI_spec -DUSE_VEC_MIN -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_MPI_driver.cvl $(VEC)/petscvec.c\
	  VecMin_MPI.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

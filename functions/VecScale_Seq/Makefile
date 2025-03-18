ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecScale_Seq_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -sysIncludePath=$(INC) VecScale_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecScale_Seq_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecScale_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecScale_Seq_driver.cvl $(VEC)/petscvec.c VecScale_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecScale_Seq=VecScale_Seq_spec -sysIncludePath=$(INC) VecScale_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecScale_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

verify_complex: VecScale_Seq_driver.cvl $(VEC)/petscvec.c VecScale_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=2 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecScale_Seq=VecScale_Seq_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecScale_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecScale_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

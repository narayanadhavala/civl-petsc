ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecSet_Seq_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=1 -inputn=10 -sysIncludePath=$(INC) VecSet_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecSet_Seq_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=1 -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecSet_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecSet_Seq_driver.cvl $(VEC)/petscvec.c VecSet_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecSet_Seq=VecSet_Seq_spec -sysIncludePath=$(INC) VecSet_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecSet_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

verify_complex: VecSet_Seq_driver.cvl $(VEC)/petscvec.c VecSet_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=5 -inputN_MAX=5)\
	 -DVecSet_Seq=VecSet_Seq_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecSet_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecSet_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

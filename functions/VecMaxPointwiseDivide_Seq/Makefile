ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

verify_real: VecMaxPointwiseDivide_Seq_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=3)\
	 -DVecMaxPointwiseDivide_Seq=VecMaxPointwiseDivide_Seq_spec -sysIncludePath=$(INC) VecMaxPointwiseDivide_Seq_driver.cvl\
	  $(VEC)/petscvec.c VecMaxPointwiseDivide_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

verify_complex: VecMaxPointwiseDivide_Seq_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide_Seq.c $(INC)/petscvec.h 
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecMaxPointwiseDivide_Seq=VecMaxPointwiseDivide_Seq_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC)\
	  VecMaxPointwiseDivide_Seq_driver.cvl $(VEC)/petscvec.c VecMaxPointwiseDivide_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl 

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

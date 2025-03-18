ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test: test_real test_complex

test_real: VecMin_Seq_test.cvl $(VEC)/petscvec.c VecMin_Seq.c
	civl verify -input_mpi_nprocs=1 -DVecMin_Seq=VecMin_Seq_spec -DUSE_VEC_MIN -inputn=10 -sysIncludePath=$(INC) VecMin_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecMin_Seq.c

test_complex: VecMin_Seq_test.cvl $(VEC)/petscvec.c VecMin_Seq.c
	civl verify -input_mpi_nprocs=1 -DVecMin_Seq=VecMin_Seq_spec -DUSE_VEC_MIN -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecMin_Seq.c

verify_real: VecMin_Seq_driver.cvl $(VEC)/petscvec.c VecMin_Seq.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=4)\
	 -DVecMin_Seq=VecMin_Seq_spec -DUSE_VEC_MIN -sysIncludePath=$(INC) VecMin_Seq_driver.cvl $(VEC)/petscvec.c VecMin_Seq.c\
	  $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecMin_Seq_driver.cvl $(VEC)/petscvec.c VecMin_Seq.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=4)\
	 -DVecMin_Seq=VecMin_Seq_spec -DUSE_VEC_MIN -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecMin_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecMin_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test all

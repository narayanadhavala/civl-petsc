ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecTDot_Seq_test.cvl $(VEC)/petscvec.c
	civl verify -input_mpi_nprocs=1 -DVecTDot_Seq=VecTDot_Seq_spec -DUSE_VEC_TDOT -inputn=10 -sysIncludePath=$(INC) VecTDot_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecTDot_Seq.c

test_complex: VecTDot_Seq_test.cvl $(VEC)/petscvec.c
	civl verify -input_mpi_nprocs=1 -DVecTDot_Seq=VecTDot_Seq_spec -DUSE_VEC_TDOT -inputn=10 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecTDot_Seq_test.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl VecTDot_Seq.c

verify_real: VecTDot_Seq_driver.cvl $(VEC)/petscvec.c VecTDot_Seq.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecTDot_Seq=VecTDot_Seq_spec -DUSE_VEC_TDOT -sysIncludePath=$(INC) VecTDot_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecTDot_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecTDot_Seq_driver.cvl $(VEC)/petscvec.c VecTDot_Seq.c
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=2 -inputN_MAX=2)\
	 -DVecTDot_Seq=VecTDot_Seq_spec -DUSE_VEC_TDOT -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecTDot_Seq_driver.cvl $(VEC)/petscvec.c\
	  VecTDot_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

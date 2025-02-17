ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

#all: VecConjugate_Seq.o VecConjugate_Seq_test.exec

test: VecConjugate_Seq_test.exec
	./VecConjugate_Seq_test.exec

VecConjugate_Seq_test.exec: VecConjugate_Seq_test.c VecConjugate_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -DUSE_COMPLEX -DVecConjugate_Seq=VecConjugate_Seq_spec -I$(INC) -o VecConjugate_Seq_test.exec VecConjugate_Seq_test.c VecConjugate_Seq.o $(VEC)/petscvec.c -lm

VecConjugate_Seq.o: VecConjugate_Seq.c $(INC)/petscvec.h
	$(CC) -DUSE_COMPLEX -DVecConjugate_Seq=VecConjugate_Seq_spec -I$(INC) -c VecConjugate_Seq.c

verify_real: VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5)\
	 -sysIncludePath=$(INC) -DVecConjugate_Seq=VecConjugate_Seq_spec VecConjugate_Seq_driver.cvl\
	  $(VEC)/petscvec.c VecConjugate_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5)\
	 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) -DVecConjugate_Seq=VecConjugate_Seq_spec VecConjugate_Seq_driver.cvl \
		$(VEC)/petscvec.c VecConjugate_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all verify clean test

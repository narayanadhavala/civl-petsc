ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test: VecCopy_Seq_test.exec
	./VecCopy_Seq_test.exec

VecCopy_Seq_test.exec: VecCopy_Seq_test.c VecCopy_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -DVecCopy_Seq=VecCopy_Seq_spec -I$(INC) -o VecCopy_Seq_test.exec VecCopy_Seq_test.c VecCopy_Seq.o $(VEC)/petscvec.c -lm

VecCopy_Seq.o: VecCopy_Seq.c $(INC)/petscvec.h
	$(CC) -DVecCopy_Seq=VecCopy_Seq_spec -I$(INC) -c VecCopy_Seq.c

verify_real: VecCopy_Seq_driver.cvl $(VEC)/petscvec.c VecCopy_Seq.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5) \
	  -sysIncludePath=$(INC) -DVecCopy_Seq=VecCopy_Seq_spec VecCopy_Seq_driver.cvl $(VEC)/petscvec.c \
	  VecCopy_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecCopy_Seq_driver.cvl $(VEC)/petscvec.c VecCopy_Seq.c $(INC)/petscvec.h
	civl verify -checkTermination=true $(if $(EXTRA_INPUTS),$(EXTRA_INPUTS),-input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5) \
	  -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) -DVecCopy_Seq=VecCopy_Seq_spec VecCopy_Seq_driver.cvl $(VEC)/petscvec.c \
	  VecCopy_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all verify clean test

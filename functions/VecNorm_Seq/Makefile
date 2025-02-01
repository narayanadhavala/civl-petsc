ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

#all: clear VecNorm_Seq.o VecNorm_Seq_test.exec

test: VecNorm_Seq_test.exec
	mpirun -np $(NP) ./VecNorm_Seq_test.exec

VecNorm_Seq_test.exec: VecNorm_Seq_test.c VecNorm_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(CC) $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -I$(INC) -o VecNorm_Seq_test.exec VecNorm_Seq_test.c VecNorm_Seq.o $(VEC)/petscvec.c -lm

VecNorm_Seq.o: VecNorm_Seq.c $(INC)/petscvec.h $(PETSC)
	$(CC) $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -I$(INC) -I$(PETSC) -c VecNorm_Seq.c

verify_test: VecNorm_Seq_test.c $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=1 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecNorm_Seq_test.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(INC)/petscvec.h
	civl verify $(TYPE) -input_mpi_nprocs=1 -inputN=2 -DVecNorm_Seq=VecNorm_Seq_spec -sysIncludePath=$(INC)  VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=1 -inputN=2 $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC)  VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: all verify clean test

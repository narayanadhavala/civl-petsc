ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: VecConjugate_Seq.o VecConjugate_Seq_test.exec

test: VecConjugate_Seq_test.exec
	./VecConjugate_Seq_test.exec

VecConjugate_Seq_test.exec: VecConjugate_Seq_test.c VecConjugate_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -DUSE_COMPLEX -DVecConjugate_Seq=VecConjugate_Seq_spec -I$(INC) -o VecConjugate_Seq_test.exec VecConjugate_Seq_test.c VecConjugate_Seq.o $(VEC)/petscvec.c -lm

VecConjugate_Seq.o: VecConjugate_Seq.c $(INC)/petscvec.h
	$(CC) -DUSE_COMPLEX -DVecConjugate_Seq=VecConjugate_Seq_spec -I$(INC) -c VecConjugate_Seq.c

verify_complex: VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c $(INC)/petscvec.h
	civl verify -DUSE_COMPLEX -sysIncludePath=$(INC) -DVecConjugate_Seq=VecConjugate_Seq_spec VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c

verify_real: VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c $(INC)/petscvec.h
	civl verify -sysIncludePath=$(INC) -DVecConjugate_Seq=VecConjugate_Seq_spec VecConjugate_Seq_driver.cvl $(VEC)/petscvec.c VecConjugate_Seq.c

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all verify clean test

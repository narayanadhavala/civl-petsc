ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: VecNorm_Seq.o VecNorm_Seq_test.exec

test: VecNorm_Seq_test.exec
	./VecNorm_Seq_test.exec

VecNorm_Seq_test.exec: VecNorm_Seq_test.c VecNorm_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(CC) $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -I$(INC) -o VecNorm_Seq_test.exec VecNorm_Seq_test.c VecNorm_Seq.o $(VEC)/petscvec.c -lm

VecNorm_Seq.o: VecNorm_Seq.c $(INC)/petscvec.h
	$(CC) $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -I$(INC) -c VecNorm_Seq.c

verify_real: VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(INC)/petscvec.h
	civl verify $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -sysIncludePath=$(INC)  VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c

verify_complex: VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c $(INC)/petscvec.h
	civl verify $(TYPE) -DVecNorm_Seq=VecNorm_Seq_spec -DUSE_COMPLEX -showPathCondition=BLOCK -sysIncludePath=$(INC)  VecNorm_Seq_driver.cvl $(VEC)/petscvec.c VecNorm_Seq.c

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all verify clean test

ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: VecCopy_Seq.o VecCopy_Seq_test.exec

test: VecCopy_Seq_test.exec
	./VecCopy_Seq_test.exec

VecCopy_Seq_test.exec: VecCopy_Seq_test.c VecCopy_Seq.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -DVecCopy_Seq=VecCopy_Seq_spec -I$(INC) -o VecCopy_Seq_test.exec VecCopy_Seq_test.c VecCopy_Seq.o $(VEC)/petscvec.c -lm

VecCopy_Seq.o: VecCopy_Seq.c $(INC)/petscvec.h
	$(CC) -DVecCopy_Seq=VecCopy_Seq_spec -I$(INC) -c VecCopy_Seq.c

verify: VecCopy_Seq_driver.cvl $(VEC)/petscvec.c VecCopy_Seq.c $(INC)/petscvec.h
	civl verify -sysIncludePath=$(INC) -DVecCopy_Seq=VecCopy_Seq_spec VecCopy_Seq_driver.cvl $(VEC)/petscvec.c VecCopy_Seq.c

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all verify clean test

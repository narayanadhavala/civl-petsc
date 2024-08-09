ROOT = ../../..
include $(ROOT)/common.mk

all: petscvec.o

petscvec.o: petscvec.c $(INC)/petscvec.h
	$(CC) $(TYPE) -DVecCopy_Seq=VecCopy_Seq_spec -DVecNorm_Seq=VecNorm_Seq_spec -I$(INC) -c petscvec.c

clean:
	rm -f *.o *~ *.tmp *.exec

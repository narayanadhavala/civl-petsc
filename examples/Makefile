ROOT = ..
include $(ROOT)/common.mk
VEC = $(SRC)/vec

all: clear clean ex1.o test_ex1

verify: verify_ex1 verify_ex11

test: test_ex1 test_ex11

test_ex1: ex1.exec
	mpirun -np $(NP) gdb -ex run --args ./ex1.exec
	mpiexec -n $(NP) xterm -e gdb ./ex1.exec

test_ex11: ex11.exec
	mpirun -np $(NP) gdb -ex run --args ./ex11.exec

ex1.exec: ex1.o $(VEC)/petscvec.c  $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -o ex1.exec ex1.o $(VEC)/petscvec.o -lmpich -lm -lmpi

ex11.exec: ex11.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -o ex11.exec ex11.o $(VEC)/petscvec.o -lmpich -lm -lmpi

ex1.o: ex1.c $(INC)/petscvec.h
	$(CC) -I$(INC) -c ex1.c

ex11.o: ex11.c $(INC)/petscvec.h
	$(CC) -I$(INC) -c ex11.c

verify_ex11: ex11.c $(VEC)/petscvec.c $(INC)/petscvec.h 
	civl verify -sysIncludePath=$(INC) ex11.c $(VEC)/petscvec.c

verify_ex1: ex1.c $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -sysIncludePath=$(INC):$(CIVL) ex1.c $(VEC)/petscvec.c

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP

clear:
	clear

.PHONY: all verify verify_ex11 verify_ex1 clean test test_ex1 test_ex11

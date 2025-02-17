ROOT = ..
include $(ROOT)/common.mk
VEC = $(SRC)/vec

all: clear clean ex1.o test_ex1

verify: verify_ex1 verify_ex11

test: test_ex1 test_ex11

test_ex1: ex1.exec
#./ex1.exec
	mpirun -np $(NP) ./ex1.exec

sample_test: test.exec
	mpirun -np $(NP) ./test.exec

test.exec: test.o $(VEC)/petscvec.c  $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -o test.exec test.o $(VEC)/petscvec.o -lmpich -lm -lmpi

test.o: test.c $(INC)/petscvec.h
	$(CC) -I$(INC) -c test.c

test_ex11: ex11.exec
	mpirun -np $(NP) gdb -ex run --args ./ex11.exec

ex1.exec: ex1.o $(VEC)/petscvec.c  $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -o ex1.exec ex1.o $(VEC)/petscvec.o -lmpich -lm -lmpi

ex11.exec: ex11.o $(VEC)/petscvec.c $(INC)/petscvec.h
	$(MAKE) -C $(VEC)
	$(CC) -o ex11.exec ex11.o $(VEC)/petscvec.o -lmpich -lm -lmpi

ex1.o: ex1.c $(INC)/petscvec.h
	$(CC) -D__CIVL -I$(INC) -c ex1.c

ex11.o: ex11.c $(INC)/petscvec.h
	$(CC) -I$(INC) -c ex11.c

verify_ex11: ex11.c $(VEC)/petscvec.c $(INC)/petscvec.h 
	civl verify -input_mpi_nprocs=5 -sysIncludePath=$(INC) ex11.c $(VEC)/petscvec.c

verify_ex1: ex1.c $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -input_mpi_nprocs=5 -sysIncludePath=$(INC) ex1.c $(VEC)/petscvec.c

verify_driver: driver.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh
		civl verify -input_mpi_nprocs=4 -sysIncludePath=$(INC) driver.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP

clear:
	clear

.PHONY: all verify verify_ex11 verify_ex1 clean test test_ex1 test_ex11 sample_test

ROOT = /opt/sw/petsc
INC = $(ROOT)/include
LIB = $(ROOT)/LIB
N = 1
CC = /opt/sw/petsc/PETScV2/bin/mpicc
# CC = gcc
PETSC_CC_INCLUDES = -I/opt/sw/petsc/include -I/opt/sw/petsc/PETScV2/include
MPIEXEC = /opt/sw/petsc/PETScV2/bin/mpiexec
CC_FLAGS =  -Wall -Wwrite-strings -Wno-unknown-pragmas -Wno-lto-type-mismatch -Wno-stringop-overflow -fstack-protector -fvisibility=hidden -g3 -O0
BLASLAPACK_LIB = -Wl,-rpath,/opt/sw/petsc/PETScV2/lib -L/opt/sw/petsc/PETScV2/lib -lflapack -lfblas
PETSC_WITH_EXTERNAL_LIB = -Wl,-rpath,/opt/sw/petsc/PETScV2/lib -L/opt/sw/petsc/PETScV2/lib -Wl,-rpath,/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/gcc/x86_64-linux-gnu/13 -lpetsc -lflapack -lfblas -lm -lX11 -lmpifort -lmpi -lgfortran -lm -lgfortran -lm -lgcc_s -lquadmath -lstdc++
CIVL_INC = /opt/sw/petsc/include /opt/sw/petsc/PETScV2/include/petscconf.h
DEBUG = -start_in_debugger

all: testNormFrobenius testex34

testex34: ex34 runex34

runex34: ex34.exec
	$(MPIEXEC) -n $(N) ./ex34.exec $(DEBUG)

ex34: ex34.c $(INC)
	$(CC) $(CCFLAGS) $(PETSC_CC_INCLUDES) ex34.c -L$(PETSC_WITH_EXTERNAL_LIB) -o ex34.exec

testex18: ex18 runex18

runex18: ex18.exec
	$(MPIEXEC) -n $(N) ./ex18.exec $(DEBUG)

ex18: ex18.c $(INC)
	$(CC) $(CCFLAGS) $(PETSC_CC_INCLUDES) ex18.c -L$(PETSC_WITH_EXTERNAL_LIB) -o ex18.exec

testex43: ex43 runex43

runex43: ex43.exec
	$(MPIEXEC) -n $(N) ./ex43.exec $(DEBUG)

ex43: ex43.c $(INC)
	$(CC) $(CCFLAGS) $(PETSC_CC_INCLUDES) ex43.c -L$(PETSC_WITH_EXTERNAL_LIB) -o VecOperations.exec

testVecOperations: VecOperations runVecOperations

runVecOperations: VecOperations.exec
	$(MPIEXEC) -n $(N) ./VecOperations.exec $(DEBUG)

VecOperations: VecOperations.c $(INC)
	$(CC) $(CCFLAGS) $(PETSC_CC_INCLUDES) VecOperations.c -L$(PETSC_WITH_EXTERNAL_LIB) -o VecOperations.exec

testNormFrobenius: NormFrobenius runNormFrobenius

runNormFrobenius: NormFrobenius.exec
	$(MPIEXEC) -n $(N) ./NormFrobenius.exec

NormFrobenius: NormFrobenius.c $(INC)
	$(CC) $(CCFLAGS) -I$(INC) NormFrobenius.c -L$(PETSC_WITH_EXTERNAL_LIB) -o NormFrobenius.exec

verify: NormFrobenius.c $(INC)
	civl verify -sysIncludePath=$(CIVL_INC) NormFrobenius.c

clean:
	rm -f *~ *.o *.d *.exec

.PHONY: all testNormFrobenius testex34
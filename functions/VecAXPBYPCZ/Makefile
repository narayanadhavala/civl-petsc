ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecAXPBYPCZ_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=5 -inputn=10 -inputscal=5.0 -DVecAXPBYPCZ=VecAXPBYPCZ_spec -sysIncludePath=$(INC) VecAXPBYPCZ_test.cvl $(ROOT)/functions/VecAXPBYPCZ/VecAXPBYPCZ.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecAXPBYPCZ_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=5 -inputn=10 -inputscal=5.0 -DVecAXPBYPCZ=VecAXPBYPCZ_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAXPBYPCZ_test.cvl $(ROOT)/functions/VecAXPBYPCZ/VecAXPBYPCZ.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecAXPBYPCZ_driver.cvl $(VEC)/petscvec.c VecAXPBYPCZ.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecAXPBYPCZ=VecAXPBYPCZ_spec -sysIncludePath=$(INC) VecAXPBYPCZ_driver.cvl $(VEC)/petscvec.c VecAXPBYPCZ.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecAXPBYPCZ_driver.cvl $(VEC)/petscvec.c VecAXPBYPCZ.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecAXPBYPCZ=VecAXPBYPCZ_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAXPBYPCZ_driver.cvl $(VEC)/petscvec.c VecAXPBYPCZ.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

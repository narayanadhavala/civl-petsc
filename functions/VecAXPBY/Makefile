ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecAXPBY_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=5 -inputn=10 -inputscal=5.0 -DVecAXPBY=VecAXPBY_spec -sysIncludePath=$(INC) VecAXPBY_test.cvl $(ROOT)/functions/VecAXPBY/VecAXPBY.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecAXPBY_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=5 -inputn=10 -inputscal=5.0 -DVecAXPBY=VecAXPBY_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAXPBY_test.cvl $(ROOT)/functions/VecAXPBY/VecAXPBY.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecAXPBY_driver.cvl $(VEC)/petscvec.c VecAXPBY.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecAXPBY=VecAXPBY_spec -sysIncludePath=$(INC) VecAXPBY_driver.cvl $(VEC)/petscvec.c VecAXPBY.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecAXPBY_driver.cvl $(VEC)/petscvec.c VecAXPBY.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecAXPBY=VecAXPBY_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecAXPBY_driver.cvl $(VEC)/petscvec.c VecAXPBY.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

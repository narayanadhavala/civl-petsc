ROOT = ../..
include $(ROOT)/common.mk
INC = $(ROOT)/scaffolding/include
SRC = $(ROOT)/scaffolding/src
VEC = $(SRC)/vec

all: verify_real verify_complex

test_real: VecCopy_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputn=10 -inputscal=2.0 -DVecCopy=VecCopy_spec -sysIncludePath=$(INC) VecCopy_test.cvl $(ROOT)/functions/VecCopy/VecCopy.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

test_complex: VecCopy_test.cvl $(VEC)/petscvec.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputn=10 -inputscal=2.0 -DVecCopy=VecCopy_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecCopy_test.cvl $(ROOT)/functions/VecCopy/VecCopy.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_real: VecCopy_driver.cvl $(VEC)/petscvec.c VecCopy.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecCopy=VecCopy_spec -sysIncludePath=$(INC) VecCopy_driver.cvl $(VEC)/petscvec.c VecCopy.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

verify_complex: VecCopy_driver.cvl $(VEC)/petscvec.c VecCopy.c $(INC)/petscvec.h
	civl verify -checkTermination=true -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=2 -DVecCopy=VecCopy_spec -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) VecCopy_driver.cvl $(VEC)/petscvec.c VecCopy.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

clear:
	clear

.PHONY: clean test

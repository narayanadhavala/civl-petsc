ROOT = ..
include $(ROOT)/common.mk
VERIFY = civl verify -checkTermination

all: complex vec p2c_complex p2c_real ownership_test

complex: civlcomplex_test.cvl $(SRC)/civlcomplex.cvl $(INC)/civlcomplex.cvh $(INC)/scalars.cvh
	$(VERIFY) -DUSE_COMPLEX -DCIVL_COMPLEX -userIncludePath=$(INC) civlcomplex_test.cvl $(SRC)/civlcomplex.cvl

vec: civlvec_test.cvl $(SRC)/civlcomplex.cvl $(SRC)/civlvec.cvl $(INC)/civlvec.cvh
	$(VERIFY) -DUSE_COMPLEX -DCIVL_COMPLEX -userIncludePath=$(INC) \
  civlvec_test.cvl $(SRC)/civlcomplex.cvl $(SRC)/civlvec.cvl

p2c_complex: petscToCivl.cvl $(SRC)/petscvec.cvl $(INC)/civlcomplex.cvh \
  $(SRC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh $(INC)/scalars.cvh
	$(VERIFY) -input_mpi_nprocs=2 -input_mpi_nprocs_hi=5 -inputN_MIN=1 -inputN_MAX=3\
  -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) \
  petscToCivl.cvl $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl $(SRC)/civlcomplex.cvl

p2c_real: petscToCivl.cvl $(SRC)/petscvec.cvl $(INC)/civlcomplex.cvh\
  $(SRC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh $(INC)/scalars.cvh
	$(VERIFY) -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=5 -inputN_MIN=1 -inputN_MAX=3\
  -sysIncludePath=$(INC) petscToCivl.cvl $(SRC)/petscvec.cvl \
  $(SRC)/civlvec.cvl $(SRC)/civlcomplex.cvl

ownership_test: vectorOwnershipTest.cvl $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh
	$(VERIFY) -input_mpi_nprocs=3 -inputN=5 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) \
  vectorOwnershipTest.cvl $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl $(SRC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP

.PHONY: complex vec p2c_complex p2c_real ownership_test clean

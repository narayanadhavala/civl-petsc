ROOT = ../..
include $(ROOT)/common.mk
VEC = $(SRC)/vec
VERIFY = civl verify -checkTermination 

all: complex vec p2c_complex p2c_real

complex: civlcomplex_test.cvl $(VEC)/civlcomplex.cvl $(INC)/civlcomplex.cvh
	$(VERIFY) -DUSE_COMPLEX -DCIVL_COMPLEX -userIncludePath=$(INC) \
  civlcomplex_test.cvl $(VEC)/civlcomplex.cvl

vec: civlvec_test.cvl $(VEC)/civlcomplex.cvl $(VEC)/civlvec.cvl $(INC)/civlvec.cvh
	$(VERIFY) -DUSE_COMPLEX -DCIVL_COMPLEX -userIncludePath=$(INC) \
  civlvec_test.cvl $(VEC)/civlcomplex.cvl $(VEC)/civlvec.cvl

p2c_complex: petscToCivl.cvl $(VEC)/petscvec.c \
  $(VEC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh
	$(VERIFY) -input_mpi_nprocs=2 -input_mpi_nprocs_hi=5 \
  -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) \
  petscToCivl.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

p2c_real: petscToCivl.cvl $(VEC)/petscvec.c \
  $(VEC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh
	$(VERIFY) -input_mpi_nprocs_lo=2 -input_mpi_nprocs_hi=5 \
  -sysIncludePath=$(INC) petscToCivl.cvl $(VEC)/petscvec.c \
  $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

ownership_test: vectorOwnershipTest.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(INC)/petscvec.h $(INC)/civlvec.cvh
	$(VERIFY) -input_mpi_nprocs=3 -DUSE_COMPLEX -DCIVL_COMPLEX -sysIncludePath=$(INC) \
  vectorOwnershipTest.cvl $(VEC)/petscvec.c $(VEC)/civlvec.cvl $(VEC)/civlcomplex.cvl

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP

.PHONY: complex vec p2c_complex p2c_real clean

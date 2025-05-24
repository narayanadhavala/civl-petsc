# Makefile for CIVL verification of function VecNorm_MPI
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecNorm_MPI_driver.cvl VecNorm_MPI.c $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl
MODSC = $(MODSR) $(SRC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecNorm_MPI=VecNorm_MPI_spec
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX 

big: small

small: ver2r ver2c

ver2r: $(DEPS)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSR)

ver2c: $(DEPS)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all ver1r ver1c clean

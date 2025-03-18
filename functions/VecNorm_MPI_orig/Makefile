# Makefile for CIVL verification of function VecNorm_MPI
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecNorm_MPI_driver.cvl VecNorm_MPI.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl
MODSC = $(MODSR) $(VEC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecNorm_MPI=VecNorm_MPI_spec
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX 

all: ver2r ver2c

VecNorm_MPI.o: VecNorm_MPI.c $(INC)/petscvec.h $(PETSC)
	$(CC) $(TYPE) -DVecNorm_MPI=VecNorm_MPI_spec -I$(INC) -I$(PETSC) -c VecNorm_MPI.c

ver2r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSR)

ver2c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all ver1r ver1c clean

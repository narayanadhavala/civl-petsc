# Makefile for CIVL verification of function VecNorm
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecNorm_driver.cvl VecNorm.c $(VEC)/petscvec.cvl $(VEC)/civlvec.cvl
MODSC = $(MODSR) $(VEC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecNorm=VecNorm_spec
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX 

all: runsmall

runbig: ver1r ver1c ver2r ver2c ver3r ver3c ver4r ver4c ver5r ver5c

runsmall: versmall

# Note: add -DDEBUG to any command below to see lots of detail

ver1r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver1c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

ver2r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver2c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

ver3r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=3 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver3c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=3 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

ver4r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=4 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver4c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=4 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

ver5r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=5 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver5c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=5 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

debug: $(DEPS)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs=1 -inputN_MIN=3 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs=1 -inputN_MIN=3 -inputN_MAX=3 $(MODSC)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=3 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=3 -inputN_MAX=3 $(MODSC)

versmall: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=3 $(MODSC)
	$(VERIFYR) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=3 $(MODSC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all runbig runsmall ver1r ver1c ver2r ver2c ver3r ver3c ver4r ver4c ver5r ver5c debug versmall clean

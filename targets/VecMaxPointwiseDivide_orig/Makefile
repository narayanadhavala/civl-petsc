# Makefile for CIVL verification of function VecMaxPointwiseDivide
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecMaxPointwiseDivide_driver.cvl VecMaxPointwiseDivide.c $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl
MODSC = $(MODSR) $(SRC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecMaxPointwiseDivide=VecMaxPointwiseDivide_spec -DVecMaxPointwiseDivide_BUGGY
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX

big: small

small: ver2r ver2c

ver2r: $(DEPS)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSR)

ver2c: $(DEPS)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs=2 -inputN_MIN=2 -inputN_MAX=3 $(MODSC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: small ver2r ver2c clean

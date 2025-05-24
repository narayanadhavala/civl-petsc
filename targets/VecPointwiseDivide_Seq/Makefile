# Makefile for CIVL verification of function VecPointwiseDivide_Seq
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecPointwiseDivide_Seq_driver.cvl VecPointwiseDivide_Seq.c $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl
MODSC = $(MODSR) $(SRC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecPointwiseDivide_Seq=VecPointwiseDivide_Seq_spec
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX 
TESTR =  VecPointwiseDivide_Seq_test.cvl VecPointwiseDivide_Seq.c $(SRC)/petscvec.cvl $(SRC)/civlvec.cvl
TESTC = $(TESTR) $(SRC)/civlcomplex.cvl 

big: ver1r ver1c ver2r ver2c ver3r ver3c ver4r ver4c

# Note: add -DDEBUG to any command below to see lots of detail
small: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=3 $(MODSC)

ver1r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=4 $(MODSR)

ver1c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=4 $(MODSC)

ver2r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=4 $(MODSR)

ver2c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=2 -inputN_MIN=1 -inputN_MAX=4 $(MODSC)

ver3r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=3 -inputN_MIN=1 -inputN_MAX=4 $(MODSR)

ver3c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=3 -inputN_MIN=1 -inputN_MAX=4 $(MODSC)

ver4r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=4 -inputN_MIN=1 -inputN_MAX=4 $(MODSR)

ver4c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=4 -inputN_MIN=1 -inputN_MAX=4 $(MODSC)

ver5r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=5 -inputN_MIN=1 -inputN_MAX=4 $(MODSR)

ver5c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=5 -inputN_MIN=1 -inputN_MAX=4 $(MODSC)

debug: $(DEPS)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=3 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=3 -inputN_MAX=3 $(MODSC)

test: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=5 $(TESTR)
	$(VERIFYC) -input_mpi_nprocs_lo=1 -input_mpi_nprocs_hi=2 -inputN_MIN=1 -inputN_MAX=5 $(TESTC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: big small ver1r ver1c ver2r ver2c ver3r ver3c ver4r ver4c ver5r ver5c test debug clean

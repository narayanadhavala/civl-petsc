# Makefile for CIVL verification of function VecCopy_Seq
ROOT = ../..
include $(ROOT)/common.mk
MODSR =  VecCopy_Seq_driver.cvl VecCopy_Seq.c $(VEC)/petscvec.c $(VEC)/civlvec.cvl
MODSC = $(MODSR) $(VEC)/civlcomplex.cvl 
DEPS = $(MODSC) $(INC)/petscvec.h $(INC)/scalars.cvh $(INC)/civlcomplex.cvh \
  $(INC)/civlvec.cvh
VERIFYR = civl verify -checkTermination -sysIncludePath=$(INC) \
  -DVecCopy_Seq=VecCopy_Seq_spec
VERIFYC = $(VERIFYR) -DCIVL_COMPLEX -DUSE_COMPLEX 

all: runbig

runbig: ver1r ver1c

runsmall: versmall

# Note: add -DDEBUG to any command below to see lots of detail

VecCopy_Seq.o: VecCopy_Seq.c $(INC)/petscvec.h $(PETSC)
	$(CC) $(TYPE) -DVecCopy_Seq=VecCopy_Seq_spec -I$(INC) -I$(PETSC) -c VecCopy_Seq.c

ver1r: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5 $(MODSR)

ver1c: $(DEPS)
	$(VERIFYC) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=5 $(MODSC)

debug: $(DEPS)
	$(VERIFYR) -DDEBUG -input_mpi_nprocs=1 -inputN_MIN=3 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -DDEBUG -input_mpi_nprocs=1 -inputN_MIN=3 -inputN_MAX=3 $(MODSC)

versmall: $(DEPS)
	$(VERIFYR) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=3 $(MODSR)
	$(VERIFYC) -input_mpi_nprocs=1 -inputN_MIN=1 -inputN_MAX=3 $(MODSC)

clean:
	rm -rf *.o *~ *.tmp *.exec CIVLREP *.dSYM

.PHONY: all runbig runsmall ver1r ver1c debug versmall clean

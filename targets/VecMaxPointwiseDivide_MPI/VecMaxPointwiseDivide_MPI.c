#include <petscvec.h>
#undef VecMaxPointwiseDivide_MPI

PetscErrorCode VecMaxPointwiseDivide_MPI(Vec xin, Vec yin, PetscReal *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecMaxPointwiseDivide_MPI called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecMaxPointwiseDivide_Seq(xin, yin, z));
  PetscCallMPI(MPIU_Allreduce(z, z, 1, MPIU_REAL, MPIU_MAX,
                              PetscObjectComm((PetscObject)xin)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

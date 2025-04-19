#include <petscvec.h>
#undef VecNorm_MPI

static inline PetscErrorCode VecNorm_MPI_Default(
    Vec xin, NormType type, PetscReal *z,
    PetscErrorCode (*VecNorm_SeqFn)(Vec, NormType, PetscReal *)) {
  PetscMPIInt zn = 1;
  MPI_Op op = MPIU_SUM;

  PetscFunctionBegin;
  PetscCall(VecNorm_SeqFn(xin, type, z));
  switch (type) {
  case NORM_1_AND_2:
    // the 2 norm needs to be squared below before being summed. NORM_2 stores
    // the norm in the first slot but while NORM_1_AND_2 stores it in the second
    z[1] *= z[1];
    zn = 2;
    break;
  case NORM_FROBENIUS:
  case NORM_2:
    z[0] *= z[0];
  case NORM_1:
    break;
  case NORM_INFINITY:
    op = MPIU_MAX;
    break;
  }
  /* Change by Venkata: removed MPI_IN_PLACE reduction for MPI_REAL type struct
   * as CIVL doesn't support it */
  {
    PetscReal tmp[zn];
    PetscCall(MPIU_Allreduce(z, tmp, zn, MPIU_REAL, op,
                             PetscObjectComm((PetscObject)xin)));
    for (PetscMPIInt i = 0; i < zn; ++i)
      z[i] = tmp[i];
  }
  if (type == NORM_2 || type == NORM_FROBENIUS || type == NORM_1_AND_2)
    z[type == NORM_1_AND_2] = PetscSqrtReal(z[type == NORM_1_AND_2]);
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscReal *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecNorm_MPI\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecNorm_MPI_Default(xin, type, z, VecNorm_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

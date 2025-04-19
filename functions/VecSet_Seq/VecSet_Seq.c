#include <petscvec.h>
#undef VecSet_Seq

PetscErrorCode VecSet_Seq(Vec xin, PetscScalar alpha) {
  const PetscInt n = xin->map->n;
  PetscScalar *xx;

#ifdef DEBUG
  $print("DEBUG: Target VecSet_Seq called. alpha =", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin, &xx));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(alpha, scalar_zero))
    PetscCall(PetscArrayzero(xx, n));
  else
    for (PetscInt i = 0; i < n; i++)
      xx[i] = alpha;
  PetscCall(VecRestoreArrayWrite(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

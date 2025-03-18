#include <petscvec.h>
#undef VecSet_Seq

PetscErrorCode VecSet_Seq(Vec xin, PetscScalar alpha) {
  const PetscInt n = xin->map->n;
  PetscScalar *xx;

  PetscFunctionBegin;
  PetscCall(VecGetArrayWrite(xin, &xx));
  // if (alpha == (PetscScalar)0.0)
  if (scalar_eq(alpha, scalar_zero))
    PetscCall(PetscArrayzero(xx, n));
  else
    for (PetscInt i = 0; i < n; i++)
      xx[i] = alpha;
  PetscCall(VecRestoreArrayWrite(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecConjugate_Seq

PetscErrorCode VecConjugate_Seq(Vec xin) {
#ifdef DEBUG
  $print("DEBUG: Target VecConjugate_Seq\n");
#endif
  PetscFunctionBegin;
  if (PetscDefined(USE_COMPLEX)) {
    const PetscInt n = xin->map->n;
    PetscScalar *x;

    PetscCall(VecGetArray(xin, &x));
    for (PetscInt i = 0; i < n; ++i)
      x[i] = PetscConj(x[i]);
    PetscCall(VecRestoreArray(xin, &x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

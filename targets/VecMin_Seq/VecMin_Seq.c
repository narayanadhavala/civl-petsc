#include <petscvec.h>
#undef VecMin_Seq

static PetscErrorCode VecMinMax_Seq(Vec xin, PetscInt *idx, PetscReal *z,
                                    PetscReal minmax,
                                    int (*const cmp)(PetscReal, PetscReal)) {
  const PetscInt n = xin->map->n;
  PetscInt j = -1;

  PetscFunctionBegin;
  if (n) {
    const PetscScalar *xx;

    PetscCall(VecGetArrayRead(xin, &xx));
    minmax = PetscRealPart(xx[(j = 0)]);
    for (PetscInt i = 1; i < n; ++i) {
      const PetscReal tmp = PetscRealPart(xx[i]);

      if (cmp(tmp, minmax)) {
        j = i;
        minmax = tmp;
      }
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
  }
  *z = minmax;
  if (idx)
    *idx = j;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMin_Seq(Vec xin, PetscInt *idx, PetscReal *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecMin_Seq called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecMinMax_Seq(xin, idx, z, PETSC_MAX_REAL, VecMin_Seq_LT));
  PetscFunctionReturn(PETSC_SUCCESS);
}

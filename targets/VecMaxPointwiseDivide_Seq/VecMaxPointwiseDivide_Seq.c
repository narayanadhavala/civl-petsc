#include <petscvec.h>
#undef VecMaxPointwiseDivide_Seq

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin, Vec yin, PetscReal *max) {
#ifdef DEBUG
  $print("DEBUG: Target VecMaxPointwiseDivide_Seq: max = ", max, "\n");
#endif
  const PetscInt n = xin->map->n;
  const PetscScalar *xx, *yy;
  PetscReal m = 0.0;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xx));
  PetscCall(VecGetArrayRead(yin, &yy));
  for (PetscInt i = 0; i < n; ++i) {
    /* Change by Venkata: replaced to avoid direct scalar operations and type
     * casts, which CIVL doesn't support */
    const PetscReal v = scalar_eq(yy[i], scalar_zero)
                            ? PetscAbsScalar(xx[i])
                            : PetscAbsScalar(scalar_div(xx[i], yy[i]));

    // use a separate value to not re-evaluate side-effects
    m = PetscMax(v, m);
  }
  PetscCall(VecRestoreArrayRead(xin, &xx));
  PetscCall(VecRestoreArrayRead(yin, &yy));
  PetscCall(PetscLogFlops(n));
  *max = m;
  PetscFunctionReturn(PETSC_SUCCESS);
}

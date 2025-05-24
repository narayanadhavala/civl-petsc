#include "petscvec.h"
#undef VecAXPBY_Seq

PetscErrorCode VecAXPBY_Seq(Vec yin, PetscScalar a, PetscScalar b, Vec xin) {
#ifdef DEBUG
  $print("Target VecAXPBY_Seq: alpha=", a, " beta=", b, "\n");
#endif
  PetscFunctionBegin;
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(a, scalar_zero)) {
    PetscCall(VecScale_Seq(yin, b));
  } else if (scalar_eq(b, scalar_of(1.0))) {
    PetscCall(VecAXPY_Seq(yin, a, xin));
  } else if (scalar_eq(a, scalar_of(1.0))) {
    PetscCall(VecAYPX_Seq(yin, b, xin));
  } else {
    const PetscInt n = yin->map->n;
    const PetscScalar *xx;
    PetscScalar *yy;

    PetscCall(VecGetArrayRead(xin, &xx));
    PetscCall(VecGetArray(yin, &yy));
    if (scalar_eq(b, scalar_zero)) {
      for (PetscInt i = 0; i < n; ++i)
        yy[i] = scalar_mul(a, xx[i]);
      PetscCall(PetscLogFlops(n));
    } else {
      for (PetscInt i = 0; i < n; ++i)
        yy[i] = scalar_add(scalar_mul(a, xx[i]), scalar_mul(b, yy[i]));
      PetscCall(PetscLogFlops(3.0 * n));
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
    PetscCall(VecRestoreArray(yin, &yy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecAYPX_Seq

PetscErrorCode VecAYPX_Seq(Vec yin, PetscScalar alpha, Vec xin) {
#ifdef DEBUG
  $print("Target VecAYPX_Seq: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(alpha, scalar_zero)) {
    PetscCall(VecCopy(xin, yin));
  } else if (scalar_eq(alpha, scalar_of(1.0))) {
    PetscCall(VecAXPY_Seq(yin, alpha, xin));
  } else {
    const PetscInt n = yin->map->n;
    const PetscScalar *xx;
    PetscScalar *yy;

    PetscCall(VecGetArrayRead(xin, &xx));
    PetscCall(VecGetArray(yin, &yy));
    if (scalar_eq(alpha, scalar_of(-1.0))) {
      for (PetscInt i = 0; i < n; ++i)
        yy[i] = scalar_sub(xx[i], yy[i]);
      PetscCall(PetscLogFlops(n));
    } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_AYPX)
      fortranaypx_(&n, &alpha, xx, yy);
#else
      for (PetscInt i = 0; i < n; ++i)
        yy[i] = scalar_add(xx[i], scalar_mul(alpha, yy[i]));
#endif
      PetscCall(PetscLogFlops(2 * n));
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
    PetscCall(VecRestoreArray(yin, &yy));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

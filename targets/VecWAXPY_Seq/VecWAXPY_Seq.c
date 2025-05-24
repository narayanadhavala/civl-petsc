#include "petscvec.h"
#undef VecWAXPY_Seq

PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha, Vec xin, Vec yin) {
#ifdef DEBUG
  $print("Target VecWAXPY_Seq: alpha=", alpha, "\n");
#endif
  const PetscInt n = win->map->n;
  const PetscScalar *yy, *xx;
  PetscScalar *ww;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xx));
  PetscCall(VecGetArrayRead(yin, &yy));
  PetscCall(VecGetArray(win, &ww));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(alpha, scalar_of(1.0))) {
    PetscCall(PetscLogFlops(n));
    // could call BLAS axpy after call to memcopy, but may be slower
    for (PetscInt i = 0; i < n; i++)
      ww[i] = scalar_add(yy[i], xx[i]);
  } else if (scalar_eq(alpha, scalar_of(-1.0))) {
    PetscCall(PetscLogFlops(n));
    for (PetscInt i = 0; i < n; i++)
      ww[i] = scalar_sub(yy[i], xx[i]);
  } else if (scalar_eq(alpha, scalar_of(0.0))) {
    PetscCall(PetscArraycpy(ww, yy, n));
  } else {
    PetscCall(PetscLogFlops(2.0 * n));
#if defined(PETSC_USE_FORTRAN_KERNEL_WAXPY)
    fortranwaxpy_(&n, &alpha, xx, yy, ww);
#else
    for (PetscInt i = 0; i < n; i++)
      ww[i] = scalar_add(yy[i], scalar_mul(alpha, xx[i]));
#endif
  }
  PetscCall(VecRestoreArrayRead(xin, &xx));
  PetscCall(VecRestoreArrayRead(yin, &yy));
  PetscCall(VecRestoreArray(win, &ww));
  PetscFunctionReturn(PETSC_SUCCESS);
}

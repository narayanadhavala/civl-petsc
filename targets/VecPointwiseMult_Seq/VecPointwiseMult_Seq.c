#include <petscvec.h>
#undef VecPointwiseMult_Seq

PetscErrorCode VecPointwiseMult_Seq(Vec win, Vec xin, Vec yin) {
#ifdef DEBUG
  $print("DEBUG: Target VecPointwiseMult_Seq called\n");
#endif
  PetscInt n = win->map->n, i;
  PetscScalar *ww, *xx, *yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecGetArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecGetArray(win, &ww));
  if (ww == xx) {
    for (i = 0; i < n; i++)
      ww[i] = scalar_mul(ww[i], yy[i]);
  } else if (ww == yy) {
    for (i = 0; i < n; i++)
      ww[i] = scalar_mul(ww[i], xx[i]);
  } else {
#if defined(PETSC_USE_FORTRAN_KERNEL_XTIMESY)
    fortranxtimesy_(xx, yy, ww, &n);
#else
    for (i = 0; i < n; i++)
      ww[i] = scalar_mul(xx[i], yy[i]);
#endif
  }
  PetscCall(VecRestoreArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecRestoreArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecRestoreArray(win, &ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

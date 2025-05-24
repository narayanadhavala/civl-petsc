#include <petscvec.h>
#undef VecPointwiseDivide_Seq

static PetscScalar ScalDiv(PetscScalar x, PetscScalar y) {
  return (scalar_eq(y, scalar_zero)) ? scalar_zero : scalar_div(x, y);
}

static PetscErrorCode
VecPointwiseApply_Seq(Vec win, Vec xin, Vec yin,
                      PetscScalar (*const func)(PetscScalar, PetscScalar)) {
#ifdef DEBUG
  $print("DEBUG: VecPointwiseApply_Seq called\n");
#endif
  const PetscInt n = win->map->n;
  PetscScalar *ww, *xx, *yy; /* cannot make xx or yy const since might be ww */

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecGetArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecGetArray(win, &ww));
  for (PetscInt i = 0; i < n; ++i)
    ww[i] = func(xx[i], yy[i]);
  PetscCall(VecRestoreArrayRead(xin, (const PetscScalar **)&xx));
  PetscCall(VecRestoreArrayRead(yin, (const PetscScalar **)&yy));
  PetscCall(VecRestoreArray(win, &ww));
  PetscCall(PetscLogFlops(n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecPointwiseDivide_Seq(Vec win, Vec xin, Vec yin) {
#ifdef DEBUG
  $print("DEBUG: VecPointwiseDivide_Seq called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, ScalDiv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

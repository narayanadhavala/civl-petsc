#include <petscvec.h>
#undef VecPointwiseMax_Seq

/* Modified by Venkata- PetscMax() to scalar_max as CIVL won't support complex
 * arthematics*/
static PetscScalar MaxRealPart(PetscScalar x, PetscScalar y) {
  return scalar_max(x, y);
}

static PetscErrorCode
VecPointwiseApply_Seq(Vec win, Vec xin, Vec yin,
                      PetscScalar (*const func)(PetscScalar, PetscScalar)) {
#ifdef DEBUG
  $print("DEBUG: Spec VecPointwiseApply_Seq called\n");
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

PetscErrorCode VecPointwiseMax_Seq(Vec win, Vec xin, Vec yin) {
#ifdef DEBUG
  $print("DEBUG: Spec VecPointwiseMax_Seq called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, MaxRealPart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecPointwiseMin_Seq

/* Modified by Venkata - PetscMin() to scalar_min as CIVL won't support complex
 * arthametics*/
static PetscScalar MinRealPart(PetscScalar x, PetscScalar y) {
  return scalar_min(x, y);
}

static PetscErrorCode
VecPointwiseApply_Seq(Vec win, Vec xin, Vec yin,
                      PetscScalar (*const func)(PetscScalar, PetscScalar)) {
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

PetscErrorCode VecPointwiseMin_Seq(Vec win, Vec xin, Vec yin) {
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Seq(win, xin, yin, MinRealPart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

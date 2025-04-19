#include <petscvec.h>
#undef VecCopy

PetscErrorCode VecCopyAsync_Private(Vec x, Vec y, PetscDeviceContext dctx) {
  PetscBool flgs[4];
  PetscReal norms[4] = {0.0, 0.0, 0.0, 0.0};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  if (x == y)
    PetscFunctionReturn(PETSC_SUCCESS);
  VecCheckSameLocalSize(x, 1, y, 2);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(y, 2));

#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i = 0; i < 4; i++) {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i],
                                             &norms[i], &flgs[i]));
  }
#endif

  PetscCall(PetscLogEventBegin(VEC_Copy, x, y, 0, 0));
#if defined(PETSC_USE_MIXED_PRECISION)
  extern PetscErrorCode VecGetArray(Vec, double **);
  extern PetscErrorCode VecRestoreArray(Vec, double **);
  extern PetscErrorCode VecGetArray(Vec, float **);
  extern PetscErrorCode VecRestoreArray(Vec, float **);
  extern PetscErrorCode VecGetArrayRead(Vec, const double **);
  extern PetscErrorCode VecRestoreArrayRead(Vec, const double **);
  extern PetscErrorCode VecGetArrayRead(Vec, const float **);
  extern PetscErrorCode VecRestoreArrayRead(Vec, const float **);
  if ((((PetscObject)x)->precision == PETSC_PRECISION_SINGLE) &&
      (((PetscObject)y)->precision == PETSC_PRECISION_DOUBLE)) {
    PetscInt i, n;
    const float *xx;
    double *yy;
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(y, &yy));
    PetscCall(VecGetLocalSize(x, &n));
    for (i = 0; i < n; i++)
      yy[i] = xx[i];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
  } else if ((((PetscObject)x)->precision == PETSC_PRECISION_DOUBLE) &&
             (((PetscObject)y)->precision == PETSC_PRECISION_SINGLE)) {
    PetscInt i, n;
    float *yy;
    const double *xx;
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(y, &yy));
    PetscCall(VecGetLocalSize(x, &n));
    for (i = 0; i < n; i++)
      yy[i] = (float)xx[i];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
  } else {
    PetscUseTypeMethod(x, copy, y);
  }
#else
  VecMethodDispatch(x, dctx, VecAsyncFnName(Copy), copy,
                    (Vec, Vec, PetscDeviceContext), y);
#endif

  PetscCall(PetscObjectStateIncrease((PetscObject)y));
#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i = 0; i < 4; i++) {
    if (flgs[i]) {
      PetscCall(
          PetscObjectComposedDataSetReal((PetscObject)y, NormIds[i], norms[i]));
    }
  }
#endif

  PetscCall(PetscLogEventEnd(VEC_Copy, x, y, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCopy(Vec x, Vec y) {
#ifdef DEBUG
  $print("Target VecCopy: x=", x, " y=", y, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecCopyAsync_Private(x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecAYPX

PetscErrorCode VecAYPXAsync_Private(Vec y, PetscScalar beta, Vec x,
                                    PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecAYPXAsync_Private: beta=", beta, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 1, y, 3);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, beta, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (x == y) {
    PetscCall(VecScale(y, scalar_add(beta, scalar_of(1))));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  if (scalar_eq(scalar_of(0.0), beta)) {
    PetscCall(VecCopy(x, y));
  } else {
    PetscCall(PetscLogEventBegin(VEC_AYPX, x, y, 0, 0));
    VecMethodDispatch(y, dctx, VecAsyncFnName(AYPX), aypx,
                      (Vec, PetscScalar, Vec, PetscDeviceContext), beta, x);
    PetscCall(PetscLogEventEnd(VEC_AYPX, x, y, 0, 0));
    PetscCall(PetscObjectStateIncrease((PetscObject)y));
  }
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
#ifdef DEBUG
  $print("Target VecAYPX: beta=", beta, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecAYPXAsync_Private(y, beta, x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

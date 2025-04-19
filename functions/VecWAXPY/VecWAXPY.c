#include "petscvec.h"
#undef VecWAXPY

PetscErrorCode VecWAXPYAsync_Private(Vec w, PetscScalar alpha, Vec x, Vec y,
                                     PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecWAXPYAsync_Private: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 4);
  PetscValidType(w, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 4);
  PetscCheckSameTypeAndComm(x, 3, y, 4);
  PetscCheckSameTypeAndComm(y, 4, w, 1);
  VecCheckSameSize(x, 3, y, 4);
  VecCheckSameSize(x, 3, w, 1);
  PetscCheck(
      w != y, PETSC_COMM_SELF, PETSC_ERR_SUP,
      "Result vector w cannot be same as input vector y, suggest VecAXPY()");
  PetscCheck(
      w != x, PETSC_COMM_SELF, PETSC_ERR_SUP,
      "Result vector w cannot be same as input vector x, suggest VecAYPX()");
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscCall(VecSetErrorIfLocked(w, 1));

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(scalar_of(0), alpha)) {
    PetscCall(VecCopyAsync_Private(y, w, dctx));
  } else {
    PetscCall(PetscLogEventBegin(VEC_WAXPY, x, y, w, 0));
    VecMethodDispatch(w, dctx, VecAsyncFnName(WAXPY), waxpy,
                      (Vec, PetscScalar, Vec, Vec, PetscDeviceContext), alpha,
                      x, y);
    PetscCall(PetscLogEventEnd(VEC_WAXPY, x, y, w, 0));
    PetscCall(PetscObjectStateIncrease((PetscObject)w));
  }
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
#ifdef DEBUG
  $print("Target VecWAXPY: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecWAXPYAsync_Private(w, alpha, x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

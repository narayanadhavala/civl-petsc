#include <petscvec.h>
#undef VecAYPX

PetscErrorCode VecAYPXAsync_Private(Vec y, PetscScalar beta, Vec x,
                                    PetscDeviceContext dctx) {
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
  // PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    // used the scalar_add to add two complex numbers
    PetscCall(VecScale(y, scalar_add(beta, scalar_of(1))));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  if ($is_scalar_zero(beta)) {
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
  PetscFunctionBegin;
  PetscCall(VecAYPXAsync_Private(y, beta, x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

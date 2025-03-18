#include <petscvec.h>
#undef VecAXPY

PetscErrorCode VecAXPYAsync_Private(Vec y, PetscScalar alpha, Vec x,
                                    PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecAXPYAsync_Private: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 3);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 3, y, 1);
  VecCheckSameSize(x, 3, y, 1);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  // used the scalar_eq to compare the complex & real numbers
  if (scalar_eq(scalar_of(0.0), alpha))
    PetscFunctionReturn(PETSC_SUCCESS);
  // PetscCall(VecSetErrorIfLocked(y, 1));
  if (x == y) {
    // used scalar_add to add two complex numbers
    PetscCall(VecScale(y, scalar_add(alpha, scalar_of(1.0))));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_AXPY, x, y, 0, 0));
  VecMethodDispatch(y, dctx, VecAsyncFnName(AXPY), axpy,
                    (Vec, PetscScalar, Vec, PetscDeviceContext), alpha, x);
  PetscCall(PetscLogEventEnd(VEC_AXPY, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
#ifdef DEBUG
  $print("Target VecAXPY: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecAXPYAsync_Private(y, alpha, x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

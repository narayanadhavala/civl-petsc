#include "petscvec.h"
#undef VecAXPBY

/*modified only to use the CIVL's scalar_add and PetscScalar_is_zero*/
PetscErrorCode VecAXPBYAsync_Private(Vec y, PetscScalar alpha, PetscScalar beta,
                                     Vec x, PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecAXPBYAsync_Private: alpha=", alpha," beta=",beta,"\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  PetscValidType(x, 4);
  PetscValidType(y, 1);
  PetscCheckSameTypeAndComm(x, 4, y, 1);
  VecCheckSameSize(y, 1, x, 4);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveScalar(y, alpha, 2);
  PetscValidLogicalCollectiveScalar(y, beta, 3);
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(scalar_of(0.0), alpha) && scalar_eq(scalar_of(1.0), beta))
    PetscFunctionReturn(PETSC_SUCCESS);
  if (x == y) {
    PetscCall(VecScale(y, scalar_add(alpha, beta)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_AXPY, y, x, 0, 0));
  VecMethodDispatch(y, dctx, VecAsyncFnName(AXPBY), axpby,
                    (Vec, PetscScalar, PetscScalar, Vec, PetscDeviceContext),
                    alpha, beta, x);
  PetscCall(PetscLogEventEnd(VEC_AXPY, y, x, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x) {
#ifdef DEBUG
  $print("Target VecAXPBY: alpha=", alpha," beta=",beta,"\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecAXPBYAsync_Private(y, alpha, beta, x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

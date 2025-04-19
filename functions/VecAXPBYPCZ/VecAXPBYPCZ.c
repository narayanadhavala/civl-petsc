#include "petscvec.h"
#undef VecAXPBYPCZ

PetscErrorCode VecAXPBYPCZAsync_Private(Vec z, PetscScalar alpha,
                                        PetscScalar beta, PetscScalar gamma,
                                        Vec x, Vec y, PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecAXPBYPCZAsync_Private: alpha=", alpha," beta=",beta," gamma=",gamma,"\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(z, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 5);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 6);
  PetscValidType(z, 1);
  PetscValidType(x, 5);
  PetscValidType(y, 6);
  PetscCheckSameTypeAndComm(x, 5, y, 6);
  PetscCheckSameTypeAndComm(x, 5, z, 1);
  VecCheckSameSize(x, 5, y, 6);
  VecCheckSameSize(x, 5, z, 1);
  PetscCheck(x != y && x != z, PetscObjectComm((PetscObject)x),
             PETSC_ERR_ARG_IDN, "x, y, and z must be different vectors");
  PetscCheck(y != z, PetscObjectComm((PetscObject)y), PETSC_ERR_ARG_IDN,
             "x, y, and z must be different vectors");
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  VecCheckAssembled(z);
  PetscValidLogicalCollectiveScalar(z, alpha, 2);
  PetscValidLogicalCollectiveScalar(z, beta, 3);
  PetscValidLogicalCollectiveScalar(z, gamma, 4);
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(scalar_of(0), alpha) && scalar_eq(scalar_of(0), beta) &&
      scalar_eq(scalar_of(1), gamma)) {
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(VecSetErrorIfLocked(z, 1));
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_AXPBYPCZ, x, y, z, 0));
  VecMethodDispatch(z, dctx, VecAsyncFnName(AXPBYPCZ), axpbypcz,
                    (Vec, PetscScalar, PetscScalar, PetscScalar, Vec, Vec,
                     PetscDeviceContext),
                    alpha, beta, gamma, x, y);
  PetscCall(PetscLogEventEnd(VEC_AXPBYPCZ, x, y, z, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)z));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y) {
#ifdef DEBUG
  $print("Target VecAXPBYPCZ: alpha=", alpha," beta=",beta,"gamma=",gamma,"\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecAXPBYPCZAsync_Private(z, alpha, beta, gamma, x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

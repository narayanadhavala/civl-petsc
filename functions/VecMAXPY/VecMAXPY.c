#include <petscvec.h>
#undef VecMAXPY

PetscErrorCode VecMAXPYAsync_Private(Vec y, PetscInt nv,
                                     const PetscScalar alpha[], Vec x[],
                                     PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("Target VecMAXPYAsync_Private: alpha[0]=", alpha[0], " nv =", nv,
         "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveInt(y, nv, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCheck(nv >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "Number of vectors (given %" PetscInt_FMT ") cannot be negative",
             nv);
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (nv) {
    PetscInt zeros = 0;

    PetscAssertPointer(alpha, 3);
    PetscAssertPointer(x, 4);
    for (PetscInt i = 0; i < nv; ++i) {
      PetscValidLogicalCollectiveScalar(y, alpha[i], 3);
      PetscValidHeaderSpecific(x[i], VEC_CLASSID, 4);
      PetscValidType(x[i], 4);
      PetscCheckSameTypeAndComm(y, 1, x[i], 4);
      VecCheckSameSize(y, 1, x[i], 4);
      PetscCheck(y != x[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Array of vectors 'x' cannot contain y, found x[%" PetscInt_FMT
                 "] == y",
                 i);
      VecCheckAssembled(x[i]);
      PetscCall(VecLockReadPush(x[i]));
      zeros += scalar_eq(alpha[i], scalar_zero);
    }

    if (zeros < nv) {
      PetscCall(PetscLogEventBegin(VEC_MAXPY, y, *x, 0, 0));
      VecMethodDispatch(
          y, dctx, VecAsyncFnName(MAXPY), maxpy,
          (Vec, PetscInt, const PetscScalar[], Vec[], PetscDeviceContext), nv,
          alpha, x);
      PetscCall(PetscLogEventEnd(VEC_MAXPY, y, *x, 0, 0));
      PetscCall(PetscObjectStateIncrease((PetscObject)y));
    }

    for (PetscInt i = 0; i < nv; ++i)
      PetscCall(VecLockReadPop(x[i]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[],
                        Vec x[]) {
#ifdef DEBUG
  $print("Target VecMAXPY: alpha[0]=", alpha[0], " nv =", nv, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecMAXPYAsync_Private(y, nv, alpha, x, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

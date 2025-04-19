#include <petscvec.h>
#undef VecMTDot
PETSC_EXTERN PetscLogEvent VEC_MTDot;

PetscErrorCode VecMXDot_Private(
    Vec x, PetscInt nv, const Vec y[], PetscScalar result[],
    PetscErrorCode (*mxdot)(Vec, PetscInt, const Vec[], PetscScalar[]),
    PetscLogEvent event) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveInt(x, nv, 2);
  if (!nv)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(y, 3);
  for (PetscInt i = 0; i < nv; ++i) {
    PetscValidHeaderSpecific(y[i], VEC_CLASSID, 3);
    PetscValidType(y[i], 3);
    PetscCheckSameTypeAndComm(x, 1, y[i], 3);
    VecCheckSameSize(x, 1, y[i], 3);
    VecCheckAssembled(y[i]);
    PetscCall(VecLockReadPush(y[i]));
  }
  PetscAssertPointer(result, 4);
  PetscValidFunction(mxdot, 5);

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(event, x, *y, 0, 0));
  // CIVL expects to see a function pointer used for a function call
  PetscCall((mxdot)(x, nv, y, result));
  PetscCall(PetscLogEventEnd(event, x, *y, 0, 0));
  PetscCall(VecLockReadPop(x));
  for (PetscInt i = 0; i < nv; ++i)
    PetscCall(VecLockReadPop(y[i]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
#ifdef DEBUG
  $print("DEBUG: Target VecMTDot: nv", nv, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscCall(VecMXDot_Private(x, nv, y, val, x->ops->mtdot, VEC_MTDot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

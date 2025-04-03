#include <petscvec.h>
#undef VecTDot

PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val) {
#ifdef DEBUG
  $print("Target VecTDot \n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscAssertPointer(val, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);

  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscCall(PetscLogEventBegin(VEC_TDot, x, y, 0, 0));
  PetscUseTypeMethod(x, tdot, y, val);
  PetscCall(PetscLogEventEnd(VEC_TDot, x, y, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

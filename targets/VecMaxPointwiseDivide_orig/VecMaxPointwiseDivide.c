#include <petscvec.h>
#undef VecMaxPointwiseDivide

PetscErrorCode VecMaxPointwiseDivide(Vec x, Vec y, PetscReal *max) {
#ifdef DEBUG
  $print("DEBUG: Target VecMaxPointwiseDivide\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscAssertPointer(max, 3);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecLockReadPush(x));
  PetscCall(VecLockReadPush(y));
  PetscUseTypeMethod(x, maxpointwisedivide, y, max);
  PetscCall(VecLockReadPop(x));
  PetscCall(VecLockReadPop(y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecMax

PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  if (p)
    PetscAssertPointer(p, 2);
  PetscAssertPointer(val, 3);
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Max, x, 0, 0, 0));
  PetscUseTypeMethod(x, max, p, val);
  PetscCall(PetscLogEventEnd(VEC_Max, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

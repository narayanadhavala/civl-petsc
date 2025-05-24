#include <petscvec.h>
#undef VecMin

PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Target VecMin: val =", val, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  if (p)
    PetscAssertPointer(p, 2);
  PetscAssertPointer(val, 3);
  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Min, x, 0, 0, 0));
  PetscUseTypeMethod(x, min, p, val);
  PetscCall(PetscLogEventEnd(VEC_Min, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

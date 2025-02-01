#include <petscvec.h>
#undef VecSetValuesBlocked

PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora) {
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ix, 3);
  PetscAssertPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvaluesblocked, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

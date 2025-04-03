#include "petscvec.h"
#undef VecSetValues

PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            const PetscScalar y[], InsertMode iora) {
#ifdef DEBUG
  $print("DEBUG: Target VecSetValues called\n");
#endif
  PetscFunctionBeginHot;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ix, 3);
  PetscAssertPointer(y, 4);
  PetscValidType(x, 1);

  PetscCall(PetscLogEventBegin(VEC_SetValues, x, 0, 0, 0));
  PetscUseTypeMethod(x, setvalues, ni, ix, y, iora);
  PetscCall(PetscLogEventEnd(VEC_SetValues, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecGetValues

PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            PetscScalar y[]) {
#ifdef DEBUG
  $print("DEBUG: Target VecGetValues: ni=", ni, "ix=", ix, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  if (!ni)
    PetscFunctionReturn(PETSC_SUCCESS);
  PetscAssertPointer(ix, 3);
  PetscAssertPointer(y, 4);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscUseTypeMethod(x, getvalues, ni, ix, y);
  PetscFunctionReturn(PETSC_SUCCESS);
}

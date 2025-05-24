#include <petscvec.h>
#undef VecGetSize

PetscErrorCode VecGetSize(Vec x, PetscInt *size) {
#ifdef DEBUG
  $print("DEBUG: Target VecGetSize\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscAssertPointer(size, 2);
  PetscValidType(x, 1);
  PetscUseTypeMethod(x, getsize, size);
  PetscFunctionReturn(PETSC_SUCCESS);
}

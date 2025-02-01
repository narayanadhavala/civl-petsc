#include <petscvec.h>
#undef VecCopy

PetscErrorCode VecCopy(Vec x, Vec y) {
  PetscFunctionBegin;
  PetscCall(VecCopyAsync_Private(x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

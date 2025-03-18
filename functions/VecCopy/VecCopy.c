#include <petscvec.h>
#undef VecCopy

PetscErrorCode VecCopy(Vec x, Vec y) {
#ifdef DEBUG
  $print("Target VecCopy: x=", x," y=", y, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecCopyAsync_Private(x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

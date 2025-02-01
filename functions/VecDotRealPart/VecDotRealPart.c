#include <petscvec.h>
#undef VecDotRealPart

PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val) {
  PetscScalar fdot;

  PetscFunctionBegin;
  PetscCall(VecDot(x, y, &fdot));
  *val = PetscRealPart(fdot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

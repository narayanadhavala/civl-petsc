#include <petscvec.h>
#undef VecDotRealPart

PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Target VecDotRealPart: x=", x, ", y=", y, "\n");
#endif
  PetscScalar fdot;

  PetscFunctionBegin;
  PetscCall(VecDot(x, y, &fdot));
  *val = PetscRealPart(fdot);
  PetscFunctionReturn(PETSC_SUCCESS);
}

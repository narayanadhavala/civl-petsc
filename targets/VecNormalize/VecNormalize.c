#include <petscvec.h>
#undef VecNormalize

PetscErrorCode VecNormalize(Vec x, PetscReal *val) {
  PetscReal norm;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscCall(VecSetErrorIfLocked(x, 1));
  if (val)
    PetscAssertPointer(val, 2);
  PetscCall(PetscLogEventBegin(VEC_Normalize, x, 0, 0, 0));
  PetscCall(VecNorm(x, NORM_2, &norm));
  if (norm == 0.0) {
    PetscCall(PetscInfo(x, "Vector of zero norm can not be normalized; "
                           "Returning only the zero norm\n"));
  } else if (PetscIsInfOrNanReal(norm)) {
    PetscCall(PetscInfo(x, "Vector with Inf or Nan norm can not be normalized; "
                           "Returning only the norm\n"));
  } else {
    PetscScalar s = scalar_of(1.0 / norm);
    PetscCall(VecScale(x, s));
  }
  PetscCall(PetscLogEventEnd(VEC_Normalize, x, 0, 0, 0));
  if (val)
    *val = norm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

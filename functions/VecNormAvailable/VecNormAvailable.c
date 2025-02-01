#include <petscvec.h>

PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available,
                                PetscReal *val) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  PetscAssertPointer(available, 3);
  PetscAssertPointer(val, 4);

  if (type == NORM_1_AND_2) {
    *available = PETSC_FALSE;
  } else {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[type], val,
                                             available));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

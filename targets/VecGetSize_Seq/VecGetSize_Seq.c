#include <petscvec.h>
#undef VecGetSize_Seq

PetscErrorCode VecGetSize_Seq(Vec vin, PetscInt *size) {
#ifdef DEBUG
  $print("DEBUG: Target VecGetSize_Seq\n");
#endif
  PetscFunctionBegin;
  *size = vin->map->n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

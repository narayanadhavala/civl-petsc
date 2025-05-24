#include <petscvec.h>
#undef VecGetSize_MPI

PetscErrorCode VecGetSize_MPI(Vec xin, PetscInt *N) {
#ifdef DEBUG
  $print("DEBUG: Target VecGetSize_MPI\n");
#endif
  PetscFunctionBegin;
  *N = xin->map->N;
  PetscFunctionReturn(PETSC_SUCCESS);
}

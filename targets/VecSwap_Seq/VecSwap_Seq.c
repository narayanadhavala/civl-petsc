#include <petscvec.h>
#undef VecSwap_Seq

PetscErrorCode VecSwap_Seq(Vec xin, Vec yin) {
#ifdef DEBUG
  $print("DEBUG: Target VecSwap_Seq called\n");
#endif
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscBLASInt one = 1;
    PetscScalar *ya, *xa;
    PetscBLASInt bn;

    PetscCall(PetscBLASIntCast(xin->map->n, &bn));
    PetscCall(VecGetArray(xin, &xa));
    PetscCall(VecGetArray(yin, &ya));
    PetscCallBLAS("BLASswap", BLASswap_(&bn, xa, &one, ya, &one));
    PetscCall(VecRestoreArray(xin, &xa));
    PetscCall(VecRestoreArray(yin, &ya));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

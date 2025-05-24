#include <petscvec.h>
#undef VecCopy_Seq

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin)
{
#ifdef DEBUG
  $print("Target VecCopy_Seq: xin=", xin," yin=", yin, "\n");
#endif
  PetscFunctionBegin;
  if (xin != yin) {
    const PetscScalar *xa;
    PetscScalar       *ya;

    PetscCall(VecGetArrayRead(xin, &xa));
    PetscCall(VecGetArray(yin, &ya));
    PetscCall(PetscArraycpy(ya, xa, xin->map->n));
    PetscCall(VecRestoreArrayRead(xin, &xa));
    PetscCall(VecRestoreArray(yin, &ya));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

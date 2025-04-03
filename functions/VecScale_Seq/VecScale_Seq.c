#include <petscvec.h>
#undef VecScale_Seq

PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha) {
#ifdef DEBUG
  $print("DEBUG: Target VecScale_Seq Alpha = ", alpha, "\n");
#endif
  PetscFunctionBegin;
  // if (alpha == (PetscScalar)0.0)
  if (scalar_eq(alpha, scalar_zero)) {
    PetscCall(VecSet_Seq(xin, alpha));
  } else if (!scalar_eq(alpha, scalar_of(1.0))) {
    const PetscBLASInt one = 1;
    PetscBLASInt bn;
    PetscScalar *xarray;

    PetscCall(PetscBLASIntCast(xin->map->n, &bn));
    PetscCall(PetscLogFlops(bn));
    PetscCall(VecGetArray(xin, &xarray));
    PetscCallBLAS("BLASscal", BLASscal_(&bn, &alpha, xarray, &one));
    PetscCall(VecRestoreArray(xin, &xarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecAXPY_Seq

PetscErrorCode VecAXPY_Seq(Vec yin, PetscScalar alpha, Vec xin) {
#ifdef DEBUG
  $print("DEBUG: Target VecAXPY_Seq: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  /* assume that the BLAS handles alpha == 1.0 efficiently since we have no fast
   * code for it */
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (!scalar_eq(alpha, scalar_zero)) {
    const PetscScalar *xarray;
    PetscScalar *yarray;
    const PetscBLASInt one = 1;
    PetscBLASInt bn;

    PetscCall(PetscBLASIntCast(yin->map->n, &bn));
    PetscCall(PetscLogFlops(2.0 * bn));
    PetscCall(VecGetArrayRead(xin, &xarray));
    PetscCall(VecGetArray(yin, &yarray));
    PetscCallBLAS("BLASaxpy",
                  BLASaxpy_(&bn, &alpha, xarray, &one, yarray, &one));
    PetscCall(VecRestoreArrayRead(xin, &xarray));
    PetscCall(VecRestoreArray(yin, &yarray));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include "petscvec.h"
#undef VecAXPBYPCZ_Seq

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscScalar alpha, PetscScalar beta,
                               PetscScalar gamma, Vec xin, Vec yin) {
#ifdef DEBUG
  $print("Target VecAXPBYPCZ_Seq: alpha=", alpha, " beta=", beta,
         " gamma=", gamma, "\n");
#endif
  const PetscInt n = zin->map->n;
  const PetscScalar *yy, *xx;
  PetscInt flops = 4 * n; // common case
  PetscScalar *zz;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xin, &xx));
  PetscCall(VecGetArrayRead(yin, &yy));
  PetscCall(VecGetArray(zin, &zz));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(alpha, scalar_of(1.0))) {
    for (PetscInt i = 0; i < n; ++i) {
      zz[i] = scalar_add(scalar_add(xx[i], scalar_mul(beta, yy[i])),
                         scalar_mul(gamma, zz[i]));
    }
  } else if (scalar_eq(gamma, scalar_of(1.0))) {
    for (PetscInt i = 0; i < n; ++i) {
      // zz[i] = alpha * xx[i] + beta * yy[i] + zz[i];
      zz[i] = scalar_add(
          scalar_add(scalar_mul(alpha, xx[i]), scalar_mul(beta, yy[i])), zz[i]);
    }
  } else if (scalar_eq(gamma, scalar_of(0.0))) {
    for (PetscInt i = 0; i < n; ++i)
      zz[i] = scalar_add(scalar_mul(alpha, xx[i]), scalar_mul(beta, yy[i]));
    flops -= n;
  } else {
    for (PetscInt i = 0; i < n; ++i) {
      zz[i] = scalar_add(
          scalar_add(scalar_mul(alpha, xx[i]), scalar_mul(beta, yy[i])),
          scalar_mul(gamma, zz[i]));
    }
    flops += n;
  }
  PetscCall(VecRestoreArrayRead(xin, &xx));
  PetscCall(VecRestoreArrayRead(yin, &yy));
  PetscCall(VecRestoreArray(zin, &zz));
  PetscCall(PetscLogFlops(flops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

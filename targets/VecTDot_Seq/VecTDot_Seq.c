#include <petscvec.h>
#undef VecTDot_Seq

static inline PetscScalar BLASdotu_(const PetscBLASInt *n, const PetscScalar *x,
                                    const PetscBLASInt *sx,
                                    const PetscScalar *y,
                                    const PetscBLASInt *sy) {
  PetscScalar sum = scalar_zero;
  PetscInt i, j, k;

#ifdef DEBUG
  $print("Target BLASdotu n=", *n, "\n");
#endif
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (*sx == 1 && *sy == 1)
    for (i = 0; i < *n; i++)
      sum = scalar_add(sum, scalar_mul(x[i], y[i]));
  else
    for (i = 0, j = 0, k = 0; i < *n; i++, j += *sx, k += *sy)
      sum = scalar_add(sum, scalar_mul(x[j], y[k]));
  return sum;
}

static PetscErrorCode VecXDot_Seq_Private(
    Vec xin, Vec yin, PetscScalar *z,
    PetscScalar (*const BLASfn)(const PetscBLASInt *, const PetscScalar *,
                                const PetscBLASInt *, const PetscScalar *,
                                const PetscBLASInt *)) {
  const PetscInt n = xin->map->n;
  const PetscBLASInt one = 1;
  const PetscScalar *ya, *xa;
  PetscBLASInt bn;

#ifdef DEBUG
  $print("Target VecXDot_Seq_Private \n");
#endif

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(n, &bn));
  if (n > 0)
    PetscCall(PetscLogFlops(2.0 * n - 1));
  PetscCall(VecGetArrayRead(xin, &xa));
  PetscCall(VecGetArrayRead(yin, &ya));

  // arguments ya, xa are reversed because BLAS complex conjugates the first
  //   argument, PETSc the second
  PetscCallBLAS("BLASdot", *z = BLASfn(&bn, ya, &one, xa, &one));
  PetscCall(VecRestoreArrayRead(xin, &xa));
  PetscCall(VecRestoreArrayRead(yin, &ya));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecTDot_Seq(Vec xin, Vec yin, PetscScalar *z) {
#ifdef DEBUG
  $print("Target VecTDot_Seq \n");
#endif
  PetscFunctionBegin;
  /*
    pay close attention!!! xin and yin are SWAPPED here so that the eventual
    BLAS call is dot(&bn, xa, &one, ya, &one)
  */
  PetscCall(VecXDot_Seq_Private(yin, xin, z, BLASdotu_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

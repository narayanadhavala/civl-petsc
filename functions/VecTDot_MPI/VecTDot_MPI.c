#include <petscvec.h>
#undef VecTDot_MPI

static inline PetscScalar BLASdotu_(const PetscBLASInt *n, const PetscScalar *x,
                                    const PetscBLASInt *sx,
                                    const PetscScalar *y,
                                    const PetscBLASInt *sy) {
  PetscScalar sum = scalar_zero;
  PetscInt i, j, k;
  if (*sx == 1 && *sy == 1) {
    for (i = 0; i < *n; i++) {
      // sum += x[i] * y[i];
      sum = scalar_add(sum, scalar_mul(x[i], y[i]));
    }
  } else {
    for (i = 0, j = 0, k = 0; i < *n; i++, j += *sx, k += *sy) {
      // sum += x[j] * y[k];
      sum = scalar_add(sum, scalar_mul(x[j], y[k]));
    }
  }
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

static inline PetscErrorCode
VecXDot_MPI_Default(Vec xin, Vec yin, PetscScalar *z,
                    PetscErrorCode (*VecXDot_SeqFn)(Vec, Vec, PetscScalar *)) {
  PetscFunctionBegin;
  PetscCall(VecXDot_SeqFn(xin, yin, z));
#ifdef USE_COMPLEX
  {
    PetscReal local[2], tmp[2];
    // Copy the real and imaginary parts from the computed complex result
    local[0] = (*z).real;
    local[1] = (*z).imag;
    // Use MPI_Allreduce on the temporary double array
    PetscCall(MPIU_Allreduce(local, tmp, 2, MPI_DOUBLE, MPIU_SUM,
                             PetscObjectComm((PetscObject)xin)));
    *z = scalar_make(tmp[0], tmp[1]);
  }
#else
  {
    PetscReal tmp[1];
    PetscCall(MPIU_Allreduce(z, tmp, 1, MPIU_REAL, MPIU_SUM,
                             PetscObjectComm((PetscObject)xin)));
    *z = tmp[0];
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecTDot_MPI(Vec xin, Vec yin, PetscScalar *z) {
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecTDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}
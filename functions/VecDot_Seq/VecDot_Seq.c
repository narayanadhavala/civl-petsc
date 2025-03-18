#include <petscvec.h>
#undef VecDot_Seq

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

PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscScalar *z) {
#ifdef DEBUG
  $print("Target VecDot_Seq: xin=", xin, ", yin=", yin, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecXDot_Seq_Private(xin, yin, z, BLASdot_));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecMAXPY_Seq

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv, const PetscScalar *alpha,
                            Vec *y) {
  const PetscInt j_rem = nv & 0x3, n = xin->map->n;
  const PetscScalar *yptr[4];
  PetscScalar *xx;
#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*xx, **yptr, *aptr)
#endif

#ifdef DEBUG
  $print("Target VecMAXPY_Seq: alpha[0]=", alpha[0]," nv =",nv,"\n");
#endif

  PetscFunctionBegin;
  PetscCall(PetscLogFlops(nv * 2.0 * n));
  PetscCall(VecGetArray(xin, &xx));
  for (PetscInt i = 0; i < j_rem; ++i)
    PetscCall(VecGetArrayRead(y[i], yptr + i));
  switch (j_rem) {
  case 3:
    PetscKernelAXPY3(xx, alpha[0], alpha[1], alpha[2], yptr[0], yptr[1],
                     yptr[2], n);
    break;
  case 2:
    PetscKernelAXPY2(xx, alpha[0], alpha[1], yptr[0], yptr[1], n);
    break;
  case 1:
    PetscKernelAXPY(xx, alpha[0], yptr[0], n);
  default:
    break;
  }
  for (PetscInt i = 0; i < j_rem; ++i)
    PetscCall(VecRestoreArrayRead(y[i], yptr + i));
  alpha += j_rem;
  y += j_rem;
  for (PetscInt j = j_rem, inc = 4; j < nv; j += inc, alpha += inc, y += inc) {
    for (PetscInt i = 0; i < inc; ++i)
      PetscCall(VecGetArrayRead(y[i], yptr + i));
    PetscKernelAXPY4(xx, alpha[0], alpha[1], alpha[2], alpha[3], yptr[0],
                     yptr[1], yptr[2], yptr[3], n);
    for (PetscInt i = 0; i < inc; ++i)
      PetscCall(VecRestoreArrayRead(y[i], yptr + i));
  }
  PetscCall(VecRestoreArray(xin, &xx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

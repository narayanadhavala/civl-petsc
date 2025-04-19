#include <petscvec.h>
#undef VecDot_MPI

static inline PetscErrorCode
VecXDot_MPI_Default(Vec xin, Vec yin, PetscScalar *z,
                    PetscErrorCode (*VecXDot_SeqFn)(Vec, Vec, PetscScalar *)) {
  PetscFunctionBegin;
  PetscCall(VecXDot_SeqFn(xin, yin, z));
  /* Change by Venkata: removed MPI_IN_PLACE reduction for MPI_SCALAR as CIVL
  doesn't support it */
#ifdef USE_COMPLEX
  {
    PetscReal local[2], tmp[2];
    local[0] = (*z).real;
    local[1] = (*z).imag;
    PetscCall(MPIU_Allreduce(local, tmp, 2, MPI_DOUBLE, MPIU_SUM,
                             PetscObjectComm((PetscObject)xin)));
    *z = scalar_make(tmp[0], tmp[1]);
  }
#else
  {
    PetscReal tmp[1];
    PetscCall(MPIU_Allreduce(z, tmp, 1, MPIU_SCALAR, MPIU_SUM,
                             PetscObjectComm((PetscObject)xin)));
    *z = tmp[0];
  }
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecDot_MPI(Vec xin, Vec yin, PetscScalar *z) {
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}
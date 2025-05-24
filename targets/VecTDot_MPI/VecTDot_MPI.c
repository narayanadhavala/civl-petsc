#include <petscvec.h>
#undef VecTDot_MPI

static inline PetscErrorCode
VecXDot_MPI_Default(Vec xin, Vec yin, PetscScalar *z,
                    PetscErrorCode (*VecXDot_SeqFn)(Vec, Vec, PetscScalar *)) {
#ifdef DEBUG
  $print("Target VecXDot_MPI_Default \n");
#endif
  PetscFunctionBegin;
  PetscCall(VecXDot_SeqFn(xin, yin, z));
  /* Change by Venkata: removed MPI_IN_PLACE reduction for MPI_SCALAR type
   * struct as CIVL doesn't support it */
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
#ifdef DEBUG
  $print("Target VecTDot_MPI \n");
#endif
  PetscFunctionBegin;
  PetscCall(VecXDot_MPI_Default(xin, yin, z, VecTDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

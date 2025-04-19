#include <petscvec.h>
#undef VecMTDot_MPI

static inline PetscErrorCode VecMXDot_MPI_Default(
    Vec xin, PetscInt nv, const Vec y[], PetscScalar *z,
    PetscErrorCode (*VecMXDot_SeqFn)(Vec, PetscInt, const Vec[],
                                     PetscScalar *)) {
  PetscFunctionBegin;
  PetscCall(VecMXDot_SeqFn(xin, nv, y, z));
  /* Change by Venkata: removed MPI_IN_PLACE reduction for MPI_SCALAR as CIVL
   *doesn't support it */
#ifdef USE_COMPLEX
  double in_real[nv], in_imag[nv], out_real[nv], out_imag[nv];
  // Separate real and imaginary parts
  for (int i = 0; i < nv; i++) {
    in_real[i] = z[i].real;
    in_imag[i] = z[i].imag;
  }
  // Perform Allreduce on real and imaginary parts separately
  MPI_Allreduce(in_real, out_real, nv, MPI_DOUBLE, MPI_SUM, xin->comm);
  MPI_Allreduce(in_imag, out_imag, nv, MPI_DOUBLE, MPI_SUM, xin->comm);
  for (int i = 0; i < nv; i++)
    z[i] = scalar_make(out_real[i], out_imag[i]);
#else
  MPI_Allreduce(z, z, nv, MPI_DOUBLE, MPI_SUM, xin->comm);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[],
                            PetscScalar *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecMTDot_MPI called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecMXDot_MPI_Default(xin, nv, y, z, VecMTDot_Seq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

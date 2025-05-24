#include "petscvec.h"
#include <mpi.h>

int main(void) {
  PetscReal norm;
  NormType type = NORM_2;
  PetscInt n = 4;
  STYPE values[n];
  MPI_Init(NULL, NULL);
  MPI_Comm comm = MPI_COMM_WORLD;
  for (int i = 0; i < n; i++) {
#ifdef USE_COMPLEX
    values[i] = $make_complex(1.0, 0.0);
#else
    values[i] = 1.0;
#endif
  }
  $vec c_x = $vec_make_from_dense(n, values);
  Vec p_x = CIVL_CivlToPetscVec(c_x, PETSC_DECIDE, comm);
  PetscErrorCode actual = VecNorm_Seq(p_x, type, &norm);
  $print("Norm ", type, " = ", norm, "\n");
  VecDestroy(&p_x);
  MPI_Finalize();
}

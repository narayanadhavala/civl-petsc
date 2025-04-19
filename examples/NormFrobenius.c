#include <assert.h>
#include <petscvec.h>

int main(int argc, char **argv) {
  Vec x, y;
  MPI_Comm comm;
  PetscScalar one = 1.0;
  PetscReal norm_frobenius_x, norm_frobenius_y;
  PetscInt n = 5, N;
  PetscMPIInt rank, size;

  PetscCall(PetscInitialize(
      &argc, &argv, NULL, "Compute Frobenius Norm for MPI and Seq Vectors\n"));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  N = size * n; // Total global size for MPI vector

  /* Create MPI parallel vector x */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x, one)); // Set all elements to 1.0
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /* Create Sequential vector y on rank 0 */
  if (rank == 0) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, N, &y));
    PetscCall(VecSet(y, one)); // Set all elements to 1.0
    PetscCall(VecAssemblyBegin(y));
    PetscCall(VecAssemblyEnd(y));
  } else {
    y = NULL; // Other ranks do not allocate y
  }

  /* Compute Frobenius Norm for MPI vector */
  PetscCall(VecNorm(x, NORM_FROBENIUS, &norm_frobenius_x));

  /* Compute Frobenius Norm for Sequential vector (only rank 0) */
  if (rank == 0)
    PetscCall(VecNorm(y, NORM_FROBENIUS, &norm_frobenius_y));

  /* Print results */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Rank %d: Frobenius Norm of MPI Vector x: %e\n", rank,
                        (double)norm_frobenius_x));
  if (rank == 0) {
    PetscCall(PetscPrintf(
        PETSC_COMM_SELF, "Rank %d: Frobenius Norm of Sequential Vector y: %e\n",
        rank, (double)norm_frobenius_y));
    assert(fabs(norm_frobenius_x - norm_frobenius_y) < 0.001);
  }

  /* Clean up */
  PetscCall(VecDestroy(&x));
  if (rank == 0)
    PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

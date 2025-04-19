static char help[] = "Minimal example for VecNorm, VecSetValues, VecScale, "
                     "VecDuplicate, VecCopy.\n";

#include <petsc/private/petscimpl.h> /* to gain access to the private PetscObjectStateIncrease() */
#include <petscvec.h>

int main(int argc, char **argv) {
  Vec V, W;
  MPI_Comm comm;
  PetscScalar one = 1.0, two = 2.0;
  PetscReal normV, normW;
  PetscInt index = 0;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = MPI_COMM_SELF;

  /* Create a vector V of local size 5 */
  PetscCall(VecCreate(comm, &V));
  PetscCall(VecSetSizes(V, 5, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(V));

  /* Set the value 'one' at index 0 of V */
  PetscCall(VecSetValues(V, 1, &index, &one, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(V));
  PetscCall(VecAssemblyEnd(V));

  /* Compute and print the 2-norm of V */
  PetscCall(VecNorm(V, NORM_2, &normV));
  PetscCall(
      PetscPrintf(comm, "V norm (after VecSetValues): %e\n", (double)normV));

  /* Scale V by 2.0 and re-compute norm */
  PetscCall(VecScale(V, two));
  PetscCall(VecNorm(V, NORM_2, &normV));
  PetscCall(
      PetscPrintf(comm, "V norm (after VecScale by 2): %e\n", (double)normV));

  /* Duplicate V into W and copy V to W */
  PetscCall(VecDuplicate(V, &W));
  PetscCall(VecCopy(V, W));
  PetscCall(VecNorm(W, NORM_2, &normW));
  PetscCall(PetscPrintf(comm, "W norm (after VecCopy): %e\n", (double)normW));

  PetscCall(VecDestroy(&V));
  PetscCall(VecDestroy(&W));
  PetscCall(PetscFinalize());
  return 0;
}

// VecCreation.c
#include <petscvec.h>

void VecCreationFunction() {
  PetscFunctionBeginUser;
  PetscInitialize(NULL, NULL, NULL, NULL);

  // Create a sequential vector
  Vec x;
  // PetscErrorCode ierr;
  PetscInt indices[] = {0, 1, 2, 3, 4};
  VecCreate(PETSC_COMM_WORLD, &x); // Creating a vector of size 5
  VecSetType(x, "mpi");

  // Set values in the vector
  PetscScalar values[] = {6.0, 7.0, 8.0, 9.0, 10.0};
  VecSetValues(x, 5, indices, values, INSERT_VALUES);

  // Assemble the vector
  VecAssemblyBegin(x);
  VecAssemblyEnd(x);

  // View the vector
  PetscPrintf(PETSC_COMM_WORLD, "Vector: \n");
  VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  // Destroy the vector and finalize PETSc
  VecDestroy(&x);
  PetscFinalize();
}

int main() {
  VecCreationFunction();
  return 0;
}

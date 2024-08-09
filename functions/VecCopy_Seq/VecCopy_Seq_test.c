#include <petscvec.h>
#include <stdio.h>
#include <assert.h>

int main(int argc, char **argv) {
    Vec x, y;
    PetscInt n = 4;
    PetscScalar two = 2.0, zero = 0.0;
    const PetscScalar *array;

    PetscCall(PetscInitialize(NULL, NULL, NULL, NULL));

    /* Create and set up the first vector */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSet(x, two));

    /* Create and set up the second vector */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
    PetscCall(VecSetSizes(y, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(y));
    PetscCall(VecSet(y, zero));

    /* Print initial values of y */
    PetscCall(VecGetArrayRead(y, &array));
    printf("Initial values of y:\n");
    for (PetscInt i = 0; i < n; i++) {
        printf("%g ", (double)array[i]);
    }
    printf("\n");
    PetscCall(VecRestoreArrayRead(y, &array));

    PetscErrorCode expected = PetscCall(VecCopy_Seq_spec(x, y));
    /* Copy x to y using VecCopy_Seq */
    PetscErrorCode actual = PetscCall(VecCopy_Seq(x, y));

    /* Print values of y after copy */
    PetscCall(VecGetArrayRead(y, &array));
    printf("Values of y after VecCopy_Seq:\n");
    for (PetscInt i = 0; i < n; i++) {
        printf("%g ", (double)array[i]);
    }
    printf("\n");
    PetscCall(VecRestoreArrayRead(y, &array));
    assert(expected == actual);
    /* Destroy the vectors */
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(PetscFinalize());
}

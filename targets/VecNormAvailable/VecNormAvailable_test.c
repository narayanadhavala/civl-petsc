#include "petscvec.h"
#include <assert.h>

int main(void) {
    Vec x;
    PetscReal norm;
    PetscBool available;
    PetscInt n = 4;
#ifdef USE_COMPLEX
    PetscScalar one = MY_COMPLEX(1.0, 1.0);
#else
    PetscScalar one = 1.0;
#endif
    PetscCall(PetscInitialize(NULL, NULL, NULL, NULL));

    /* Create a vector */
    PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
    PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(x));
    PetscCall(VecSet(x, one));

    /* Check if the norm is available before calculation */
    PetscCall(VecNormAvailable(x, NORM_2, &available, &norm));
    assert(available == PETSC_FALSE);  // Initially, norm should not be available

    /* Compute the norm using VecNorm */
    PetscCall(VecNorm(x, NORM_2, &norm));

    /* Check if the norm is available after calculation */
    PetscCall(VecNormAvailable(x, NORM_2, &available, &norm));
    assert(available == PETSC_TRUE);   // After VecNorm, norm should be available

    /* Print the computed norm */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Norm of the vector: %g\n", (double)norm));

    /* Destroy the Vector */
    PetscCall(VecDestroy(&x));
    PetscCall(PetscFinalize());
}


#include <petscvec.h>
#include <assert.h>

int main(void) {
    Vec x;
    PetscReal norm;
    PetscInt    n = 4;
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
    PetscErrorCode expected = PetscCall(VecNorm_Seq_spec(x, NORM_2, &norm)); 
    /* Compute the norm using VecNorm_Seq */
    PetscErrorCode actual = PetscCall(VecNorm_Seq(x, NORM_2, &norm)); 
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "L_2 Norm of the vector: %g\n", (double)norm));
    assert(expected == actual);
    /* Destroy the Vector */
    PetscCall(VecDestroy(&x));
    PetscCall(PetscFinalize());
}

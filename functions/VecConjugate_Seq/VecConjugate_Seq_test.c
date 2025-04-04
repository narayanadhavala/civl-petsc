#include <assert.h>
#include <petscvec.h>
#include <stdio.h>

// New function to print conjugate values
void printConjugate(const char *label, Vec v, PetscInt n) {
  const PetscScalar *array;
  PetscCall(VecGetArrayRead(v, &array));
  printf("\nConjugate value of %s:\n", label);
  for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
    printf("real part:%g imag part: %g\n", array[i].real, array[i].imag);
#else
    printf("%g ", array[i]);
#endif
  }
  PetscCall(VecRestoreArrayRead(v, &array));
}

int main(int argc, char **argv) {
  Vec x, y;
  const PetscScalar *x_array;
#ifdef USE_COMPLEX
  PetscScalar input_values[] = {MY_COMPLEX(1.0, 2.0), MY_COMPLEX(3.0, 4.0),
                                MY_COMPLEX(5.0, -6.0), MY_COMPLEX(7.0, -8.0)};
  PetscScalar zero = MY_COMPLEX(0.0, 0.0);
#else
  PetscScalar input_values[] = {1.0, 3.0, 5.0, 7.0};
  PetscScalar zero = 0.0;
#endif
  PetscInt n = 4;
  PetscInitialize(&argc, &argv, NULL, NULL);

  // Create and set up the vector
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x));

  /* Create and set up the second vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecSet(y, zero));

  for (PetscInt i = 0; i < n; i++)
    PetscCall(VecSetValue(x, i, input_values[i], INSERT_VALUES));

  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  // Print values of x before conjugation
  printf("\nBefore VecConjugate_Seq:\n");
  PetscCall(VecGetArrayRead(x, &x_array));
  CIVL_PrintVec("X", x);
  PetscCall(VecRestoreArrayRead(x, &x_array));

  PetscCall(VecCopy_Seq(x, y));

  // Calculate the conjugate using VecConjugate_Seq
  PetscErrorCode actual = PetscCall(VecConjugate_Seq(x));

  // Print values of x after conjugation
  printf("\nAfter calling VecConjugate_Seq:\n");
  PetscCall(VecGetArrayRead(x, &x_array));
  CIVL_PrintVec("X", x);

  // Calculate the conjugate using VecConjugate_Spec
  PetscErrorCode expected = PetscCall(VecConjugate_Seq_spec(y));

  // Print values of y after conjugation
  printf("\nAfter calling VecConjugate_Seq_Spec:\n");
  PetscCall(VecGetArrayRead(y, &x_array));
  CIVL_PrintVec("X", x);

  // Verify that the conjugation is correct
  for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
    assert(input_values[i].real == x_array[i].real);
    assert(input_values[i].imag == -x_array[i].imag);
#else
    assert(input_values[i] == x_array[i]);
#endif
  }
  assert(expected == actual);
  PetscCall(VecRestoreArrayRead(x, &x_array));

  // Destroy the vectors
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
}

#include <petscvec.h>
#include <math.h>
#include <assert.h>
#include <stdlib.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

#define max(a, b) (a > b ? a : b)

#define PETSC_FLOPS_PER_OP 1.0

PetscLogDouble petsc_TotalFlops = 0.0;

PetscLogDouble petsc_TotalFlops_th = 0.0;

PETSC_EXTERN PetscLogDouble petsc_TotalFlops;

PETSC_EXTERN_TLS PetscLogDouble petsc_TotalFlops_th;

#define PetscAddLogDouble(a, b, c) ((PetscErrorCode)((*(a) += (c)), (*(b) += (c)), PETSC_SUCCESS))

PetscReal my_cabs(MyComplex z) {
    return sqrt(z.real*z.real + z.imag*z.imag);
}

MyComplex my_conj(MyComplex z) {
    return (MyComplex){z.real, -z.imag};
}

MyComplex my_cadd(MyComplex x, MyComplex y) {
    return (MyComplex){x.real+y.real, x.imag+y.imag};
}

MyComplex my_csub(MyComplex x, MyComplex y) {
    MyComplex result;
    result.real = x.real - y.real;
    result.imag = x.imag - y.imag;
    return result;
}

MyComplex my_cmul(MyComplex x, MyComplex y) {
    return (MyComplex){x.real*y.real - x.imag*y.imag, x.real*y.imag + x.imag*y.real};
}

PetscErrorCode PetscInitialize(int *argc, char ***args, const char file[], const char help[])
{
  return 0;
}

PetscErrorCode PetscOptionsGetInt(PetscOptions options, const char pre[], const char name[], PetscInt *ivalue, PetscBool *set)
{
  return 0;
}

Vec vec_create_seq(int n, PetscScalar *data) {
    Vec vec = (Vec)malloc(sizeof(struct Vec_s));
    vec->map = (SimpleMap)malloc(sizeof(struct map_s));
    vec->map->n = n;
    vec->map->N = n;
    vec->block_size = 0;
    vec->data = (PetscScalar*)malloc(n * sizeof(PetscScalar));
    if (data != NULL) {
        for (int i = 0; i < n; i++)
#ifdef USE_COMPLEX
            vec->data[i] = MY_COMPLEX(data[i].real, data[i].imag);
#else
            vec->data[i] = data[i];
#endif
    } else {
        for (int i = 0; i < n; i++)
#ifdef USE_COMPLEX
            vec->data[i] = MY_COMPLEX(0.0, 0.0);
#else
            vec->data[i] = 0.0;
#endif
    }
    return vec;
}

void vec_destroy_seq(Vec vec) {
    free(vec->data);
    free(vec->map);
    free(vec);
}

PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec) {
    // Allocate memory for the Vec structure
    *vec = (Vec)malloc(sizeof(struct Vec_s));

    // Allocate memory for the SimpleMap structure
    (*vec)->map = (SimpleMap)malloc(sizeof(struct map_s));

    // Initialize the map fields
    (*vec)->map->n = 0; // Default initial local size
    (*vec)->map->N = 0; // Default initial global size

    // Optionally, you can initialize other fields of Vec if needed
    (*vec)->block_size = 0;
    (*vec)->data = NULL;

    return 0;
}

PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a)
{
  if (x == NULL || a == NULL)
  {
    return 1;
  }
  *a = x->data;
  assert(*a != NULL);
  return 0;
}

PetscErrorCode VecGetArray(Vec x, PetscScalar **a)
{
    if (x == NULL || a == NULL) {
        return 1; // Return an error code if the input arguments are NULL
    }
    *a = x->data;
    if (*a == NULL) {
        return 1; // Return an error code if the vector data is NULL
    }
    assert(*a != NULL);
    return 0; // Success
}

bool vec_eq_seq(Vec vec1, Vec vec2) {
    PetscInt n = vec1->map->n;
    if (n != vec2->map->n) return false;
    if (vec1->map->N != vec2->map->N) return false;
    if (vec1->block_size != vec2->block_size) return false;
    PetscScalar * a1 = vec1->data, * a2 = vec2->data;
    for (int i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        if (a1[i].real != a2[i].real || a1[i].imag != a2[i].imag) 
            return false;
#else
        if (a1[i] != a2[i])
            return false;
#endif
    }
    return true;
}

PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg) {
    *flg = vec_eq_seq(vec1, vec2) ? PETSC_TRUE : PETSC_FALSE;
    return PETSC_SUCCESS;
}

void vecprint_seq(const char* name, Vec vin) {
    printf("%s", name);
    PetscInt n = vin->map->n;
    for (PetscInt i = 0; i < n; i++) {
        #ifdef USE_COMPLEX
            printf("(%g + %gi) ", vin->data[i].real, vin->data[i].imag);
        #else
        printf("%g ", vin->data[i]);
        #endif
    }
    printf("\n");
}

PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a)
{
  if (x == NULL || a == NULL)
  {
    return 1;
  }

  // Reset the array pointer to NULL
  *a = NULL;
  assert(*a == NULL);

  return 0;
}

PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a)
{
    if (x == NULL || a == NULL || *a == NULL) {
        return 1; // Return an error only if input arguments are invalid
    }

    // Reset the array pointer to NULL
    *a = NULL;
    assert(*a == NULL);

    return 0; // Success
}

PetscReal PetscAbsReal(PetscReal v1)
{
  return fabs(v1);
}

PetscErrorCode PetscBLASIntCast(PetscInt a, PetscBLASInt *b)
{
  // Check if the integer is negative

  if (a < 0)
  {
    return 1; // input argument, out of range
  }

  // Check if PetscBLASInt can hold the value of a
  if ((PetscBLASInt)a != a)
  {
    return 1; // input argument, out of range
  }

  // Assign the value of a to b
  *b = (PetscBLASInt)a;
  return 0;
}

PetscErrorCode PetscLogFlops(PetscLogDouble n)
{
  // Check if n is non-negative
  if (n < 0)
  {
    return 1;
  }

  // Update the total flops count
  return PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th, PETSC_FLOPS_PER_OP * n);
}

PetscErrorCode VecSetSizes(Vec v, PetscInt n, PetscInt N) {
  if (!v->map) {
    // Allocate memory for the map if it hasn't been allocated yet.
    v->map = (SimpleMap)malloc(sizeof(struct map_s));
    if (!v->map) return -1; // Return an error code if memory allocation fails
  }

  if (n == PETSC_DECIDE) {
    if (N == PETSC_DETERMINE) {
      assert(0);
      return 1;
    }
    v->map->n = v->map->N = N;
  } else {
    v->map->n = n;
    if (N == PETSC_DETERMINE) {
      v->map->N = n;
    } else {
      v->map->N = N;
    }
  }
  return 0;
}

PetscErrorCode VecSetBlockSize(Vec v, PetscInt bs)
{
  v->block_size = bs;
  return 0;
}

PetscErrorCode VecSetFromOptions(Vec vec) {
    if (vec == NULL) {
        // Handle invalid vector pointer
        return 1;
    }

    if (vec->map) {
        // Allocate memory for the vector data based on the local size
        vec->data = malloc(vec->map->n * sizeof(PetscScalar));
        if (vec->data == NULL) {
            // Handle memory allocation failure
            return 1; // Return an error code or take appropriate action
        }
    } else {
        // If the map is not available, allocate a new map structure
        vec->map = malloc(sizeof(struct map_s));
        if (vec->map == NULL) {
            // Handle memory allocation failure
            return 1; // Return an error code or take appropriate action
        }
        // Initialize the map with default values or desired values
        vec->map->n = 0;
        vec->map->N = 0;
        // Set vec->data to NULL since the local size is 0
        vec->data = NULL;
    }

    return 0;
}

PetscErrorCode VecSet(Vec x, PetscScalar alpha) {
    PetscInt n;

    // Check if the vector is NULL
    if (!x) {
        return 1;
    }

    // Check if the map is available and retrieve the local size
    if (x->map) {
        n = x->map->n;
    } else {
        // If the map is not available, return an error
        return 1;
    }

    // Check if the data array is available
    if (x->data) {
        // Set each element of the vector data to the given scalar value
        for (PetscInt i = 0; i < n; i++) {
            x->data[i] = alpha;
        }
        return PETSC_SUCCESS; // Success
    } else {
        // If the data array is not available, return an error
        return 1;
    }
}

PetscErrorCode VecView(Vec vec, PetscViewer viewer) {
    PetscInt i, n;
    const PetscScalar *array;

    PetscFunctionBegin;
    // Get the local size of the vector
    PetscCall(VecGetLocalSize(vec, &n));

    // Access the array inside the vector
    PetscCall(VecGetArrayRead(vec, &array));

    // Determine the viewer context (for simplicity, we'll handle stdout here)
    if (viewer->format == PETSC_VIEWER_STDOUT_SELF || viewer->format == PETSC_VIEWER_DEFAULT) {
        // Print the vector contents
        for (i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            MyComplex value = array[i];
            printf("Element %d: %g + %gi\n", i, value.real, value.imag);
#else
            printf("Element %d: %g\n", i, array[i]);
#endif
        }
    } else {
        // Handle other viewers if needed (not implemented here)
        return 1;
    }

    // Restore the array
    PetscCall(VecRestoreArrayRead(vec, &array));
    PetscFunctionReturn(0);
}

PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val) {
    PetscInt n;

    // Check if the maps are available and retrieve the local & global sizes
    if (x->map && y->map) {
        // Check if the local & global sizes are the same
        if (x->map->n != y->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = x->map->n;
    } else {
        // If either map is not available, return an error
        return 1;
    }

// Initialize dot product accumulator
#ifdef USE_COMPLEX
    PetscScalar dot_product = MY_COMPLEX(0.0, 0.0);
#else
    PetscScalar dot_product = 0.0;
#endif

    // Compute dot product
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        // Complex dot product
        dot_product = my_cadd(dot_product, my_cmul(x->data[i], my_conj(y->data[i])));
#else
        // Real dot product
        dot_product += x->data[i] * y->data[i];
#endif
    }

    // Store the dot product in val
    *val = dot_product;

    return 0; // Success
}

PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
    PetscInt n;

    // Check if the map of x is available and retrieve the local & global sizes
    if (x->map) {
        n = x->map->n;
    } else {
        // If map is not available, return an error
        return 1;
    }

    // Check if each y vector has the same size as x
    for (PetscInt i = 0; i < nv; i++) {
        if (y[i]->map && y[i]->map->n != n) {
            return 1; // Error code for incompatible vector sizes
        }
    }

    for (PetscInt i = 0; i < nv; i++) {
#ifdef USE_COMPLEX
        PetscScalar dot_product = MY_COMPLEX(0.0, 0.0);
#else
        PetscScalar dot_product = 0.0;
#endif

        // Compute dot product
        for (PetscInt j = 0; j < n; j++) {
#ifdef USE_COMPLEX
            // Complex dot product
            dot_product = my_cadd(dot_product, my_cmul(x->data[j], my_conj(y[i]->data[j])));
#else
            // Real dot product
            dot_product += x->data[j] * y[i]->data[j];
#endif
        }

        // Store the dot product in val[i]
        val[i] = dot_product;
    }

    return 0; // Success
}

PetscErrorCode VecCopy(Vec xin, Vec yin) {
    PetscInt n;

    // Check if the maps are available for both vectors and retrieve the local sizes
    if (xin->map && yin->map) {
        // Check if the local sizes are the same
        if (xin->map->n != yin->map->n) {
            return 1; // Error code for vector size mismatch
        }
        n = xin->map->n;
    } else {
        // If either map is not available, return an error
        return 1;
    }

    // Copying data from xin to yin
    for (PetscInt i = 0; i < n; i++) {
        yin->data[i] = xin->data[i];
    }
    assert(yin);
    return 0; // Success
}

PetscErrorCode VecGetSize(Vec x, PetscInt *size) {
    if (x->map) {
        *size = x->map->N;
    } else {
        return 1; // Error code if the map is not available
    }
    return 0;
}

PetscErrorCode VecGetLocalSize(Vec x, PetscInt *size) {
    *size = x->map->n;  // Get the local size of the vector
    return 0;
}

PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
    PetscInt n;

    // Check if the map is available and retrieve the local size
    if (x->map) {
        n = x->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    *val = PETSC_MIN_REAL;
    *p = -1;
    for (PetscInt i = 0; i < n; ++i) {
#ifdef USE_COMPLEX
        if (x->data[i].real > *val || (x->data[i].real == *val && x->data[i].imag > 0)) {
            *val = x->data[i].real;
            *p = i;
        }
#else
        if (x->data[i] > *val) {
            *val = x->data[i];
            *p = i;
        }
#endif
    }
    return 0;
}

PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
    PetscInt n;

    // Check if the map is available and retrieve the local size
    if (x->map) {
        n = x->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    *val = PETSC_MAX_REAL;
    *p = -1;
    for (PetscInt i = 0; i < n; ++i) {
#ifdef USE_COMPLEX
        if (x->data[i].real < *val || (x->data[i].real == *val && x->data[i].imag < 0)) {
            *val = x->data[i].real;
            *p = i;
        }
#else
        if (x->data[i] < *val) {
            *val = x->data[i];
            *p = i;
        }
#endif
    }
    return 0;
}

PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
    PetscInt n;

    // Check if the map is available and retrieve the local size
    if (x->map) {
        n = x->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    // Scale each component of the vector by alpha
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        x->data[i] = my_cmul(x->data[i], alpha);
#else
        x->data[i] *= alpha;
#endif
    }
    assert(x);
    return 0; // Success
}

PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[]) {
    PetscInt n;

    // Ensure that y is not in the x array
    for (PetscInt i = 0; i < nv; i++) {
        if (y == x[i]) {
            return 1; // Error code for incompatible arguments
        }
    }

    // Check if the map is available for vector y and retrieve the local size
    if (y->map) {
        n = y->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    // Check if the vectors have compatible sizes
    for (PetscInt i = 0; i < nv; i++) {
        if (x[i]->map == NULL || x[i]->map->n != n) {
            return 1; // Error code for incompatible vector sizes
        }
    }

    // Compute y = y + sum alpha[i] x[i] for each component
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        MyComplex sum = MY_COMPLEX(0.0, 0.0);
        for (PetscInt j = 0; j < nv; j++) {
            sum = my_cadd(sum, my_cmul(alpha[j], x[j]->data[i]));
        }
        y->data[i] = my_cadd(y->data[i], sum);
#else
        PetscScalar sum = 0.0;
        for (PetscInt j = 0; j < nv; j++) {
            sum += alpha[j] * x[j]->data[i];
        }
        y->data[i] += sum;
#endif
    }
    assert(y);
    return 0; // Success
}

PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
    PetscInt n;

    // Check if the maps are available for both vectors and retrieve the local sizes
    if (y->map && x->map) {
        // Check if the local sizes are the same
        if (y->map->n != x->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = y->map->n;
    } else {
        // If either map is not available, return an error
        return 1;
    }

    // Perform the operation
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        // Use PetscRealPart and PetscImaginaryPart macros to access real and imaginary parts
        PetscReal real_part_alpha = PetscRealPart(alpha);
        PetscReal imag_part_alpha = PetscImaginaryPart(alpha);
        y->data[i].real += real_part_alpha * x->data[i].real - imag_part_alpha * x->data[i].imag;
        y->data[i].imag += imag_part_alpha * x->data[i].real + real_part_alpha * x->data[i].imag;
#else
        y->data[i] += alpha * x->data[i];
#endif
    }

    return 0; // Success
}

PetscErrorCode VecSwap(Vec x, Vec y) {
    PetscInt n;

    // Check if the maps are available for both vectors and retrieve the local sizes
    if (x->map && y->map) {
        // Check if the local sizes are the same
        if (x->map->n != y->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = x->map->n;
    } else {
        // If either map is not available, return an error
        return 1;
    }

    // Swap the values between the two vectors
    for (PetscInt i = 0; i < n; i++) {
        PetscScalar temp = x->data[i];
        x->data[i] = y->data[i];
        y->data[i] = temp;
    }
    assert(x);
    assert(y);
    return 0; // Success
}

PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
    PetscInt n;

    // Check if the maps are available for all vectors and retrieve the local sizes
    if (w->map && x->map && y->map) {
        // Check if the local sizes are the same
        if (w->map->n != x->map->n || w->map->n != y->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = w->map->n;
    } else {
        // If any map is not available, return an error
        return 1;
    }

    // Compute w = alpha * x + y
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        w->data[i] = my_cadd(my_cmul(alpha, x->data[i]), y->data[i]);
#else
        w->data[i] = alpha * x->data[i] + y->data[i];
#endif
    }
    assert(w);
    return 0; // Success
}

PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
    PetscInt n;

    // Check if the maps are available for both vectors and retrieve the local sizes
    if (y->map && x->map) {
        // Check if the local sizes are the same
        if (y->map->n != x->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = y->map->n;
    } else {
        // If either map is not available, return an error
        return 1;
    }

    // Optimize for common values of beta
    if (PetscRealPart(beta) == 0.0 && PetscImaginaryPart(beta) == 0.0) {
        // If beta is 0, y remains unchanged
        return 0; // Success
    } else if (PetscRealPart(beta) == 1.0 && PetscImaginaryPart(beta) == 0.0) {
        // If beta is 1, y becomes the sum of x and y
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            y->data[i] = my_cadd(y->data[i], x->data[i]);
#else
            y->data[i] += x->data[i];
#endif
        }
    } else if (PetscRealPart(beta) == -1.0 && PetscImaginaryPart(beta) == 0.0) {
        // If beta is -1, y becomes the difference of y and x
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            y->data[i] = my_csub(y->data[i], x->data[i]);
#else
            y->data[i] -= x->data[i];
#endif
        }
    } else {
        // For other values of beta, perform the standard operation, y = (beta * y) + x
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            y->data[i] = my_cadd(my_cmul(beta, y->data[i]), x->data[i]);
#else
            y->data[i] = beta * y->data[i] + x->data[i];
#endif
        }
    }
    assert(y);
    return 0; // Success
}

PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y) {
    PetscInt n;

    // Check if the maps are available for all vectors and retrieve the local sizes
    if (w->map && x->map && y->map) {
        // Check if the local sizes are the same
        if (w->map->n != x->map->n || w->map->n != y->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = w->map->n;
    } else {
        // If any map is not available, return an error
        return 1;
    }

    // Compute w[i] = x[i] * y[i] component-wise
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        // Complex multiplication
        w->data[i].real = x->data[i].real * y->data[i].real - x->data[i].imag * y->data[i].imag;
        w->data[i].imag = x->data[i].real * y->data[i].imag + x->data[i].imag * y->data[i].real;
#else
        // Real multiplication
        w->data[i] = x->data[i] * y->data[i];
#endif
    }

    return 0; // Success
}

PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y) {
    PetscInt n;

    // Check if the maps are available for all vectors and retrieve the local sizes
    if (w->map && x->map && y->map) {
        // Check if the local sizes are the same
        if (w->map->n != x->map->n || w->map->n != y->map->n) {
            return 1; // Error code for incompatible vector sizes
        }
        n = w->map->n;
    } else {
        // If any map is not available, return an error
        return 1;
    }

    // Compute w[i] = x[i] / y[i] component-wise
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
        // Complex division
        PetscReal denom = y->data[i].real * y->data[i].real + y->data[i].imag * y->data[i].imag;
        if (denom == 0.0) {
            return 1; // Error code for division by zero
        }
        w->data[i].real = (x->data[i].real * y->data[i].real + x->data[i].imag * y->data[i].imag) / denom;
        w->data[i].imag = (x->data[i].imag * y->data[i].real - x->data[i].real * y->data[i].imag) / denom;
#else
        // Real division
        if (y->data[i] == 0.0) {
            return 1; // Error code for division by zero
        }
        w->data[i] = x->data[i] / y->data[i];
#endif
    }

    return 0; // Success
}

PetscErrorCode VecAssemblyBegin(Vec vec)
{
  return 0;
}

PetscErrorCode VecAssemblyEnd(Vec vec)
{
  return 0;
}

PetscErrorCode VecDuplicate(Vec v, Vec *newv) {
    // Allocate memory for the new vector
    *newv = malloc(sizeof(struct Vec_s));
    if (*newv == NULL) {
        // Handle memory allocation failure
        return 1; // Error code for memory allocation failure
    }

    // Copy the size information from the original vector
    if (v->map) {
        (*newv)->map = malloc(sizeof(struct map_s));
        if ((*newv)->map == NULL) {
            free(*newv);
            // Handle memory allocation failure
            return 1; // Error code for memory allocation failure
        }
        (*newv)->map->n = v->map->n;
        (*newv)->map->N = v->map->N;
    } else {
        (*newv)->map = NULL;
    }

    (*newv)->block_size = v->block_size;

    // Allocate memory for the data array
    PetscInt local_size = (v->map) ? v->map->n : 0;
    (*newv)->data = malloc(local_size * sizeof(PetscScalar));
    if ((*newv)->data == NULL) {
        if ((*newv)->map) {
            free((*newv)->map);
        }
        free(*newv);
        // Handle memory allocation failure
        return 1; // Error code for memory allocation failure
    }

    return 0; // Success
}

PetscErrorCode VecDuplicateVecs(Vec v, PetscInt m, Vec *V[])
{
  PetscErrorCode ierr;

  // Allocate memory for the array of vectors
  *V = (Vec *)malloc(m * sizeof(Vec));
  if (!(*V))
    return 1; // Error code for memory allocation failure

  // Create m vectors of the same type as v
  for (PetscInt i = 0; i < m; i++)
  {
    ierr = VecDuplicate(v, &(*V)[i]);
    if (ierr != 0)
    {
      // Handle error, free allocated memory, and return error code
      for (PetscInt j = 0; j < i; j++)
      {
        VecDestroy(&(*V)[j]);
      }
      free(*V);
      return ierr;
    }
  }
  assert(*V);
  return 0; // Success
}

PetscErrorCode VecDestroyVecs(PetscInt m, Vec *vv[])
{

  if (!vv || !(*vv))
    return 1; // If vv or *vv is NULL, return success

  // Free the memory for each vector in the array
  for (PetscInt i = 0; i < m; i++)
  {
    if ((*vv)[i])
    {
      VecDestroy(&(*vv)[i]);
    }
  }

  // Free the memory for the array itself
  free(*vv);
  *vv = NULL; // Set the pointer to NULL to avoid dangling references

  return 0; // Success
}

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
    PetscInt n;

    // Check if the map is available and retrieve the local size
    if (x->map) {
        n = x->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    switch (type) {
    case NORM_1: {
        *val = 0.0;
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            *val += my_cabs(x->data[i]);
#else
            *val += fabs(x->data[i]);
#endif
        }
        return 0;
    }
    case NORM_FROBENIUS:
    case NORM_2: {
        PetscReal s = 0.0;
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            PetscScalar d = x->data[i];
            s += d.real * d.real + d.imag * d.imag;
#else
            PetscScalar d = x->data[i];
            s += d * d;
#endif
        }
        *val = sqrt(s);
        return 0;
    }
    case NORM_INFINITY: {
        *val = 0.0;
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            *val = fmax(*val, my_cabs(x->data[i]));
#else
            *val = fmax(*val, fabs(x->data[i]));
#endif
        }
        return 0;
    }
    case NORM_1_AND_2: {
        PetscReal norm1 = 0.0, norm2 = 0.0;
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            norm1 += my_cabs(x->data[i]);
            PetscScalar d = x->data[i];
            norm2 += d.real * d.real + d.imag * d.imag;
#else
            norm1 += fabs(x->data[i]);
            norm2 += x->data[i] * x->data[i];
#endif
        }
        norm2 = sqrt(norm2);
        val[0] = norm1;
        val[1] = norm2;
        return 0;
    }
    default:
        assert(0); // not dealing with this for now
        return 1;
    }
}

PetscErrorCode VecStrideNorm(Vec x, PetscInt start, NormType ntype, PetscReal *val) {
    PetscInt n, stride;

    // Check if the map and block_size are available and retrieve the local size and stride
    if (x->map && x->block_size > 0) {
        n = x->map->n;
        stride = x->block_size;
    } else {
        return 1; // Error code if the map or block_size is not available
    }

    switch (ntype) {
    case NORM_1: {
        *val = 0.0;
        for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
            *val += my_cabs(x->data[i]);
#else
            *val += fabs(x->data[i]);
#endif
        }
        return 0;
    }
    case NORM_FROBENIUS:
    case NORM_2: {
        double s = 0.0;
        for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
            PetscScalar d = x->data[i];
            s += d.real * d.real + d.imag * d.imag;
#else
            PetscScalar d = x->data[i];
            s += d * d;
#endif
        }
        *val = sqrt(s);
        return 0;
    }
    case NORM_INFINITY: {
        *val = 0.0;
        for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
            *val = fmax(*val, my_cabs(x->data[i]));
#else
            *val = fmax(*val, fabs(x->data[i]));
#endif
        }
        return 0;
    }
    case NORM_1_AND_2: {
        PetscReal norm1 = 0.0, norm2 = 0.0;
        for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
            norm1 += my_cabs(x->data[i]);
            PetscScalar d = x->data[i];
            norm2 += d.real * d.real + d.imag * d.imag;
#else
            norm1 += fabs(x->data[i]);
            norm2 += x->data[i] * x->data[i];
#endif
        }
        norm2 = sqrt(norm2);
        val[0] = norm1;
        val[1] = norm2;
        return 0;
    }
    default:
        assert(0); // not dealing with this for now
        return 1;
    }
}

PetscErrorCode VecSetValue(Vec v, PetscInt row, PetscScalar value, InsertMode mode) {
    PetscInt n;
    PetscScalar *v_array;

    // Check if the map is available and retrieve the local size
    if (v->map) {
        n = v->map->n;
    } else {
        return 1; // Error code if the map is not available
    }

    // Check if the row index is within the local size of the vector
    if (row < 0 || row >= n) {
        return 1; // Error code for index out of range
    }
   
    VecGetArray(v, &v_array); // Get array from vector

    // Set the value based on the mode and the type of PetscScalar
#ifdef USE_COMPLEX
    if (mode == INSERT_VALUES) {
        v_array[row].real = value.real;  
        v_array[row].imag = value.imag;
    } else if (mode == ADD_VALUES) {
        v_array[row].real += value.real;
        v_array[row].imag += value.imag;
    } else {
        return 1; // Error code for invalid mode
    }
#else
    if (mode == INSERT_VALUES) {
        v_array[row] = value; // Set the value directly
    } else if (mode == ADD_VALUES) {
        v_array[row] += value; // Add the value to the existing value
    } else {
        return 1; // Error code for invalid mode
    }
#endif
  
    VecRestoreArray(v, &v_array);

    return 0; // Success
}

PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n) {
    if (a == NULL || b == NULL) {
        return 1; // Return an error code if the input pointers are NULL
    } else {
        memcpy(a, b, n);
    }
    return 0; // Success
}

PetscErrorCode VecDestroy(Vec *v)
{
  if (!*v)
    return 0;

  // Free data array if allocated
  if ((*v)->data)
  {
    free((*v)->data);
  }

  // Free map if allocated
  if ((*v)->map)
  {
    free((*v)->map);
  }

  // Free the vector structure itself if it's not NULL
  if (*v)
  {
    free(*v);
    *v = NULL;
  }

  return 0;
}

PetscErrorCode PetscFinalize(void)
{
  return 0;
}

PetscReal BLASnrm2_(const PetscBLASInt *n, const PetscScalar *x, const PetscBLASInt *stride) {
    PetscBLASInt in = *n;
    PetscBLASInt istride = *stride;
    double s = 0.0;

    for (PetscBLASInt i = 0; i < in; i += istride) {
#ifdef USE_COMPLEX
        s += my_cabs(x[i]) * my_cabs(x[i]);
#else
        s += x[i] * x[i];
#endif
    }

    return sqrt(s);
}

PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x, const PetscBLASInt *sx,
                     const PetscScalar *y, const PetscBLASInt *sy) {
    PetscBLASInt i, ix = 0, iy = 0;
#ifdef USE_COMPLEX
    PetscScalar sum = MY_COMPLEX(0.0, 0.0);
#else
    PetscScalar sum = 0.0;
#endif
    if (*n == 0) return sum;
    assert(!(*n < 0 || *sx <= 0 || *sy <= 0));
    for (i = 0; i < *n; i++) {
#ifdef USE_COMPLEX
        sum = my_cadd(sum, my_cmul(x[ix], my_conj(y[iy])));
#else
        sum += x[ix] * y[iy];
#endif
        ix += *sx;
        iy += *sy;
    }
    return sum;
}

PetscReal BLASasum_(const PetscBLASInt *n, const PetscScalar *dx, const PetscBLASInt *incx) {
    const PetscBLASInt n_int = *n, incx_int = *incx;
    assert(incx_int >= 1);
    assert(n_int >= 0);
    assert(n_int == 0 || dx != NULL);
    PetscReal sum = 0.0;
    for (PetscBLASInt i = 0, ix = 0; i < n_int; i++) {
        sum += PetscAbsScalar(dx[ix]);
        ix += incx_int;
    }
    return sum;
}

/* spec stub's implementation*/

PetscErrorCode VecNorm_Seq(Vec x, NormType type, PetscReal *val) {
    PetscInt n;
    const PetscScalar *x_array;
    if (x == NULL || val == NULL)
        return 1; // Error: invalid input
    VecGetLocalSize(x, &n);
    VecGetArrayRead(x, &x_array);
    if (x_array == NULL) {
        return 1; // Error: failed to get array
    }
    *val = 0.0;
    switch (type) {
    case NORM_1:
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            val[0] += my_cabs(x_array[i]);
#else
            val[0] += fabs(x_array[i]);
#endif
        }
        break;
    case NORM_2:
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            val[0] += x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
#else
            val[0] += x_array[i] * x_array[i];
#endif
        }
        val[0] = sqrt(val[0]);
        break;
    case NORM_FROBENIUS:
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            val[0] += x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
#else
            val[0] += x_array[i] * x_array[i];
#endif
        }
        val[0] = sqrt(val[0]);
        break;
    case NORM_INFINITY:
        val[0] = 0.0;
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            double magnitude = my_cabs(x_array[i]);
#else
            double magnitude = fabs(x_array[i]);
#endif
            if (magnitude > val[0]) {
                val[0] = magnitude;
            }
        }
        break;
    case NORM_1_AND_2:
        for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
            val[0] += my_cabs(x_array[i]);
            val[1] += x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
#else
            val[0] += fabs(x_array[i]);
            val[1] += x_array[i] * x_array[i];
#endif
        }
        val[1] = sqrt(val[1]);
        break;
    default:
        assert(0); // not dealing with this for now
        VecRestoreArrayRead(x, &x_array);
        return 1;
    }
    VecRestoreArrayRead(x, &x_array);
    return 0;
}

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin)
{
    PetscInt i;
    PetscScalar *x, *y;
    PetscInt n;
    // Check if the vector sizes are compatible
    if (xin->map->n != yin->map->n) {
        return 1;
    }
    n = xin->map->n;
    x = xin->data;
    y = yin->data;
    // Copy the data
    for (i = 0; i < n; i++) {
        y[i] = x[i];
    }
    assert(y != NULL);
    return 0;
}

PetscErrorCode VecConjugate_Seq(Vec xin) {
    PetscInt n = xin->map->n;
    PetscScalar *x = xin->data;
    for (PetscInt i = 0; i < n; ++i)
        x[i] = PetscConj(x[i]);
    return 0;
}

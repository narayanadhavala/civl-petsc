#include <petscvec.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <petscvec.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define max(a, b) (a > b ? a : b)

#define PETSC_FLOPS_PER_OP 1.0

PetscLogDouble petsc_TotalFlops = 0.0;

PetscLogDouble petsc_TotalFlops_th = 0.0;

PETSC_EXTERN PetscLogDouble petsc_TotalFlops;

PETSC_EXTERN_TLS PetscLogDouble petsc_TotalFlops_th;

#define PetscAddLogDouble(a, b, c)                                             \
  ((PetscErrorCode)((*(a) += (c)), (*(b) += (c)), PETSC_SUCCESS))

PetscReal my_cabs(MyComplex z) {
  return sqrt(z.real * z.real + z.imag * z.imag);
}

MyComplex my_conj(MyComplex z) { return (MyComplex){z.real, -z.imag}; }

MyComplex my_cadd(MyComplex x, MyComplex y) {
  return (MyComplex){x.real + y.real, x.imag + y.imag};
}

MyComplex my_csub(MyComplex x, MyComplex y) {
  MyComplex result;
  result.real = x.real - y.real;
  result.imag = x.imag - y.imag;
  return result;
}

MyComplex my_cmul(MyComplex x, MyComplex y) {
  return (MyComplex){x.real * y.real - x.imag * y.imag,
                     x.real * y.imag + x.imag * y.real};
}

PetscErrorCode PetscInitialize(int *argc, char ***args, const char file[],
                               const char help[]) {
  int ierr;
  ierr = MPI_Init(argc, args);
  if (ierr != MPI_SUCCESS) {
    return 1; // or an appropriate error code
  }
  return 0;
}

PetscErrorCode PetscOptionsGetInt(PetscOptions options, const char pre[],
                                  const char name[], PetscInt *ivalue,
                                  PetscBool *set) {
  return 0;
}

PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec) {
  *vec = (Vec)malloc(sizeof(struct Vec_s));
  assert(*vec);

  (*vec)->map = (SimpleMap)malloc(sizeof(struct map_s));
  assert((*vec)->map);

  (*vec)->map->n = 0;
  (*vec)->map->N = 0;
  (*vec)->map->rstart = 0;
  (*vec)->map->rend = 0;
  (*vec)->map->bs = 1;

  // Initialize ISLocalToGlobalMapping
  (*vec)->map->mapping =
      (ISLocalToGlobalMapping)malloc(sizeof(struct _p_ISLocalToGlobalMapping));
  assert((*vec)->map->mapping);
  (*vec)->map->mapping->n = 0;
  (*vec)->map->mapping->bs = 1;
  (*vec)->map->mapping->indices = 0;
  (*vec)->block_size = 1;
  (*vec)->data = NULL;
  (*vec)->type = NULL;
  return 0;
}

PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a) {
  if (x == NULL || a == NULL)
    return 1;
  *a = x->data;
  assert(*a != NULL);
  return 0;
}

PetscErrorCode VecGetArray(Vec x, PetscScalar **a) {
  if (x == NULL || a == NULL)
    return 1; // Return an error code if the input arguments are NULL
  *a = x->data;
  if (*a == NULL)
    return 1; // Return an error code if the vector data is NULL
  assert(*a != NULL);
  return 0; // Success
}

PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg) {
  *flg = vec_eq_seq(vec1, vec2) ? PETSC_TRUE : PETSC_FALSE;
  return PETSC_SUCCESS;
}

PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a) {
  if (x == NULL || a == NULL)
    return 1;
  // Reset the array pointer to NULL
  *a = NULL;
  assert(*a == NULL);
  return 0;
}

PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a) {
  if (x == NULL || a == NULL || *a == NULL)
    return 1; // Return an error only if input arguments are invalid
  // Reset the array pointer to NULL
  *a = NULL;
  assert(*a == NULL);
  return 0; // Success
}

PetscReal PetscAbsReal(PetscReal v1) { return fabs(v1); }

PetscErrorCode PetscBLASIntCast(PetscInt a, PetscBLASInt *b) {
  if (a < 0)
    return 1; // input argument, out of range

  // Check if PetscBLASInt can hold the value of a
  if ((PetscBLASInt)a != a)
    return 1; // input argument, out of range

  // Assign the value of a to b
  *b = (PetscBLASInt)a;
  return 0;
}

PetscErrorCode PetscLogFlops(PetscLogDouble n) {
  // Check if n is non-negative
  if (n < 0)
    return 1;
  // Update the total flops count
  return PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th,
                           PETSC_FLOPS_PER_OP * n);
}

PetscErrorCode VecGetOwnershipRange(Vec x, PetscInt *low, PetscInt *high) {
  assert(x && x->map);
  if (low)
    *low = x->map->rstart;
  if (high)
    *high = x->map->rend;
  return 0;
}

PetscErrorCode PetscSplitOwnership(MPI_Comm comm, PetscInt *n, PetscInt *N) {
  int size, rank;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  if (*n == PETSC_DECIDE) {
    assert(*N != PETSC_DETERMINE);
    PetscInt nlocal = *N / size + (rank < *N % size);
    *n = nlocal;
  } else if (*N == PETSC_DETERMINE) {
    PetscInt sum;
    MPI_Allreduce(n, &sum, 1, MPI_INT, MPI_SUM, comm);
    *N = sum;
  }
  return 0;
}

PetscErrorCode VecGetOwnershipRanges(Vec x, const PetscInt *ranges[]) {
  assert(x && x->map);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  PetscInt *all_ranges = (PetscInt *)malloc((size + 1) * sizeof(PetscInt));
  assert(all_ranges);

  MPI_Allgather(&x->map->rstart, 1, MPI_INT, all_ranges, 1, MPI_INT,
                MPI_COMM_WORLD);
  all_ranges[size] = x->map->N;

  *ranges = all_ranges;

  return 0;
}

PetscErrorCode VecSetSizes(Vec v, PetscInt n, PetscInt N) {
  assert(v && v->map);
  printf("VecSetSizes: v=%p, v->map=%p, n=%d, N=%d\n", (void *)v,
         (void *)v->map, n, N);

  if (n == PETSC_DECIDE) {
    if (N == PETSC_DETERMINE)
      // Both n and N are not set, this is not allowed
      return 1;
    // Set local size based on global size
    PetscCall(PetscSplitOwnership(MPI_COMM_WORLD, &n, &N));
  } else if (N == PETSC_DETERMINE) {
    // Set global size based on local size
    MPI_Allreduce(&n, &N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  v->map->n = n;
  v->map->N = N;

  // Compute rstart and rend
  MPI_Comm_rank(MPI_COMM_WORLD, &v->map->rstart);
  MPI_Scan(&n, &v->map->rend, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  v->map->rstart = v->map->rend - n;

  // Allocate memory for the vector data
  if (v->data)
    free(v->data);
  v->data = (PetscScalar *)calloc(n, sizeof(PetscScalar));
  assert(v->data);

  if (v->map->mapping)
    ISLocalToGlobalMappingDestroy(&v->map->mapping);
  PetscInt *indices = (PetscInt *)malloc(n * sizeof(PetscInt));
  for (PetscInt i = 0; i < n; i++)
    indices[i] = v->map->rstart + i;
  ISLocalToGlobalMappingCreate(MPI_COMM_WORLD, v->map->bs, n, indices,
                               PETSC_COPY_VALUES, &v->map->mapping);
  free(indices);
  printf("VecSetSizes: After allocation, v->data=%p, v->map->mapping=%p\n",
         (void *)v->data, (void *)v->map->mapping);
  return 0;
}

PetscErrorCode VecSetBlockSize(Vec v, PetscInt bs) {
  assert(bs > 0);
  v->block_size = bs;
  v->map->bs = bs;
  return 0;
}

PetscErrorCode PetscLayoutSetBlockSize(SimpleMap map, PetscInt bs) {
  assert(map);
  assert(bs > 0);
  map->bs = bs;
  return 0;
}

PetscErrorCode ISLocalToGlobalMappingCreate(MPI_Comm comm, PetscInt bs,
                                            PetscInt n,
                                            const PetscInt indices[],
                                            PetscCopyMode mode,
                                            ISLocalToGlobalMapping *mapping) {
  *mapping =
      (ISLocalToGlobalMapping)malloc(sizeof(struct _p_ISLocalToGlobalMapping));
  (*mapping)->n = n;
  (*mapping)->bs = bs;

  if (mode == PETSC_COPY_VALUES) {
    (*mapping)->indices = (PetscInt *)malloc(n * sizeof(PetscInt));
    memcpy((*mapping)->indices, indices, n * sizeof(PetscInt));
  } else if (mode == PETSC_OWN_POINTER) {
    (*mapping)->indices = (PetscInt *)indices;
  } else if (mode == PETSC_USE_POINTER) {
    (*mapping)->indices = (PetscInt *)indices;
  }

  return 0;
}

PetscErrorCode VecSetFromOptions(Vec vec) {
  assert(vec);
  // Set default type based on number of processes
  if (vec->type == NULL) {
    vec->type = (vec->map->N > vec->map->n) ? VECMPI : VECSEQ;
  }
  return 0;
}

PetscErrorCode PetscStrcmp(const char a[], const char b[], PetscBool *flg) {
  if (a == NULL || b == NULL)
    *flg = PETSC_FALSE;
  else
    *flg = (strcmp(a, b) == 0) ? PETSC_TRUE : PETSC_FALSE;
  return 0;
}

PetscErrorCode VecSetType(Vec vec, VecType newType) {
  PetscInt size;
  PetscBool match;

  // Check if the new type is the same as the current type
  if (vec->type) {
    PetscCall(PetscStrcmp(vec->type, newType, &match));
    if (match)
      return 0;
  }

  // Check for illegal MPI to sequential conversion
  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  if (size > 1 && strcmp(newType, VECSEQ) == 0) {
    return 1;
  }

  // Set the new type
  vec->type = newType;

  // Perform any necessary conversions or reinitializations here
  // For simplicity, we'll just reallocate the data if necessary
  if (vec->map && vec->map->n > 0) {
    PetscScalar *new_data =
        (PetscScalar *)malloc(vec->map->n * sizeof(PetscScalar));
    assert(new_data != NULL);
    if (vec->data) {
      memcpy(new_data, vec->data, vec->map->n * sizeof(PetscScalar));
      free(vec->data);
    }
    vec->data = new_data;
  }
  return 0;
}

PetscErrorCode VecSet(Vec x, PetscScalar alpha) {
  assert(x && x->map);
  if (x->type && strcmp(x->type, VECMPI) == 0) {
    // For MPI vectors, use VecSetValues for each local element
    PetscScalar *values = malloc(x->map->n * sizeof(PetscScalar));
    PetscInt *indices = malloc(x->map->n * sizeof(PetscInt));
    if (!values || !indices) {
      free(values);
      free(indices);
      return 1; // Memory allocation failed
    }
    for (PetscInt i = 0; i < x->map->n; i++) {
      indices[i] = x->map->rstart + i;
      values[i] = alpha;
    }
    VecSetValues(x, x->map->n, indices, values, INSERT_VALUES);
    free(values);
    free(indices);
    // Use MPI_Barrier instead of VecAssemblyBegin/End
    MPI_Barrier(PETSC_COMM_WORLD);
  } else if (x->type && (strcmp(x->type, VECSEQ) == 0 || strcmp(x->type, VECSTANDARD) == 0)) {
    // For sequential vectors, directly set the data
    assert(x->data);
    for (PetscInt i = 0; i < x->map->n; i++) {
      x->data[i] = alpha;
    }
  } else {
    return 1; // not handling this for now
  }
  return 0;
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
  if (viewer->format == PETSC_VIEWER_STDOUT_SELF ||
      viewer->format == PETSC_VIEWER_DEFAULT) {
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
  printf("VecDot: x=%p, y=%p, x->data=%p, y->data=%p, n=%d\n", (void *)x,
         (void *)y, (void *)x->data, (void *)y->data, x->map->n);
  assert(x->map->n == y->map->n);

#ifdef USE_COMPLEX
  PetscScalar local_sum = MY_COMPLEX(0.0, 0.0);
#else
  PetscScalar local_sum = 0.0;
#endif
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    local_sum = my_cadd(local_sum, my_cmul(x->data[i], my_conj(y->data[i])));
#else
    local_sum += x->data[i] * y->data[i];
#endif
  }

  MPI_Allreduce(&local_sum, val, 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                MPI_COMM_WORLD);
  return 0;
}

PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  for (PetscInt j = 0; j < nv; j++) {
    assert(x->map->n == y[j]->map->n);
    PetscScalar local_sum = 0.0;
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      local_sum =
          my_cadd(local_sum, my_cmul(x->data[i], my_conj(y[j]->data[i])));
#else
      local_sum += x->data[i] * y[j]->data[i];
#endif
    }
    MPI_Allreduce(&local_sum, &val[j], 1, MPI_DOUBLE_COMPLEX, MPI_SUM,
                  MPI_COMM_WORLD);
  }
  return 0;
}

PetscErrorCode VecCopy(Vec x, Vec y) {
  assert(x->map->n == y->map->n);
  for (PetscInt i = 0; i < x->map->n; i++) {
    y->data[i] = x->data[i];
  }
  return 0;
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
  *size = x->map->n; // Get the local size of the vector
  return 0;
}

PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
  PetscReal local_max = PETSC_MIN_REAL;
  PetscInt local_index = -1;
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    PetscReal abs_val = my_cabs(x->data[i]);
#else
    PetscReal abs_val = fabs(x->data[i]);
#endif
    if (abs_val > local_max) {
      local_max = abs_val;
      local_index = i;
    }
  }
  struct {
    PetscReal val;
    PetscInt index;
  } in, out;
  in.val = local_max;
  in.index = local_index + x->map->rstart;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
  if (p)
    *p = out.index;
  *val = out.val;
  return 0;
}

PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
  PetscReal local_min = PETSC_MAX_REAL;
  PetscInt local_index = -1;
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    PetscReal abs_val = my_cabs(x->data[i]);
#else
    PetscReal abs_val = fabs(x->data[i]);
#endif
    if (abs_val < local_min) {
      local_min = abs_val;
      local_index = i;
    }
  }
  struct {
    PetscReal val;
    PetscInt index;
  } in, out;
  in.val = local_min;
  in.index = local_index + x->map->rstart;
  MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
  if (p)
    *p = out.index;
  *val = out.val;
  return 0;
}

PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    x->data[i] = my_cmul(x->data[i], alpha);
#else
    x->data[i] *= alpha;
#endif
  }
  return 0;
}

PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[],
                        Vec x[]) {
  assert(y && x && alpha);
  for (PetscInt j = 0; j < nv; j++) {
    assert(y->map->n == x[j]->map->n);
    for (PetscInt i = 0; i < y->map->n; i++) {
#ifdef USE_COMPLEX
      y->data[i] = my_cadd(y->data[i], my_cmul(alpha[j], x[j]->data[i]));
#else
      y->data[i] += alpha[j] * x[j]->data[i];
#endif
    }
  }
  return 0;
}

PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
  assert(x->map->n == y->map->n);
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    y->data[i] = my_cadd(y->data[i], my_cmul(alpha, x->data[i]));
#else
    y->data[i] += alpha * x->data[i];
#endif
  }
  return 0;
}

PetscErrorCode VecSwap(Vec x, Vec y) {
  assert(x->map->n == y->map->n);
  for (PetscInt i = 0; i < x->map->n; i++) {
    PetscScalar temp = x->data[i];
    x->data[i] = y->data[i];
    y->data[i] = temp;
  }
  assert(x != NULL && y != NULL);
  return 0;
}

PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
  assert(w && x && y);
  assert(w->map->n == x->map->n && w->map->n == y->map->n);
  for (PetscInt i = 0; i < w->map->n; i++) {
#ifdef USE_COMPLEX
    w->data[i] = my_cadd(my_cmul(alpha, x->data[i]), y->data[i]);
#else
    w->data[i] = alpha * x->data[i] + y->data[i];
#endif
  }
  return 0;
}

PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
  assert(x->map->n == y->map->n);

  // Optimize for common values of beta
  if (PetscRealPart(beta) == 0.0 && PetscImaginaryPart(beta) == 0.0) {
    // If beta is 0, y remains unchanged
    return 0; // Success
  } else if (PetscRealPart(beta) == 1.0 && PetscImaginaryPart(beta) == 0.0) {
    // If beta is 1, y becomes the sum of x and y
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      y->data[i] = my_cadd(y->data[i], x->data[i]);
#else
      y->data[i] += x->data[i];
#endif
    }
  } else if (PetscRealPart(beta) == -1.0 && PetscImaginaryPart(beta) == 0.0) {
    // If beta is -1, y becomes the difference of y and x
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      y->data[i] = my_csub(y->data[i], x->data[i]);
#else
      y->data[i] -= x->data[i];
#endif
    }
  } else {
    // For other values of beta, perform the standard operation, y = (beta * y)
    // + x
    for (PetscInt i = 0; i < x->map->n; i++) {
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
  assert(w && x && y);
  assert(w->map->n == x->map->n && w->map->n == y->map->n);
  for (PetscInt i = 0; i < w->map->n; i++) {
#ifdef USE_COMPLEX
    w->data[i] = my_cmul(x->data[i], y->data[i]);
#else
    w->data[i] = x->data[i] * y->data[i];
#endif
  }
  return 0;
}

PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y) {
  assert(w && x && y);
  assert(w->map->n == x->map->n && w->map->n == y->map->n);

  // Compute w[i] = x[i] / y[i] component-wise
  for (PetscInt i = 0; i < w->map->n; i++) {
#ifdef USE_COMPLEX
    // Complex division
    PetscReal denom =
        y->data[i].real * y->data[i].real + y->data[i].imag * y->data[i].imag;
    if (denom == 0.0) {
      return 1; // Error code for division by zero
    }
    w->data[i].real = (x->data[i].real * y->data[i].real +
                       x->data[i].imag * y->data[i].imag) /
                      denom;
    w->data[i].imag = (x->data[i].imag * y->data[i].real -
                       x->data[i].real * y->data[i].imag) /
                      denom;
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

PetscErrorCode VecAssemblyBegin(Vec vec) { return 0; }

PetscErrorCode VecAssemblyEnd(Vec vec) { return 0; }

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

PetscErrorCode VecDuplicateVecs(Vec v, PetscInt m, Vec *V[]) {
  PetscErrorCode ierr;

  // Allocate memory for the array of vectors
  *V = (Vec *)malloc(m * sizeof(Vec));
  if (!(*V))
    return 1; // Error code for memory allocation failure

  // Create m vectors of the same type as v
  for (PetscInt i = 0; i < m; i++) {
    ierr = VecDuplicate(v, &(*V)[i]);
    if (ierr != 0) {
      // Handle error, free allocated memory, and return error code
      for (PetscInt j = 0; j < i; j++)
        VecDestroy(&(*V)[j]);
      free(*V);
      return ierr;
    }
  }
  assert(*V);
  return 0; // Success
}

PetscErrorCode VecDestroyVecs(PetscInt m, Vec *vv[]) {
  if (m <= 0 || vv == NULL || *vv == NULL) {
    return 0; // Nothing to destroy
  }

  for (PetscInt i = 0; i < m; i++) {
    if ((*vv)[i]) {
      VecDestroy(&((*vv)[i]));
    }
  }

  free(*vv);
  *vv = NULL;

  return 0;
}

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
  PetscReal local_val[2] = {0.0, 0.0}; // For NORM_1_AND_2
  switch (type) {
  case NORM_1:
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      local_val[0] += my_cabs(x->data[i]);
#else
      local_val[0] += fabs(x->data[i]);
#endif
    }
    MPI_Allreduce(local_val, val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    break;
  case NORM_2:
  case NORM_FROBENIUS:
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      local_val[0] +=
          x->data[i].real * x->data[i].real + x->data[i].imag * x->data[i].imag;
#else
      local_val[0] += x->data[i] * x->data[i];
#endif
    }
    MPI_Allreduce(local_val, val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *val = sqrt(*val);
    break;
  case NORM_INFINITY:
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      local_val[0] = fmax(local_val[0], my_cabs(x->data[i]));
#else
      local_val[0] = fmax(local_val[0], fabs(x->data[i]));
#endif
    }
    MPI_Allreduce(local_val, val, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    break;
  case NORM_1_AND_2:
    for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
      PetscReal abs_val = my_cabs(x->data[i]);
      local_val[0] += abs_val;
      local_val[1] += abs_val * abs_val;
#else
      local_val[0] += fabs(x->data[i]);
      local_val[1] += x->data[i] * x->data[i];
#endif
    }
    MPI_Allreduce(local_val, val, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    val[1] = sqrt(val[1]);
    break;
  default:
    assert(0);
    return 1; // Error for unsupported norm type
  }
  return 0;
}

PetscErrorCode VecStrideNorm(Vec v, PetscInt start, NormType ntype,
                             PetscReal *nrm) {
  assert(start >= 0 && start < v->map->bs);
  PetscReal local_val[2] = {0.0, 0.0}; // For NORM_1_AND_2
  PetscInt stride = v->map->bs;
  switch (ntype) {
  case NORM_1:
    for (PetscInt i = start; i < v->map->n; i += stride) {
#ifdef USE_COMPLEX
      local_val[0] += my_cabs(v->data[i]);
#else
      local_val[0] += fabs(v->data[i]);
#endif
    }
    MPI_Allreduce(local_val, nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    break;

  case NORM_2:
  case NORM_FROBENIUS:
    for (PetscInt i = start; i < v->map->n; i += stride) {
#ifdef USE_COMPLEX
      local_val[0] +=
          v->data[i].real * v->data[i].real + v->data[i].imag * v->data[i].imag;
#else
      local_val[0] += v->data[i] * v->data[i];
#endif
    }
    MPI_Allreduce(local_val, nrm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    *nrm = sqrt(*nrm);
    break;

  case NORM_INFINITY:
    for (PetscInt i = start; i < v->map->n; i += stride) {
#ifdef USE_COMPLEX
      local_val[0] = fmax(local_val[0], my_cabs(v->data[i]));
#else
      local_val[0] = fmax(local_val[0], fabs(v->data[i]));
#endif
    }
    MPI_Allreduce(local_val, nrm, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    break;

  case NORM_1_AND_2:
    for (PetscInt i = start; i < v->map->n; i += stride) {
#ifdef USE_COMPLEX
      PetscReal abs_val = my_cabs(v->data[i]);
      local_val[0] += abs_val;
      local_val[1] += abs_val * abs_val;
#else
      local_val[0] += fabs(v->data[i]);
      local_val[1] += v->data[i] * v->data[i];
#endif
    }
    MPI_Allreduce(local_val, nrm, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    nrm[1] = sqrt(nrm[1]);
    break;

  default:
    assert(0);
    return 1; // Error for unsupported norm type
  }
  return 0;
}

PetscErrorCode VecSetValue(Vec v, PetscInt row, PetscScalar value,
                           InsertMode mode) {
  assert(v && v->data);
  PetscInt local_row = row - v->map->rstart;
  if (local_row < 0 || local_row >= v->map->n)
    return 0;

#ifdef USE_COMPLEX
  if (mode == INSERT_VALUES) {
    v->data[local_row].real = value.real;
    v->data[local_row].imag = value.imag;
  } else if (mode == ADD_VALUES) {
    v->data[local_row].real += value.real;
    v->data[local_row].imag += value.imag;
  } else {
    return 1;
  }
#else
  if (mode == INSERT_VALUES) {
    v->data[local_row] = value;
  } else if (mode == ADD_VALUES) {
    v->data[local_row] += value;
  } else {
    return 1;
  }
#endif
  return 0;
}

PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora) {
    assert(x && x->data);
    assert(ix && y);
    assert(ni >= 0);

    for (PetscInt i = 0; i < ni; i++) {
        PetscInt local_index = ix[i] - x->map->rstart;
        
        if (local_index < 0 || local_index >= x->map->n) {
            continue;  // Skip indices that are not local to this process
        }

        #ifdef USE_COMPLEX
        if (iora == INSERT_VALUES) {
            x->data[local_index].real = y[i].real;
            x->data[local_index].imag = y[i].imag;
        } else if (iora == ADD_VALUES) {
            x->data[local_index].real += y[i].real;
            x->data[local_index].imag += y[i].imag;
        } else {
            return 1;  // Error: unsupported InsertMode
        }
        #else
        if (iora == INSERT_VALUES) {
            x->data[local_index] = y[i];
        } else if (iora == ADD_VALUES) {
            x->data[local_index] += y[i];
        } else {
            return 1;  // Error: unsupported InsertMode
        }
        #endif
    }
    return 0;
}

PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n) {
  if (a == NULL || b == NULL) {
    return 1;
  } else {
    memcpy(a, b, n);
  }
  return 0;
}

PetscErrorCode ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping) {
  if (mapping && *mapping) {
    ISLocalToGlobalMapping m = *mapping;
    printf("ISLocalToGlobalMappingDestroy: m=%p, m->n=%d, m->bs=%d, "
           "m->indices=%p\n",
           (void *)m, m->n, m->bs, (void *)m->indices);

    if (m->indices) {
      free(m->indices);
      m->indices = NULL;
    }
    free(m);
    *mapping = NULL;
  } else {
    printf("ISLocalToGlobalMappingDestroy: mapping=%p, *mapping=%p\n",
           (void *)mapping, (void *)(mapping ? *mapping : NULL));
  }
  return 0;
}

PetscErrorCode VecDestroy(Vec *v) {
  printf("Entering VecDestroy: v=%p\n", (void *)v);
  if (v && *v) {
    Vec vec = *v;
    printf("VecDestroy: vec=%p, vec->data=%p, vec->map=%p\n", (void *)vec,
           (void *)vec->data, (void *)vec->map);

    if (vec->data) {
      free(vec->data);
      vec->data = NULL;
    }

    if (vec->map) {
      printf("VecDestroy: vec->map->mapping=%p\n", (void *)vec->map->mapping);
      if (vec->map->mapping) {
        ISLocalToGlobalMappingDestroy(&vec->map->mapping);
      }
      free(vec->map);
      vec->map = NULL;
    }

    free(vec);
    *v = NULL;
  } else {
    printf("VecDestroy: v=%p, *v=%p\n", (void *)v, (void *)(v ? *v : NULL));
  }
  printf("Exiting VecDestroy\n");
  return 0;
}

PetscErrorCode PetscFinalize(void) {
  int ierr;
  ierr = MPI_Finalize();
  if (ierr != MPI_SUCCESS) {
    return 1; // or an appropriate error code
  }
  return 0;
}

PetscReal BLASnrm2_(const PetscBLASInt *n, const PetscScalar *x,
                    const PetscBLASInt *stride) {
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

PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x,
                     const PetscBLASInt *sx, const PetscScalar *y,
                     const PetscBLASInt *sy) {
  PetscBLASInt i, ix = 0, iy = 0;
#ifdef USE_COMPLEX
  PetscScalar sum = MY_COMPLEX(0.0, 0.0);
#else
  PetscScalar sum = 0.0;
#endif
  if (*n == 0)
    return sum;
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

PetscReal BLASasum_(const PetscBLASInt *n, const PetscScalar *dx,
                    const PetscBLASInt *incx) {
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
      val[0] +=
          x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
#else
      val[0] += x_array[i] * x_array[i];
#endif
    }
    val[0] = sqrt(val[0]);
    break;
  case NORM_FROBENIUS:
    for (PetscInt i = 0; i < n; i++) {
#ifdef USE_COMPLEX
      val[0] +=
          x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
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
      val[1] +=
          x_array[i].real * x_array[i].real + x_array[i].imag * x_array[i].imag;
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

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin) {
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

Vec vec_create_seq(int n, PetscScalar *data) {
  Vec vec = (Vec)malloc(sizeof(struct Vec_s));
  vec->map = (SimpleMap)malloc(sizeof(struct map_s));
  vec->map->n = n;
  vec->map->N = n;
  vec->block_size = 0;
  vec->data = (PetscScalar *)malloc(n * sizeof(PetscScalar));
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

bool vec_eq_seq(Vec vec1, Vec vec2) {
  PetscInt n = vec1->map->n;
  if (n != vec2->map->n)
    return false;
  if (vec1->map->N != vec2->map->N)
    return false;
  if (vec1->block_size != vec2->block_size)
    return false;
  PetscScalar *a1 = vec1->data, *a2 = vec2->data;
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

void vecprint_seq(const char *name, Vec vin) {
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

void vec_destroy_seq(Vec vec) {
  free(vec->data);
  free(vec->map);
  free(vec);
}

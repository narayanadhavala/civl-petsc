#include "petscvec.h"
#include <math.h>
#include <mpi.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

#define PETSC_FLOPS_PER_OP 1.0

PetscLogDouble petsc_TotalFlops = 0.0;

PetscLogDouble petsc_TotalFlops_th = 0.0;

PETSC_EXTERN PetscLogDouble petsc_TotalFlops;

PETSC_EXTERN_TLS PetscLogDouble petsc_TotalFlops_th;

PetscInt NormIds[] = {0, 1, 2, 3, 4};

PetscLogEvent VEC_MTDot = 0;

#define PetscAddLogDouble(a, b, c)                                             \
  ((PetscErrorCode)((*(a) += (c)), (*(b) += (c)), PETSC_SUCCESS))

bool $is_lessthan(PetscScalar alpha, double n) {
#ifdef USE_COMPLEX
  return alpha.real < n;
#else
  return alpha < n;
#endif
}

bool $is_greaterthan(PetscScalar alpha, double n) {
#ifdef USE_COMPLEX
  return alpha.real > n;
#else
  return alpha > n;
#endif
}

PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj, PetscInt id,
                                              PetscReal *val,
                                              PetscBool *available) {
  if (obj->realcomposedstate && obj->realcomposeddata) {
    if (id >= 0 && id < NUM_NORM_TYPES &&
        obj->realcomposedstate[id] == obj->state) {
      *val = obj->realcomposeddata[id];
      *available = PETSC_TRUE;
    } else {
      *available = PETSC_FALSE;
    }
  } else {
    *available = PETSC_FALSE;
  }
  return PETSC_SUCCESS;
}

PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj, PetscInt id,
                                              PetscReal data) {
  /* Check that the PetscObject is not NULL */
  PetscCheck(obj != NULL, PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "PetscObject is NULL");
  /* Check that the composed data arrays are allocated */
  PetscCheck(obj->realcomposedstate != NULL && obj->realcomposeddata != NULL,
             PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,
             "Real composed state or data arrays are NULL");
  /* Check that the identifier is within the valid range */
  PetscCheck(
      id >= 0 && id < NUM_NORM_TYPES, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
      "Identifier id=%d is out of valid range [0, %d)", id, NUM_NORM_TYPES);
  /* Attach the data and update the state */
  obj->realcomposeddata[id] = data;
  obj->realcomposedstate[id] = obj->state;
  return PETSC_SUCCESS;
}

PetscErrorCode PetscError(MPI_Comm comm, int line, const char *func,
                          const char *file, PetscErrorCode n, int p,
                          const char *mess, ...) {
  $assert(true);
  return n;
}

PetscErrorCode PetscValidHeaderSpecific(void *x, PetscClassId cid, int arg) {
  PetscObject obj = (PetscObject)x;
  $assert(obj != NULL, "Error: Null pointer for parameter %d", arg);
  $assert(obj->classid != PETSCFREEDHEADER,
          "Error: Object already freed for parameter %d", arg);
  //$assert(obj->classid == cid, "Error: Wrong type of object for parameter %d",
  // arg);
  return 0;
}

PetscErrorCode PetscInitialize(int *argc, char ***args, const char file[],
                               const char help[]) {
  MPI_Init(argc, args);
  return 0;
}

PetscErrorCode PetscOptionsGetInt(PetscOptions options, const char pre[],
                                  const char name[], PetscInt *ivalue,
                                  PetscBool *set) {
  return 0;
}

/* Defined in veccreate.c
        - The type Vec is defined in petscvec.h. Vec is a pointer to _p_Vec,
   which is defined in "vecimpl.h".
        - The definition of _p_Vec uses macro called PETSCHEADER, which is
   defined in petscimpl.h.
        - The macro definition uses the type _p_PetscObject which is also
   defined in petscimpl.h.
*/
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec) {
  *vec = (Vec)malloc(sizeof(struct Vec_s));
  (*vec)->comm = comm;
  (*vec)->map = (SimpleMap)malloc(sizeof(struct map_s));
  (*vec)->map->n = 0;
  (*vec)->map->N = 0;
  (*vec)->map->rstart = 0;
  (*vec)->map->rend = 0;
  (*vec)->map->bs = -1;
  (*vec)->data = NULL;
  (*vec)->type = 0;
  // Initialize hdr fields
  (*vec)->hdr.classid = VEC_CLASSID;
  (*vec)->hdr.type_name = NULL;
  (*vec)->hdr.class_name = "Vec";
  (*vec)->hdr.state = 0;
  (*vec)->read_lock_count = 0;
  // Allocate arrays
  (*vec)->hdr.realcomposedstate =
      (PetscInt *)malloc(NUM_NORM_TYPES * sizeof(PetscInt));
  (*vec)->hdr.realcomposeddata =
      (PetscReal *)malloc(NUM_NORM_TYPES * sizeof(PetscReal));
  // Initialize the arrays
  for (int i = 0; i < NUM_NORM_TYPES; i++) {
    (*vec)->hdr.realcomposedstate[i] = -1; // Indicate uninitialized
    (*vec)->hdr.realcomposeddata[i] = 0.0;
  }
  (*vec)->ops->norm = VecNorm;
  (*vec)->ops->maxpointwisedivide = VecMaxPointwiseDivide;
  (*vec)->ops->dot = VecDot;
  (*vec)->ops->tdot = VecTDot;
  (*vec)->ops->mtdot = VecMTDot;
  (*vec)->ops->max = VecMax;
  (*vec)->ops->min = VecMin;
  (*vec)->ops->scale = VecScale;
  (*vec)->ops->set = VecSet;
  (*vec)->ops->setvalues = VecSetValues;
  (*vec)->ops->setvaluesblocked = VecSetValuesBlocked;
  (*vec)->ops->copy = VecCopy;
  (*vec)->ops->concatenate = VecConcatenate;
  (*vec)->ops->axpy = VecAXPY;
  (*vec)->ops->maxpy = VecMAXPY;
  (*vec)->ops->maxpby = VecMAXPBY;
  (*vec)->ops->axpby = VecAXPBY;
  (*vec)->ops->aypx = VecAYPX;
  (*vec)->ops->waxpy = VecWAXPY;
  (*vec)->ops->axpbypcz = VecAXPBYPCZ;
  (*vec)->ops->getvalues = VecGetValues;
  (*vec)->ops->getsubvector = VecGetSubVector;
}

MPI_Comm PetscObjectComm(PetscObject obj) {
  Vec v = (Vec)obj;
  return v ? v->comm : MPI_COMM_NULL;
}

PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a) {
  *a = x->data;
  return 0;
}

PetscErrorCode VecGetArray(Vec x, PetscScalar **a) {
  *a = x->data;
  return 0;
}

PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg) {
  $assert(vec1->map->N == vec2->map->N, "Vector Global sizes mismatch");
  $assert(vec1->map->n == vec2->map->n, "Vector Local sizes mismatch");

  *flg = $vec_eq(petscToCivlVec(vec1), petscToCivlVec(vec2));
  return 0;
}

PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a) {
  *a = NULL;
  return 0;
}

PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a) {
  *a = NULL;
  return 0;
}

PetscReal PetscAbsReal(PetscReal v1) { return (PetscReal)fabs(v1); }

/* Defined in petscsys.h */
PetscErrorCode PetscBLASIntCast(PetscInt a, PetscBLASInt *b) {
  $assert(a >= 0);
  *b = (PetscBLASInt)a;
  return 0;
}

PetscErrorCode VecGetOwnershipRange(Vec x, PetscInt *low, PetscInt *high) {
  $assert(x && x->map);
  if (low)
    *low = x->map->rstart;
  if (high)
    *high = x->map->rend;
  return 0;
}

PetscErrorCode VecGetOwnershipRanges(Vec x, const PetscInt *ranges[]) {
  $assert(x && x->map);
  PetscInt *all_ranges = (PetscInt *)malloc((x->nproc + 1) * sizeof(PetscInt));
  MPI_Allgather(&x->map->rstart, 1, MPI_INT, all_ranges, 1, MPI_INT,
                PETSC_COMM_WORLD);
  all_ranges[x->nproc] = x->map->N;
  *ranges = all_ranges;
  return 0;
}

PetscErrorCode PetscSplitOwnership(MPI_Comm comm, PetscInt *n, PetscInt *N) {
  int size, rank, buf;
  MPI_Comm_size(comm, &size);
  MPI_Comm_rank(comm, &rank);
  // check all processess agree on N ...
  MPI_Allreduce(N, &buf, 1, MPI_INT, MPI_MIN, comm);
  $assert(*N == buf);
  if (*n == PETSC_DECIDE) { // check all processess agree *n=PETSC_DECIDE
    MPI_Allreduce(n, &buf, 1, MPI_INT, MPI_MIN, comm);
    $assert(*n == buf);
    $assert(*N != PETSC_DETERMINE);
    $assert(*N >= 0);
    PetscInt nlocal = *N / size + (rank < *N % size);
    *n = nlocal;
  } else if (*N == PETSC_DETERMINE) {
    $assert(*n >= 0);
    MPI_Allreduce(n, N, 1, MPI_INT, MPI_SUM, comm);
  } else { // check the sum of all n's is big N
    MPI_Allreduce(n, &buf, 1, MPI_INT, MPI_SUM, comm);
    $assert(*N == buf);
  }
  return 0;
}

PetscErrorCode VecSetSizes(Vec v, PetscInt n, PetscInt N) {
  PetscInt rank;
  MPI_Comm_rank(v->comm, &rank);
  MPI_Comm_size(v->comm, &v->nproc);
  PetscSplitOwnership(v->comm, &n, &N);
  v->map->nproc = v->nproc;
  v->map->n = n;
  v->map->N = N;
  PetscInt start_value = 0;
  // Compute rstart and rend
  MPI_Exscan(&n, &start_value, 1, MPI_INT, MPI_SUM, v->comm);
  if (rank == 0)
    start_value = 0;
  v->map->rstart = start_value;
  v->map->rend = start_value + n;
  // Allocate memory for the vector data
  if (v->data)
    free(v->data);
  if (n == 0)
    v->data = NULL;
  else
    v->data = (PetscScalar *)malloc(n * sizeof(PetscScalar));
  return 0;
}

PetscErrorCode VecSetUp(Vec v) {
  PetscMPIInt size;
  if (!v->type) {
    MPI_Comm_size(v->comm, &size);
    v->type = (size == 1) ? VECSEQ : VECMPI;
  }
  return 0;
}

PetscErrorCode VecGetSubVector(Vec X, IS is, Vec *Y) {
  int first, n;
  $vec civl_X = petscToCivlVec(X);
  $assert(is && is->data, "VecGetSubVector: Index set is must be valid.");
  first = X->map->rstart;
  n = X->map->n;
  int rank;
  MPI_Comm_rank(X->comm, &rank);
  $print("Rank = ", rank, "\n");
  // Extract subvector
  $vec civl_Y = $vec_subseq(civl_X, first, n);
  Vec newVec = civlToPetscVec(civl_Y, PETSC_DECIDE, (X)->comm);
  *Y = newVec;
  return 0;
}

PetscErrorCode VecRestoreSubVector(Vec X, IS is, Vec *Y) {
  if (!Y || !(*Y))
    return 0;
  VecDestroy(Y);
  return 0;
}

PetscErrorCode VecSetBlockSize(Vec v, PetscInt bs) {
  $assert(bs > 0);
  v->map->bs = bs;
  return 0;
}

PetscErrorCode VecSetFromOptions(Vec vec) {
  if (vec->nproc > 1) {
    vec->type = VECMPI;
    vec->comm = PETSC_COMM_WORLD;
  } else {
    vec->type = VECSEQ;
    vec->comm = PETSC_COMM_SELF;
  }
  return 0;
}

PetscErrorCode VecSetType(Vec vec, VecType newType) {
  // Check if the new type is the same as the current type
  if (vec->type == newType)
    return 0;
  // Update the type and communicator based on the new type
  switch (newType) {
  case VECSEQ:
    // Sequential vector, requires one process and PETSC_COMM_SELF
    $assert(vec->nproc > 1,
            "Error: Cannot set a parallel vector to sequential.");
    vec->type = VECSEQ;
    vec->comm = PETSC_COMM_SELF;
    vec->ops->norm = VecNorm_Seq;
    break;
  case VECMPI:
    // Parallel vector, requires more than one process and PETSC_COMM_WORLD
    $assert(vec->nproc > 1,
            "Error: Cannot set a sequential vector to parallel.");
    vec->type = VECMPI;
    vec->comm = PETSC_COMM_WORLD;
    vec->ops->norm = VecNorm;
    break;
  case VECSTANDARD:
    // Standard vector type, decide based on the number of processes
    if (vec->nproc > 1) {
      vec->type = VECMPI;
      vec->comm = PETSC_COMM_WORLD;
      vec->ops->norm = VecNorm;
    } else {
      vec->type = VECSEQ;
      vec->comm = PETSC_COMM_SELF;
      vec->ops->norm = VecNorm_Seq;
    }
    break;
  default:
    // Unsupported type
    printf("Error: Unknown vector type.\n");
    return -1;
  }
  return 0;
}

PetscErrorCode VecGetType(Vec vec, VecType *type) {
  *type = (vec)->type;
  return 0;
}

PetscBool PetscIsNanScalar(PetscScalar v) {
  return PetscIsNanReal(PetscAbsScalar(v));
}

PetscBool PetscEqualReal(PetscReal a, PetscReal b) {
  return (a == b) ? PETSC_TRUE : PETSC_FALSE;
}

PetscErrorCode VecSet(Vec x, PetscScalar alpha) {
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  for (int i = 0; i < x->map->n; i++)
    x->data[i] = alpha;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecView(Vec vec, PetscViewer viewer) {
  $vec civl_vec = petscToCivlVec(vec);
  $vec_print(civl_vec);
  vec = civlToPetscVec(civl_vec, PETSC_DECIDE, vec->comm);
  return 0;
}

PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val) {
  $vec civl_vec_x = petscToCivlVec(x), civl_vec_y = petscToCivlVec(y);
  STYPE dot_product = $vec_dot(civl_vec_x, civl_vec_y);
  *val = (PetscScalar)dot_product;
  return 0;
}

PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val) {
  $vec civl_vec_x = petscToCivlVec(x);
  $vec civl_vec_y = petscToCivlVec(y);

  $assert(civl_vec_x.len == civl_vec_y.len, "Vector lengths must match");

  STYPE dot_product = scalar_make(0.0, 0.0);
  for (int i = 0; i < civl_vec_x.len; i++) {
    dot_product = scalar_add(
        dot_product, scalar_mul(civl_vec_y.data[i], civl_vec_x.data[i]));
  }
  *val = (PetscScalar)dot_product;
  return 0;
}

PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  $assert(x && y && val, "Input pointers must be non-null.");
  $assert(nv >= 0, "Number of vectors (given %d) cannot be negative", nv);

  // Handle nv=0 case (no operation)
  if (nv == 0)
    return 0;

  // Vector compatibility checks
  for (PetscInt i = 0; i < nv; i++) {
    $assert(x->map->n == y[i]->map->n,
            "VecMTDot: All vectors must have the same size.");
  }

  // Compute dot products
  for (PetscInt j = 0; j < nv; j++) {
    PetscScalar local_sum = scalar_make(0.0, 0.0);

    // Compute the transpose dot product
    for (PetscInt i = 0; i < x->map->n; i++) {
      // In complex case, multiply x[i] with the conjugate of y[j][i]
      local_sum = scalar_add(
          local_sum, scalar_mul(x->data[i], scalar_conj(y[j]->data[i])));
    }

    // Perform a global sum across all MPI processes
#ifdef USE_COMPLEX
    double in_tmp[2] = {local_sum.real, local_sum.imag};
    double out_tmp[2];
    MPI_Allreduce(in_tmp, out_tmp, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    val[j] = $make_complex(out_tmp[0], out_tmp[1]);
#else
    MPI_Allreduce(&local_sum, &val[j], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  }
  return 0;
}

PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val) {
  PetscScalar dotProduct;
  VecDot(x, y, &dotProduct);
  *val = PetscRealPart(dotProduct);
  return 0;
}

PetscErrorCode VecNormalize(Vec x, PetscReal *val) {
  PetscReal norm;
  $assert(x, "Error: VecNormalize called with NULL vector.\n");

  /* Compute the norm using VecNorm */
  VecNorm(x, NORM_2, &norm);

  /* Check for zero norm */
  if (norm == 0.0) {
    $print("Vector has zero norm; cannot normalize.\n");
    if (val)
      *val = norm;
    return 0;
  }

  /* Check for Inf or NaN */
  if (!isfinite(norm)) {
    $print("Vector has Inf or NaN norm; cannot normalize.\n");
    if (val)
      *val = norm;
    return 0;
  }
  /* Scale the vector by 1/norm */
  PetscScalar scale = scalar_of(1.0 / norm);
  VecScale(x, scale);

  if (val)
    *val = norm;
  return 0;
}

PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  for (PetscInt j = 0; j < nv; j++) {
    $assert(x->map->n == y[j]->map->n);
    PetscScalar local_sum = scalar_make(0.0, 0.0);
    for (PetscInt i = 0; i < x->map->n; i++) {
      local_sum = scalar_add(
          local_sum, scalar_mul(x->data[i], scalar_conj(y[j]->data[i])));
    }
#ifdef USE_COMPLEX
    double in_tmp[2] = {local_sum.real, local_sum.imag};
    double out_tmp[2];
    MPI_Allreduce(in_tmp, out_tmp, 2, PETSC_REAL, MPI_SUM, PETSC_COMM_WORLD);
    val[j] = $make_complex(out_tmp[0], out_tmp[1]);
#else
    MPI_Allreduce(&local_sum, &val[j], 1, PETSC_REAL, MPI_SUM,
                  PETSC_COMM_WORLD);
#endif
  }
  return 0;
}

PetscErrorCode VecCopyAsync_Private(Vec x, Vec y, PetscDeviceContext dctx) {
  PetscBool flgs[4];
  PetscReal norms[4] = {0.0, 0.0, 0.0, 0.0};

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  if (x == y)
    PetscFunctionReturn(PETSC_SUCCESS);
  VecCheckSameLocalSize(x, 1, y, 2);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(y, 2));

#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i = 0; i < 4; i++) {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i],
                                             &norms[i], &flgs[i]));
  }
#endif

  PetscCall(PetscLogEventBegin(VEC_Copy, x, y, 0, 0));
#if defined(PETSC_USE_MIXED_PRECISION)
  extern PetscErrorCode VecGetArray(Vec, double **);
  extern PetscErrorCode VecRestoreArray(Vec, double **);
  extern PetscErrorCode VecGetArray(Vec, float **);
  extern PetscErrorCode VecRestoreArray(Vec, float **);
  extern PetscErrorCode VecGetArrayRead(Vec, const double **);
  extern PetscErrorCode VecRestoreArrayRead(Vec, const double **);
  extern PetscErrorCode VecGetArrayRead(Vec, const float **);
  extern PetscErrorCode VecRestoreArrayRead(Vec, const float **);
  if ((((PetscObject)x)->precision == PETSC_PRECISION_SINGLE) &&
      (((PetscObject)y)->precision == PETSC_PRECISION_DOUBLE)) {
    PetscInt i, n;
    const float *xx;
    double *yy;
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(y, &yy));
    PetscCall(VecGetLocalSize(x, &n));
    for (i = 0; i < n; i++)
      yy[i] = xx[i];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
  } else if ((((PetscObject)x)->precision == PETSC_PRECISION_DOUBLE) &&
             (((PetscObject)y)->precision == PETSC_PRECISION_SINGLE)) {
    PetscInt i, n;
    float *yy;
    const double *xx;
    PetscCall(VecGetArrayRead(x, &xx));
    PetscCall(VecGetArray(y, &yy));
    PetscCall(VecGetLocalSize(x, &n));
    for (i = 0; i < n; i++)
      yy[i] = (float)xx[i];
    PetscCall(VecRestoreArrayRead(x, &xx));
    PetscCall(VecRestoreArray(y, &yy));
  } else {
    PetscUseTypeMethod(x, copy, y);
  }
#else
  VecMethodDispatch(x, dctx, VecAsyncFnName(Copy), copy,
                    (Vec, Vec, PetscDeviceContext), y);
#endif

  PetscCall(PetscObjectStateIncrease((PetscObject)y));
#if !defined(PETSC_USE_MIXED_PRECISION)
  for (PetscInt i = 0; i < 4; i++) {
    if (flgs[i]) {
      PetscCall(
          PetscObjectComposedDataSetReal((PetscObject)y, NormIds[i], norms[i]));
    }
  }
#endif

  PetscCall(PetscLogEventEnd(VEC_Copy, x, y, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecCopy(Vec x, Vec y) {
  $assert(y->read_lock_count == 0,
          "Cannot Copy values: Vector is locked for reading.");
  $assert(x, "Vector cannot be null");
  int n = x->map->n;
  for (int i = 0; i < n; i++)
    y->data[i] = x->data[i];
  return 0;
}

PetscErrorCode VecGetSize(Vec x, PetscInt *size) {
  if (x->map)
    *size = x->map->N;
  else
    return 1; // Error code if the map is not available
  return 0;
}

PetscErrorCode VecGetLocalSize(Vec x, PetscInt *size) {
  *size = x->map->n; // Get the local size of the vector
  return 0;
}

PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
  if (x->map->N == 0) {
    if (p)
      *p = -1;
    *val = PETSC_MIN_REAL;
    return 0;
  }
  PetscReal local_max = PETSC_MIN_REAL;
  PetscInt local_index = -1;
  for (PetscInt i = 0; i < x->map->n; i++) {
#ifdef USE_COMPLEX
    PetscReal a = $creal(x->data[i]);
#else
    PetscReal a = x->data[i];
#endif
    if (a > local_max) {
      local_max = a;
      local_index = i;
    }
  }

  // Perform separate reductions for value and index
  PetscReal global_max;
  PetscInt global_index;
  int rank;

  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Allreduce(&local_max, &global_max, 1, PETSC_REAL, MPI_MAX,
                PETSC_COMM_WORLD);

  // Only processes with the maximum value participate in the index reduction
  PetscInt participating = (local_max == global_max)
                               ? GLOBAL_INDEX(x, rank, local_index)
                               : x->map->N;
  MPI_Allreduce(&participating, &global_index, 1, MPI_INT, MPI_MIN,
                PETSC_COMM_WORLD);

  if (p)
    *p = global_index;
  *val = global_max;
  return 0;
}

PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
  if (x->map->N == 0) {
    if (p)
      *p = -1;
    *val = PETSC_MAX_REAL;
    return 0;
  }
  PetscReal local_min = PETSC_MAX_REAL;
  PetscInt local_index = -1;
  for (PetscInt i = 0; i < x->map->n; i++) {
    PetscReal a = PetscRealPart(x->data[i]);
    if (a < local_min) {
      local_min = a;
      local_index = i;
    }
  }
  // Perform separate reductions for value and index
  PetscReal global_min;
  PetscInt global_index;
  int rank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Allreduce(&local_min, &global_min, 1, PETSC_REAL, MPI_MIN,
                PETSC_COMM_WORLD);
  // Only processes with the minimum value participate in the index reduction
  PetscInt participating = (local_min == global_min)
                               ? GLOBAL_INDEX(x, rank, local_index)
                               : x->map->N;
  MPI_Allreduce(&participating, &global_index, 1, MPI_INT, MPI_MIN,
                PETSC_COMM_WORLD);
  if (p)
    *p = global_index;
  *val = global_min;
  return 0;
}

PetscErrorCode PetscLogFlops(PetscLogDouble n) {
  $assert(n >= 0);
  return PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th,
                           PETSC_FLOPS_PER_OP * n);
}

PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
  $assert(x->read_lock_count == 0,
          "Cannot scale values: Vector is locked for reading.");
  for (PetscInt i = 0; i < x->map->n; i++)
    x->data[i] = scalar_mul(x->data[i], alpha);
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[],
                        Vec x[]) {
  // Input validation matching actual implementation
  $assert(y->read_lock_count == 0,
          "Cannot MAXPY values: Vector is locked for reading.");
  $assert(nv >= 0, "Number of vectors cannot be negative");
  if (nv > 0)
    $assert(y && x && alpha, "Input vectors and scalars must not be NULL.");

  // Handle nv=0 case (no operation)
  if (nv == 0)
    return 0;

  // Vector compatibility checks
  for (PetscInt i = 0; i < nv; i++) {
    $assert(y->map->n == x[i]->map->n,
            "VecMAXPY_spec: All vectors must have the same size.");
    $assert(y != x[i], "Input vectors cannot contain y itself");
  }

  // Check for all-zero alpha case
  int zero_alphas = 0;
  for (PetscInt i = 0; i < nv; i++)
    zero_alphas += scalar_eq(alpha[i], scalar_zero);

  if (zero_alphas == nv) {
    // Special case: no operation if all alphas are zero
    return 0;
  }

  // General case: y = y + sum(alpha[i] * x[i])
  for (PetscInt i = 0; i < y->map->n; i++)
    for (PetscInt j = 0; j < nv; j++)
      y->data[i] = scalar_add(y->data[i], scalar_mul(alpha[j], x[j]->data[i]));

  // Update vector state
  y->hdr.state++;
  return 0;
}

PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
  $assert(y->read_lock_count == 0,
          "VecAXPY: Cannot modify vector y because it is locked for reading.");
  $assert(x->map->n == y->map->n,
          "VecAXPY: Input vectors x and y must have the same local size.");
  $vec c_x = petscToCivlVec(x), c_y = petscToCivlVec(y),
       c_z = $vec_add($vec_scalar_mul(alpha, c_x), c_y);
  civlToPetscVecCopy(c_z, y);
  return 0;
}

PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x) {
  int n = x->map->n;
  $assert(y->read_lock_count == 0,
          "Cannot AXPBY values: Vector is locked for reading.");
  $assert(n == y->map->n, "VecAXPBY_spec: Vectors must have the same size.");
  $vec c_x = petscToCivlVec(x), c_y = petscToCivlVec(y);
  $vec c_z = $vec_add($vec_scalar_mul(alpha, c_x), $vec_scalar_mul(beta, c_y));
  civlToPetscVecCopy(c_z, y);
  return 0;
}

PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y) {
  $assert(z->read_lock_count == 0,
          "Cannot VecAXPBYPCZ values: Vector is locked for reading.");
  $assert(x->map->n == y->map->n && y->map->n == z->map->n,
          "VecAXPBYPCZ_spec: Vectors must have the same size.");
  if (scalar_eq(scalar_of(0), alpha) && scalar_eq(scalar_of(0), beta) &&
      scalar_eq(scalar_of(0), gamma)) {
    return 0;
  }
  $vec c_x = petscToCivlVec(x), c_y = petscToCivlVec(y),
       c_z = petscToCivlVec(z);
  $vec c_w = $vec_add(
      $vec_add($vec_scalar_mul(alpha, c_x), $vec_scalar_mul(beta, c_y)),
      $vec_scalar_mul(gamma, c_z));
  civlToPetscVecCopy(c_w, z);
  return 0;
}

PetscErrorCode VecMAXPBY(Vec y, PetscInt nv, const PetscScalar alpha[],
                         PetscScalar beta, Vec x[]) {
  $assert(y->read_lock_count == 0,
          "Cannot MAXPBY values: Vector is locked for reading.");
  $assert(nv >= 0, "Number of vectors cannot be negative");
  if (nv > 0)
    $assert(y && x && alpha, "Input vectors and scalars must not be NULL.");

  // Handle nv=0 case (just scale by beta)
  if (nv == 0) {
    for (int i = 0; i < y->map->n; i++)
      y->data[i] = scalar_mul(beta, y->data[i]);
    return 0;
  }

  // Vector compatibility checks
  for (int v = 0; v < nv; v++) {
    $assert(y->map->n == x[v]->map->n,
            "VecMAXPBY_spec: All vectors must have the same size.");
    $assert(y != x[v], "Input vectors cannot contain y itself");
  }

  // Check for all-zero alpha case
  int zero_alphas = 0;
  for (int v = 0; v < nv; v++)
    zero_alphas += scalar_eq(alpha[v], scalar_zero);

  if (zero_alphas == nv) {
    // Special case: just scale by beta
    for (int i = 0; i < y->map->n; i++)
      y->data[i] = scalar_mul(beta, y->data[i]);
  } else {
    // General case: beta*y + sum(alpha[v]*x[v])
    for (int i = 0; i < y->map->n; i++) {
      STYPE temp = scalar_mul(beta, y->data[i]);
      for (int v = 0; v < nv; v++)
        temp = scalar_add(temp, scalar_mul(alpha[v], x[v]->data[i]));
      y->data[i] = temp;
    }
  }
  return 0;
}

PetscErrorCode VecSwap(Vec x, Vec y) {
  $assert(x->read_lock_count == 0 && y->read_lock_count == 0,
          "Cannot swap values: Vector is locked for reading.");
  $assert(x->map->n == y->map->n);
  for (PetscInt i = 0; i < x->map->n; i++) {
    PetscScalar temp = x->data[i];
    x->data[i] = y->data[i];
    y->data[i] = temp;
  }
  $assert(x != NULL && y != NULL);
  return 0;
}

PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
  // Input validation matching actual implementation
  $assert(w && x && y, "VecWAXPY: Vectors cannot be NULL");
  $assert(w->map->n == x->map->n && w->map->n == y->map->n,
          "VecWAXPY: Vectors must have the same size.");
  $assert(w != y,
          "VecWAXPY: Result vector w cannot be same as input vector y.");
  $assert(w != x,
          "VecWAXPY: Result vector w cannot be same as input vector x.");

  // Handle alpha = 0 case (w = y)
  if (scalar_eq(scalar_of(0), alpha)) {
    for (int i = 0; i < w->map->n; i++)
      w->data[i] = y->data[i];
    return 0;
  }

  // General case: w = alpha * x + y
  for (int i = 0; i < w->map->n; i++)
    w->data[i] = scalar_add(scalar_mul(alpha, x->data[i]), y->data[i]);

  return 0;
}

PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
  $assert(y->read_lock_count == 0,
          "Cannot AYPX values: Vector is locked for reading.");
  $assert(x->map->n == y->map->n);
  // Optimize for common values of beta
  if (scalar_eq(scalar_of(0), beta)) {
    // If beta is 0, y remains unchanged
    for (int i = 0; i < x->map->n; i++)
      y->data[i] = x->data[i];
  } else if (scalar_eq(scalar_of(1), beta)) {
    // If beta is 1, y becomes the sum of x and y
    for (int i = 0; i < x->map->n; i++)
      y->data[i] = scalar_add(y->data[i], x->data[i]);
  } else if (scalar_eq(scalar_of(-1), beta)) {
    // If beta is -1, y becomes the difference of y and x
    for (int i = 0; i < x->map->n; i++)
      y->data[i] = scalar_sub(y->data[i], x->data[i]);
  } else {
    // For other values of beta, perform the standard operation, y = (beta * y)
    // + x
    for (int i = 0; i < x->map->n; i++)
      y->data[i] = scalar_add(scalar_mul(beta, y->data[i]), x->data[i]);
  }
  $assert(y);
  return 0;
}

PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y) {
  $assert(w && x && y);
  $assert(w->map->n == x->map->n && w->map->n == y->map->n);
  for (PetscInt i = 0; i < w->map->n; i++)
    w->data[i] = scalar_mul(x->data[i], y->data[i]);
  return 0;
}

PetscErrorCode VecMaxPointwiseDivide(Vec x, Vec y, PetscReal *max) {
  $assert(x && y && max);
  $assert(x->map->n == y->map->n);
  $assert(x->comm != MPI_COMM_NULL, "MPI communicator not initialized.");

  PetscReal local_max_sq = 0.0, global_max_val = 0.0;

  for (PetscInt i = 0; i < x->map->n; i++) {
    // Handle division by zero by substituting denominator with 1.0
    PetscReal y_abs = PetscAbsScalar(y->data[i]);
    PetscScalar denom = (y_abs > 0.0) ? y->data[i] : scalar_make(1.0, 0.0);
    PetscScalar value = scalar_div(x->data[i], denom);

    // Compute squared magnitude to avoid sqrt on negative (impossible case)
    PetscReal abs_sq = PetscRealPart(scalar_mul(value, scalar_conj(value)));
    if (abs_sq > local_max_sq)
      local_max_sq = abs_sq;
  }

  PetscReal local_max = sqrt(local_max_sq); // Safe: input is non-negative
  MPI_Allreduce(&local_max, &global_max_val, 1, MPI_DOUBLE, MPI_MAX, x->comm);
  *max = global_max_val;
  return 0;
}

PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y) {
  $assert(w && x && y, "Vectors cannot be NULL.");
  $assert(w->map->n == x->map->n && w->map->n == y->map->n,
          "Vectors Local sizes mismatch.");

  // Compute w[i] = x[i] / y[i] component-wise
  for (PetscInt i = 0; i < w->map->n; i++) {
#ifdef USE_COMPLEX
    // Complex division
    PetscReal denom =
        y->data[i].real * y->data[i].real + y->data[i].imag * y->data[i].imag;
    if (denom == 0.0)
      return 1; // Error code for division by zero
    w->data[i].real = (x->data[i].real * y->data[i].real +
                       x->data[i].imag * y->data[i].imag) /
                      denom;
    w->data[i].imag = (x->data[i].imag * y->data[i].real -
                       x->data[i].real * y->data[i].imag) /
                      denom;
#else
    // Real division
    if (y->data[i] == 0.0)
      return 1; // Error code for division by zero
    w->data[i] = x->data[i] / y->data[i];
#endif
  }
  return 0; // Success
}

PetscErrorCode VecAssemblyBegin(Vec vec) { return 0; }

PetscErrorCode VecAssemblyEnd(Vec vec) { return 0; }

PetscErrorCode VecDuplicate(Vec v, Vec *newv) {
  // Allocate memory for the new vector
  *newv = (Vec)malloc(sizeof(struct Vec_s));
  $assert(*newv != NULL); // Ensure memory allocation succeeded
  // Copy the size information from the original vector
  $assert(v->map != NULL); // Ensure original vector has a valid map
  (*newv)->map = (SimpleMap)malloc(sizeof(struct map_s));
  $assert((*newv)->map != NULL); // Ensure memory allocation succeeded
  // Copy map information
  (*newv)->map->n = v->map->n;
  (*newv)->map->N = v->map->N;
  (*newv)->map->rstart = v->map->rstart;
  (*newv)->map->rend = v->map->rend;
  (*newv)->map->bs = v->map->bs;
  (*newv)->type = v->type;
  // Allocate memory for the data array
  PetscInt local_size = v->map->n;
  $assert(local_size >= 0); // Ensure local size is non-negative
  (*newv)->data = (PetscScalar *)malloc(local_size * sizeof(PetscScalar));
  $assert((*newv)->data != NULL); // Ensure memory allocation succeeded
  (*newv)->comm = v->comm;
  return 0;
}

PetscErrorCode VecDuplicateVecs(Vec v, PetscInt m, Vec *V[]) {
  $assert(v != NULL); // Ensure input vector is not NULL
  $assert(m > 0);     // Ensure number of vectors to create is positive
  $assert(V != NULL); // Ensure output pointer is not NULL
  // Allocate memory for the array of vectors
  *V = (Vec *)malloc(m * sizeof(Vec));
  $assert(*V != NULL); // Ensure memory allocation succeeded
  // Create m vectors of the same type as v
  for (PetscInt i = 0; i < m; i++) {
    PetscErrorCode ierr = VecDuplicate(v, &(*V)[i]);
    if (ierr != 0) {
      // Handle error, free allocated memory
      for (PetscInt j = 0; j < i; j++)
        VecDestroy(&(*V)[j]);
      free(*V);
      *V = NULL;
      return ierr;
    }
    $assert((*V)[i] != NULL); // Ensure each vector was created successfully
  }
  return 0;
}

PetscErrorCode VecDestroyVecs(PetscInt m, Vec *vv[]) {
  // Ensure the pointer to array of vectors is not NULL
  $assert(vv != NULL);
  // Nothing to destroy if m is non-positive or the array is NULL
  if (m <= 0 || *vv == NULL)
    return 0;
  // Destroy each vector
  for (PetscInt i = 0; i < m; i++) {
    if ((*vv)[i] != NULL) {
      PetscErrorCode ierr = VecDestroy(&((*vv)[i]));
      $assert(ierr == 0);        // Ensure VecDestroy succeeded
      $assert((*vv)[i] == NULL); // Ensure VecDestroy pointer is NULL
    }
  }
  free(*vv); // Free the array of vector pointers
  *vv = NULL;
  return 0;
}

PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available,
                                PetscReal *val) {
  PetscInt id = NormIds[type];      // Map NormType to identifier
  PetscObject obj = (PetscObject)x; // Cast Vec to PetscObject
  $assert(obj != NULL, "Vector x is NULL");
  $assert(id >= 0 && id < NUM_NORM_TYPES, "Invalid NormType");
  if (obj->realcomposedstate && obj->realcomposeddata) {
    if (obj->realcomposedstate[id] == obj->state) {
      *val = obj->realcomposeddata[id];
      *available = PETSC_TRUE;
    } else {
      *available = PETSC_FALSE;
    }
  } else {
    *available = PETSC_FALSE;
  }
  return PETSC_SUCCESS;
}
/*
typedef enum NORM_TYPE {
  NORM_1 = 0,
  NORM_2 = 1,
  NORM_FROBENIUS = 2,
  NORM_INFINITY = 3,
  NORM_1_AND_2 = 4
} NormType;
*/

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
  $vec civlVec = petscToCivlVec(x);
  switch (type) {
  case 0:
    val[0] = $vec_norm(civlVec, 1);
    break;
  case 2:
    val[0] = $vec_norm(civlVec, 2);
    break;
  case 1:
    val[0] = $vec_norm(civlVec, 2);
    break;
  case 3:
    val[0] = $vec_norm(civlVec, $norm_infty);
    break;
  case 4:
    val[0] = $vec_norm(civlVec, 1);
    val[1] = $vec_norm(civlVec, 2);
    break;
  default:
    $assert(0, "Invalid norm type");
    return 1;
  }
  x->hdr.state++;
  //  Store the computed norm
  PetscInt id = NormIds[(int)type];
  if (id >= 0 && id < NUM_NORM_TYPES) {
    x->hdr.realcomposeddata[id] = val[0];
    x->hdr.realcomposedstate[id] = x->hdr.state;
  }
  return 0;
}

PetscErrorCode VecStrideNorm(Vec x, PetscInt start, NormType ntype,
                             PetscReal *val) {
  PetscInt n, stride;
  PetscReal local_val[2] = {0.0, 0.0}; // For NORM_1_AND_2
  // Check if the map and block_size are available and retrieve the local size
  // and stride
  if (x->map && x->map->bs > 0) {
    n = x->map->n;
    stride = x->map->bs;
  } else {
    return 1; // Error code if the map or block_size is not available
  }

  switch (ntype) {
  case NORM_1:
    for (PetscInt i = start; i < n; i += stride)
      local_val[0] += PetscAbsScalar(x->data[i]);
    MPI_Allreduce(local_val, val, 1, PETSC_REAL, MPI_SUM, PETSC_COMM_WORLD);
    break;

  case NORM_FROBENIUS:
  case NORM_2:
    for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
      PetscScalar d = x->data[i];
      local_val[0] += d.real * d.real + d.imag * d.imag;
#else
      PetscScalar d = x->data[i];
      local_val[0] += d * d;
#endif
    }
    MPI_Allreduce(local_val, val, 1, PETSC_REAL, MPI_SUM, PETSC_COMM_WORLD);
    *val = sqrt(*val);
    break;

  case NORM_INFINITY:
    for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
      local_val[0] = fmax(local_val[0], $cabs(x->data[i]));
#else
      local_val[0] = fmax(local_val[0], fabs(x->data[i]));
#endif
    }
    MPI_Allreduce(local_val, val, 1, PETSC_REAL, MPI_MAX, PETSC_COMM_WORLD);
    break;

  case NORM_1_AND_2:
    for (PetscInt i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
      PetscScalar d = x->data[i];
      local_val[0] += $cabs(d);
      local_val[1] += d.real * d.real + d.imag * d.imag;
#else
      PetscScalar d = x->data[i];
      local_val[0] += fabs(d);
      local_val[1] += d * d;
#endif
    }
    MPI_Allreduce(local_val, val, 2, PETSC_REAL, MPI_SUM, PETSC_COMM_WORLD);
    val[1] = sqrt(val[1]);
    break;

  default:
    $assert(0); // not dealing with this for now
    return 1;
  }
  return 0;
}

PetscErrorCode VecSetValue(Vec v, PetscInt row, PetscScalar value,
                           InsertMode mode) {
  $assert(v->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  $assert(v && v->data);
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
  if (mode == INSERT_VALUES)
    v->data[local_row] = value;
  else if (mode == ADD_VALUES)
    v->data[local_row] += value;
  else
    return 1;
#endif
  v->hdr.state++;
  return 0;
}

PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            const PetscScalar y[], InsertMode iora) {
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  $assert(x && x->data);
  $assert(ix && y);
  $assert(ni >= 0);

  for (PetscInt i = 0; i < ni; i++) {
    PetscInt local_index = ix[i] - x->map->rstart;
    if (local_index < 0 || local_index >= x->map->n)
      continue; // Skip indices that are not local to this process
#ifdef USE_COMPLEX
    if (iora == INSERT_VALUES) {
      x->data[local_index].real = y[i].real;
      x->data[local_index].imag = y[i].imag;
    } else if (iora == ADD_VALUES) {
      x->data[local_index].real += y[i].real;
      x->data[local_index].imag += y[i].imag;
    } else {
      return 1; // Error: unsupported InsertMode
    }
#else
    if (iora == INSERT_VALUES)
      x->data[local_index] = y[i];
    else if (iora == ADD_VALUES)
      x->data[local_index] += y[i];
    else
      return 1; // Error: unsupported InsertMode
#endif
  }
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora) {
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  $assert(x && x->data);
  $assert(ix && y);
  $assert(ni >= 0);

  /* Retrieve the block size; must be > 0 for meaningful blocked ops */
  PetscInt bs = x->map->bs;
  $assert(bs > 0, "VecSetValuesBlocked: block size must be positive.");

  /* Each ix[i] refers to a block index, not a single element index */
  for (PetscInt i = 0; i < ni; i++) {
    /* Convert block index to local block index by subtracting rstart in block
       units. The total number of local blocks is x->map->n / bs. */
    PetscInt local_block = ix[i] - (x->map->rstart / bs);

    /* If this block index is out of local range, skip it. */
    if (local_block < 0 || local_block >= (x->map->n / bs))
      continue;

    /* For each entry j in the block, compute the actual element index */
    for (PetscInt j = 0; j < bs; j++) {
      PetscInt elem_index = bs * local_block + j;

#ifdef USE_COMPLEX
      /* Complex version: set or add real/imag parts */
      if (iora == INSERT_VALUES) {
        x->data[elem_index].real = y[bs * i + j].real;
        x->data[elem_index].imag = y[bs * i + j].imag;
      } else if (iora == ADD_VALUES) {
        x->data[elem_index].real += y[bs * i + j].real;
        x->data[elem_index].imag += y[bs * i + j].imag;
      } else {
        return 1; // Unsupported InsertMode in this stub
      }
#else
      /* Real version: set or add scalar values */
      if (iora == INSERT_VALUES)
        x->data[elem_index] = y[bs * i + j];
      else if (iora == ADD_VALUES)
        x->data[elem_index] += y[bs * i + j];
      else
        return 1; // Unsupported InsertMode in this stub
#endif
    }
  }

  x->hdr.state++;
  return 0;
}

PetscErrorCode ISCreateGeneral(MPI_Comm comm, PetscInt n, const PetscInt idx[],
                               PetscCopyMode mode, IS *is) {
  // Allocate the IS structure
  struct _p_IS *newis = (struct _p_IS *)malloc(sizeof(struct _p_IS));
  $assert(newis != NULL, "Allocation for newis failed");

  // Set min and max indices
  newis->min = (n > 0) ? idx[0] : -1;
  newis->max = (n > 0) ? idx[n - 1] : -1;
  newis->local_offset = (n > 0) ? idx[0] : 0;

  // Allocate and copy index data based on the mode
  if (mode == PETSC_COPY_VALUES) {
    newis->data = (PetscInt *)malloc(n * sizeof(PetscInt));
    $assert(newis->data != NULL, "Allocation for newis->data failed");
    for (PetscInt i = 0; i < n; i++)
      ((PetscInt *)newis->data)[i] = idx[i];
  } else if (mode == PETSC_OWN_POINTER) {
    newis->data = (PetscInt *)idx; // Take ownership of idx
  } else {
    newis->data = (PetscInt *)idx; // Use the pointer directly
  }

  // Allocate and fill the total array with the same values as the data array
  newis->total = (n > 0) ? (PetscInt *)malloc(n * sizeof(PetscInt)) : NULL;
  if (newis->total) {
    $assert(newis->total != NULL, "Allocation for newis->total failed");
    for (PetscInt i = 0; i < n; i++)
      ((PetscInt *)newis->total)[i] = idx[i];
  }

  // Nonlocal part and complement are unused in this simple case
  newis->nonlocal = NULL;
  newis->complement = NULL;

  // Initialize the info arrays
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < IS_INFO_MAX; j++) {
      newis->info_permanent[i][j] = PETSC_FALSE;
      newis->info[i][j] = IS_INFO_UNKNOWN;
    }
  }

  *is = newis;
  return 0;
}

PetscErrorCode ISCreateStride(MPI_Comm comm, PetscInt n, PetscInt first,
                              PetscInt step, IS *is) {
  /* Allocate the IS structure and assert the allocation succeeded */
  struct _p_IS *newis = (struct _p_IS *)malloc(sizeof(struct _p_IS));
  $assert(newis != NULL, "Allocation for newis failed");

  /* Allocate and initialize the map field in newis */
  /* newis->map = (PetscLayout)malloc(sizeof(*newis->map));
  $assert(newis->map != NULL, "Allocation for newis->map failed");
  MPI_Comm_size(comm, &(newis->map->nproc));
  newis->map->bs = -1;
  newis->map->n = -1;
  newis->map->N = -1;
  newis->map->rstart = 0;
  newis->map->rend = 0; */

  /* Initialize the remaining IS fields */
  newis->min = first;
  newis->max = (n > 0) ? (first + step * (n - 1)) : first;
  newis->local_offset = first;

  /* Allocate and fill in the local index (data) array:
     data[i] = first + i * step, for 0 <= i < n. */
  if (n > 0) {
    newis->data = (PetscInt *)malloc(n * sizeof(PetscInt));
    $assert(newis->data != NULL, "Allocation for newis->data failed");
    for (PetscInt i = 0; i < n; i++)
      ((PetscInt *)newis->data)[i] = first + i * step;
  } else {
    newis->data = NULL;
  }

  /* Allocate and fill the total array with the same values as the data array.
   */
  if (n > 0) {
    newis->total = malloc(n * sizeof(PetscInt));
    $assert(newis->total != NULL, "Allocation for newis->total failed");
    for (PetscInt i = 0; i < n; i++)
      ((PetscInt *)newis->total)[i] = first + i * step;
  } else {
    newis->total = NULL;
  }

  /* For this simple stride IS the nonlocal array is not used */
  newis->nonlocal = NULL;

  /* The complement field is also not used in this simple case */
  newis->complement = NULL;

  /* Initialize the info arrays */
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < IS_INFO_MAX; j++) {
      newis->info_permanent[i][j] = PETSC_FALSE;
      newis->info[i][j] = IS_INFO_UNKNOWN;
    }
  }

  *is = newis;
  return 0;
}

PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[]) {
  $assert(nx >= 1,
          "VecConcatenate: number of input vectors (nx) must be >= 1.");
  $assert(X && Y, "VecConcatenate: vectors cannot be NULL.");

  $vec big = $vec_zero(0);
  for (PetscInt i = 0; i < nx; i++) {
    $vec tmp = petscToCivlVec(X[i]);
    big = $vec_concat(big, tmp);
  }
  Vec newVec =
      civlToPetscVec(big, PETSC_DECIDE, PetscObjectComm((PetscObject)X[0]));
  *Y = newVec;

  if (x_is)
    *x_is = NULL;
  return 0;
}

PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            PetscScalar y[]) {
  $assert(x->read_lock_count == 0,
          "Cannot get values: Vector is locked for reading.");
  $assert(x && x->data, "Vector x and its data must be valid.");
  $assert(ix && y, "Indices and output array must be valid.");
  $assert(ni >= 0, "Number of indices ni must be non-negative.");

  for (PetscInt i = 0; i < ni; i++) {
    PetscInt global_index = ix[i];
    PetscInt local_index = global_index - x->map->rstart;
    $assert(local_index >= 0 && local_index < x->map->n);
    y[i] = x->data[local_index];
  }
  return 0;
}

PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n) {
  memcpy(a, b, n);
  return 0;
}

PetscErrorCode PetscInfo_Private(PetscObject obj, const char message[]) {
  MPI_Comm comm;
  int rank;
  Vec v = (Vec)obj;

  if (obj) {
    comm = v->comm;
    MPI_Comm_rank(comm, &rank);
    if (rank == 0)
      printf("PETSC_INFO: %s", message);
  } else {
    /* If obj is NULL, every rank in MPI_COMM_WORLD prints the message */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("PETSC_INFO (Rank %d): %s", rank, message);
  }
  return 0;
}

PetscBool PetscIsInfReal(PetscReal a) {
  return ((a != 0.0) && (a / 2 == a)) ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool PetscIsNanReal(PetscReal a) {
  return (a != a) ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool PetscIsInfOrNanReal(PetscReal v) {
  return PetscIsInfReal(v) || PetscIsNanReal(v) ? PETSC_TRUE : PETSC_FALSE;
}

int __isfinited(double x) { return isfinite(x); }

PetscErrorCode ISDestroy(IS *is) {
  $assert(is != NULL, "ISDestroy: passed a null pointer for 'is'");
  if (!(*is))
    return 0;

  if ((*is)->total)
    free((*is)->total);
  if ((*is)->nonlocal)
    free((*is)->nonlocal);
  if ((*is)->data)
    free((*is)->data);
  /*   if ((*is)->map)
      free((*is)->map); */
  if ((*is)->complement) {
    ISDestroy(
        &((*is)->complement)); /* recursively destroy complement if needed */
  }

  free(*is);
  *is = NULL;
  return 0;
}

PetscErrorCode VecDestroy(Vec *v) {
  if (v && *v) {
    Vec vec = *v;
    if (vec->data) {
      free(vec->data);
      vec->data = NULL;
    }
    if (vec->hdr.realcomposedstate) {
      free(vec->hdr.realcomposedstate);
      vec->hdr.realcomposedstate = NULL;
    }
    if (vec->hdr.realcomposeddata) {
      free(vec->hdr.realcomposeddata);
      vec->hdr.realcomposeddata = NULL;
    }
    if (vec->map) {
      free(vec->map);
      vec->map = NULL;
    }
    free(vec);
    *v = NULL;
  }
  return 0;
}

PetscErrorCode PetscFinalize(void) {
  int ierr;
  ierr = MPI_Finalize();
  if (ierr != MPI_SUCCESS)
    return 1; // or an appropriate error code
  return 0;
}

PetscReal BLASnrm2_(const PetscBLASInt *n, const PetscScalar *x,
                    const PetscBLASInt *stride) {
  PetscBLASInt in = *n;
  PetscBLASInt istride = *stride;
  double s = 0.0;
  for (PetscBLASInt i = 0; i < in; i += istride)
    s += PetscAbsScalar(x[i]) * PetscAbsScalar(x[i]);
  return sqrt(s);
}

PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x,
                     const PetscBLASInt *sx, const PetscScalar *y,
                     const PetscBLASInt *sy) {
  PetscBLASInt i, ix = 0, iy = 0;
  PetscScalar sum = scalar_make(0.0, 0.0);
  if (*n == 0)
    return sum;
  $assert(!(*n < 0 || *sx <= 0 || *sy <= 0));
  for (i = 0; i < *n; i++) {
    sum = scalar_add(sum, scalar_mul(x[ix], scalar_conj(y[iy])));
    ix += *sx;
    iy += *sy;
  }
  return sum;
}

PetscReal BLASasum_(const PetscBLASInt *n, const PetscScalar *dx,
                    const PetscBLASInt *incx) {
  const PetscBLASInt n_int = *n, incx_int = *incx;
  $assert(incx_int >= 1);
  $assert(n_int >= 0);
  $assert(n_int == 0 || dx != NULL);
  PetscReal sum = 0.0;
  for (PetscBLASInt i = 0, ix = 0; i < n_int; i++) {
    sum += PetscAbsScalar(dx[ix]);
    ix += incx_int;
  }
  return sum;
}

PetscErrorCode VecNorm_Seq(Vec x, NormType type, PetscReal *val) {
  $assert(x != NULL);
  $assert(val != NULL);
  // Convert PETSc vector to CIVL vector
  $vec vec = petscToCivlVec(x);
  switch (type) {
  case NORM_1:
    val[0] = $vec_norm(vec, 1);
    break;
  case NORM_FROBENIUS:
  case NORM_2:
    val[0] = $vec_norm(vec, 2);
    break;
  case NORM_INFINITY:
    val[0] = $vec_norm(vec, $norm_infty);
    break;
  case NORM_1_AND_2:
    val[0] = $vec_norm(vec, 1);
    val[1] = $vec_norm(vec, 2);
    break;
  default:
    $assert(0, "Invalid norm type");
  }
  return 0;
}

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin) {
  int i, n;
  STYPE *x, *y;
  // Check if the vector sizes are compatible
  $assert(xin->map->N == yin->map->N, "Vector sizes does not match.");
  // $assert(xin->data == yin->data,"Vector's are equal");
  n = xin->map->n;
  x = xin->data;
  y = yin->data;
  // Copy the data
  for (i = 0; i < n; i++)
    y[i] = x[i];
  $assert(y != NULL);
}

PetscErrorCode VecConjugate_Seq(Vec xin) {
  int n = xin->map->N;
  for (int i = 0; i < n; i++)
    xin->data[i] = scalar_conj(xin->data[i]);
  return 0;
}

Vec vec_create_seq(int n, PetscScalar *data) {
  Vec vec = (Vec)malloc(sizeof(struct Vec_s));
  vec->map = (SimpleMap)malloc(sizeof(struct map_s));
  vec->map->n = n;
  vec->map->N = n;
  vec->map->bs = 0;
  vec->data = (PetscScalar *)malloc(n * sizeof(PetscScalar));
  if (data != NULL) {
    for (int i = 0; i < n; i++)
#ifdef USE_COMPLEX
      vec->data[i] = $make_complex(data[i].real, data[i].imag);
#else
      vec->data[i] = data[i];
#endif
  } else {
    for (int i = 0; i < n; i++)
      vec->data[i] = scalar_make(0.0, 0.0);
  }
  return vec;
}

void vecprint_seq(const char *name, Vec vin) {
  int N = vin->map->N;
  STYPE *data = vin->data;
  $vec c_v = $vec_make_from_dense(N, data);
  $print("\n", name, ": ");
  $vec_print(c_v);
}

void vec_destroy_seq(Vec vec) {
  free(vec->data);
  free(vec->hdr.realcomposedstate);
  free(vec->hdr.realcomposeddata);
  free(vec->map);
  free(vec);
}

void vecprint_mpi(const char *name, Vec vin) {
  int rank;
  MPI_Comm_rank(vin->comm, &rank);
  $vec civl_vec = petscToCivlVec(vin);
  if (rank == 0) {
    $print("\n", name, ": ");
    $vec_print(civl_vec);
  }
}

$vec petscToCivlVec(Vec petscVec) {
  int N = petscVec->map->N;
  PetscScalar *data = petscVec->data;
  if (petscVec->type == VECSEQ) {
    return $vec_make_from_dense(N, data);
  } else {
    MPI_Comm comm = petscVec->comm;
    int rank, nproc, n = petscVec->map->n;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    // Allocate recvcounts and displs only on rank 0
    int *recvcounts = rank == 0 ? (int *)malloc(nproc * sizeof(int)) : NULL;
    int *displs = rank == 0 ? (int *)malloc(nproc * sizeof(int)) : NULL;
    // Gather local sizes
    MPI_Gather(&n, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, comm);
    // Calculate displacements on rank 0
    if (rank == 0) {
      displs[0] = 0;
      for (int i = 1; i < nproc; i++)
        displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
#ifndef USE_COMPLEX
    PetscReal localData[n]; /* create a small local array on the stack */
    for (int i = 0; i < n; i++)
      localData[i] = data[i]; /* copy the local portion into localData */

    // Real case: Gather real data on rank 0
    PetscReal *globalData =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    MPI_Gatherv(localData, n, PETSC_REAL, globalData, recvcounts, displs,
                PETSC_REAL, 0, comm);
#else
    // Complex case: Gather real and imaginary parts separately
    PetscReal localReals[n], localImags[n];
    for (int i = 0; i < n; i++) {
      localReals[i] = data[i].real;
      localImags[i] = data[i].imag;
    }
    PetscReal *globalReals =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    PetscReal *globalImags =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    MPI_Gatherv(localReals, n, PETSC_REAL, globalReals, recvcounts, displs,
                PETSC_REAL, 0, comm);
    MPI_Gatherv(localImags, n, PETSC_REAL, globalImags, recvcounts, displs,
                PETSC_REAL, 0, comm);
    // Create complex data on rank 0
    STYPE *globalData = rank == 0 ? (STYPE *)malloc(N * sizeof(STYPE)) : NULL;
    if (rank == 0)
      for (int i = 0; i < N; i++)
        globalData[i] = (STYPE){globalReals[i], globalImags[i]};
#endif
    $vec result =
        rank == 0 ? $vec_make_from_dense(N, globalData) : $vec_zero(0);
    if (rank == 0) {
      free(recvcounts);
      free(displs);
#ifndef USE_COMPLEX
      free(globalData);
#else
      free(globalReals);
      free(globalImags);
      free(globalData);
#endif
    }
    return result;
  }
}

void civlToPetscVecCopy($vec in, Vec out) {
  Vec z = civlToPetscVec(in, out->map->n, out->comm);
  VecCopy(z, out);
  VecDestroy(&z);
}

/*	Converts the CIVL abstract vector on process 0 to PETSc distributed
   vector. n is the local_size of the new vector on this proc, use
   PETSC_DECIDE to use the standard distribution. On processess of non-zero
   rank, the 'in' aurgment is ignored.
*/
Vec civlToPetscVec($vec in, int n, MPI_Comm comm) {
  Vec out; // PETSc vector to be returned
  int N, rank;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    N = in.len; // Length of the input CIVL vector
  MPI_Bcast(&N, 1, MPI_INT, 0, comm);
  VecCreate(comm, &out);
  VecSetSizes(out, n, N);
  VecSetFromOptions(out);
  int nproc = out->nproc;
  n = out->map->n;
  MPI_Comm_rank(comm, &rank);
  if (out->type == VECSEQ) {
    out->comm = MPI_COMM_SELF;
    $assert(n == N || n == PETSC_DECIDE);
    for (int i = 0; i < N; i++)
      out->data[i] = in.data[i];
  } else {
    int *sendcounts = rank == 0 ? (int *)malloc(nproc * sizeof(int)) : NULL;
    int *displs = rank == 0 ? (int *)malloc(nproc * sizeof(int)) : NULL;
    MPI_Gather(&n, 1, MPI_INT, sendcounts, 1, MPI_INT, 0, comm);
    if (rank == 0) {
      displs[0] = 0;
      for (int i = 1; i < nproc; i++)
        displs[i] = displs[i - 1] + sendcounts[i - 1];
    }
#ifdef USE_COMPLEX
    PetscReal *sbuf_reals =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    PetscReal *sbuf_imags =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    if (rank == 0) {
      for (int i = 0; i < N; i++) {
        sbuf_reals[i] = in.data[i].real;
        sbuf_imags[i] = in.data[i].imag;
      }
    }
    PetscReal rbuf_reals[n], rbuf_imags[n];
    MPI_Scatterv(sbuf_reals, sendcounts, displs, PETSC_REAL, rbuf_reals, n,
                 PETSC_REAL, 0, comm);
    MPI_Scatterv(sbuf_imags, sendcounts, displs, PETSC_REAL, rbuf_imags, n,
                 PETSC_REAL, 0, comm);
    for (int i = 0; i < n; i++)
      out->data[i] = $make_complex(rbuf_reals[i], rbuf_imags[i]);
    if (rank == 0) {
      free(sbuf_reals);
      free(sbuf_imags);
    }
#else
    PetscReal *sendbuf =
        rank == 0 ? (PetscReal *)malloc(N * sizeof(PetscReal)) : NULL;
    if (rank == 0)
      for (int i = 0; i < N; i++)
        sendbuf[i] = in.data[i];
    PetscReal rbuf[n];
    MPI_Scatterv(sendbuf, sendcounts, displs, PETSC_REAL, rbuf, n, PETSC_REAL,
                 0, comm);
    for (int i = 0; i < n; i++)
      out->data[i] = rbuf[i];
    if (rank == 0)
      free(sendbuf);
#endif
    if (rank == 0) {
      free(sendcounts);
      free(displs);
    }
  }
  return out;
}

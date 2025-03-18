#include "petscvec.h"
#include <math.h>
#include <mpi.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

#define PETSC_FLOPS_PER_OP 1.0

PetscLogDouble petsc_TotalFlops = 0.0;

PetscLogDouble petsc_TotalFlops_th = 0.0;

PETSC_EXTERN PetscLogDouble petsc_TotalFlops;

PETSC_EXTERN_TLS PetscLogDouble petsc_TotalFlops_th;

int NormIds[] = {0, 1, 2, 3, 4};

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
  return 0;
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

PetscErrorCode VecGetArrayWrite(Vec x, PetscScalar **a) {
  return VecGetArray(x, a);
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

PetscErrorCode VecRestoreArrayWrite(Vec x, PetscScalar **a) {
  return VecRestoreArray(x, a);
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
    int nlocal = *N / size + (rank < *N % size);
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
  int rank;
  MPI_Comm_rank(v->comm, &rank);
  MPI_Comm_size(v->comm, &v->nproc);
  PetscSplitOwnership(v->comm, &n, &N);
  v->map->nproc = v->nproc;
  v->map->n = n;
  v->map->N = N;
  int start_value = 0;
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
  $assert(is && is->data, "VecGetSubVector: Index set must be valid.");
  // Build a local $vec from X->data using the local size.
  $vec civl_X = $vec_make_from_dense(X->map->n, X->data);
  // Since the PETSc Vec’s data is local, we use local indexing:
  first = 0;
  n = X->map->n;
  $vec civl_Y = $vec_subseq(civl_X, first, n);
  Vec newVec = civlToPetscVec(civl_Y, PETSC_DECIDE, X->comm);
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
  if (vec->nproc > 1)
    VecSetType(vec, VECMPI);
  else
    VecSetType(vec, VECSEQ);
  return 0;
}

PetscErrorCode VecSetType(Vec vec, VecType newType) {
  // Check if the new type is the same as the current type
  if (vec->type == newType)
    return 0;
  MPI_Comm_size(vec->comm, &vec->nproc);
  int nproc = vec->nproc;
  // Update the type and communicator based on the new type
  switch (newType) {
  case VECSEQ:
    vec->type = VECSEQ;
    vec->comm = PETSC_COMM_SELF;
    SetOps_Seq(vec);
    break;
  case VECMPI:
    // Parallel vector, requires more than one process and PETSC_COMM_WORLD
    vec->type = VECMPI;
    vec->comm = PETSC_COMM_WORLD;
    SetOps_MPI(vec);
    break;
  case VECSTANDARD:
    // Standard vector type, decide based on the number of processes
    if (vec->type == VECSEQ) {
      vec->type = VECMPI;
      vec->comm = PETSC_COMM_WORLD;
      SetOps_Seq(vec);
    } else if (vec->type == VECMPI) {
      vec->type = VECSEQ;
      vec->comm = PETSC_COMM_SELF;
      SetOps_MPI(vec);
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

PetscErrorCode VecSet_Seq(Vec x, PetscScalar alpha) {
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  for (int i = 0; i < x->map->n; i++)
    x->data[i] = alpha;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecSet(Vec x, PetscScalar alpha) { return VecSet_Seq(x, alpha); }

PetscErrorCode VecView(Vec vec, PetscViewer viewer) {
  $vec civl_vec = petscToCivlVec(vec);
  $vec_print(civl_vec);
  vec = civlToPetscVec(civl_vec, PETSC_DECIDE, vec->comm);
  return 0;
}

PetscErrorCode VecDot_Seq(Vec x, Vec y, PetscScalar *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecDot_Seq called\n");
#endif
  $vec v1 = $vec_make_from_dense(x->map->n, x->data),
       v2 = $vec_make_from_dense(y->map->n, y->data);
  *val = $vec_dot(v1, v2);
  return 0;
}

PetscErrorCode VecDot_MPI(Vec x, Vec y, PetscScalar *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecDot_MPI called\n");
#endif
  $vec v1 = petscToCivlVec(x), v2 = petscToCivlVec(y);
  int rank;
  MPI_Comm_rank(x->comm, &rank);
  if (rank == 0)
    *val = $vec_dot(v1, v2);
#ifdef USE_COMPLEX
  MPI_Bcast(&val->real, 1, MPIU_REAL, 0, x->comm);
  MPI_Bcast(&val->imag, 1, MPIU_REAL, 0, x->comm);
#else
  MPI_Bcast(val, 1, MPIU_REAL, 0, x->comm);
#endif
  return 0;
}

PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecDot called\n");
#endif
  $assert(x->type == y->type);
  switch (x->type) {
  case VECSEQ:
    return VecDot_Seq(x, y, val);
  case VECMPI:
    return VecDot_MPI(x, y, val);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecDot_MPI(x, y, val) : VecDot_Seq(x, y, val);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode VecTDot_Seq(Vec x, Vec y, PetscScalar *val) {
  int n = x->map->n;
  PetscScalar sum = scalar_zero;
  for (PetscInt i = 0; i < n; i++) {
    // For transpose dot product, do not conjugate y->data[i]
    sum = scalar_add(sum, scalar_mul(x->data[i], y->data[i]));
  }
  *val = sum;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecTDot_MPI(Vec x, Vec y, PetscScalar *val) {
  $assert(x != NULL);
  $assert(y != NULL);
  $assert(val != NULL);

  PetscScalar local_tdot = scalar_make(0.0, 0.0);
  VecTDot_Seq(x, y, &local_tdot);

#ifdef USE_COMPLEX
  double in_arr[2], out_arr[2];
  in_arr[0] = local_tdot.real;
  in_arr[1] = local_tdot.imag;
  // Reduce the two components (sum across processes)
  MPIU_Allreduce(in_arr, out_arr, 2, MPI_DOUBLE, MPIU_SUM, x->comm);
  local_tdot = scalar_make(out_arr[0], out_arr[1]);
#else
  // For real numbers, reduce a single scalar
  MPI_Allreduce(&local_tdot, &local_tdot, 1, MPI_DOUBLE, MPI_SUM, x->comm);
#endif
  *val = (PetscScalar)local_tdot;
  return 0;
}

PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val) {
  $assert(x->type == y->type);

  switch (x->type) {
  case VECSEQ:
    return VecTDot_Seq(x, y, val);
  case VECMPI:
    return VecTDot_MPI(x, y, val);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecTDot_MPI(x, y, val) : VecTDot_Seq(x, y, val);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode VecMTDot_Seq(Vec x, PetscInt nv, const Vec y[],
                            PetscScalar val[]) {
  int n = x->map->n; /* local size */
  for (int j = 0; j < nv; j++) {
    PetscScalar sum = scalar_zero;
    /* Compute the dot product between x and y[j] over the local elements */
    for (int i = 0; i < n; i++) {
      // For TDot (indefinite dot product), we do not apply complex conjugation
      sum = scalar_add(sum, scalar_mul(x->data[i], y[j]->data[i]));
    }
    val[j] = sum;
  }
  return 0;
}

PetscErrorCode VecMTDot_MPI(Vec x, PetscInt nv, const Vec y[],
                            PetscScalar val[]) {
  $assert(x != NULL);
  $assert(y != NULL);
  $assert(val != NULL);
  $assert(nv >= 0, "Number of vectors (nv) must be non-negative.");
  for (int i = 0; i < nv; i++) {
    $assert(x->map->N == y[i]->map->N,
            "VecMTDot_MPI: All vectors must have the same size.");
  }
  PetscScalar local_tdot[nv];
  VecMTDot_Seq(x, nv, y, local_tdot);

#ifdef USE_COMPLEX
  double in_real[nv], in_imag[nv], out_real[nv], out_imag[nv];
  for (int j = 0; j < nv; j++) {
    in_real[j] = local_tdot[j].real;
    in_imag[j] = local_tdot[j].imag;
  }
  MPI_Allreduce(in_real, out_real, nv, MPI_DOUBLE, MPI_SUM, x->comm);
  MPI_Allreduce(in_imag, out_imag, nv, MPI_DOUBLE, MPI_SUM, x->comm);
  for (int j = 0; j < nv; j++)
    val[j] = scalar_make(out_real[j], out_imag[j]);
#else
  MPI_Allreduce(local_tdot, val, nv, MPI_DOUBLE, MPI_SUM, x->comm);
#endif
  return 0;
}

PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  for (int i = 0; i < nv; i++)
    $assert(x->type == y[i]->type, "VecMTDot: Vector types must match.");

  switch (x->type) {
  case VECSEQ:
    return VecMTDot_Seq(x, nv, y, val);
  case VECMPI:
    return VecMTDot_MPI(x, nv, y, val);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecMTDot_MPI(x, nv, y, val)
                          : VecMTDot_Seq(x, nv, y, val);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecDotRealPart called\n");
#endif
  PetscScalar dotProduct;
  VecDot(x, y, &dotProduct);
  *val = PetscRealPart(dotProduct);
  return 0;
}

PetscErrorCode VecNormalize(Vec x, PetscReal *val) {
  VecNorm(x, NORM_2, val);
  $assert(*val > 0.0);
  PetscScalar a = scalar_of(1.0 / *val);
  int n = x->map->n;
  for (int i = 0; i < n; i++)
    x->data[i] = scalar_mul(x->data[i], a);
}

PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]) {
  for (int j = 0; j < nv; j++) {
    $assert(x->map->n == y[j]->map->n);
    PetscScalar local_sum = scalar_make(0.0, 0.0);
    for (int i = 0; i < x->map->n; i++) {
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

PetscErrorCode VecCopy_Seq(Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Spec VecCopy_Seq called\n");
#endif
  $assert(y->read_lock_count == 0,
          "Cannot Copy values: Vector is locked for reading.");
  $assert(x, "Vector cannot be null");
  int n = x->map->n;
  $assert(n == y->map->n, "Vector length mismatch");
  if (x != y)
    for (int i = 0; i < n; i++)
      y->data[i] = x->data[i];
  return 0;
}

PetscErrorCode VecCopy(Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Spec VecCopy called\n");
#endif
  return VecCopy_Seq(x, y);
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

PetscErrorCode VecMax_Seq(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMax_Seq\n");
#endif
  int n = x->map->n;
  int local_index = -1;
  PetscReal local_max = PETSC_MIN_REAL;
  if (n > 0) {
    local_index = 0;
    local_max = PetscRealPart(x->data[local_index]);
    for (int i = 1; i < n; ++i) {
      PetscReal current_value = PetscRealPart(x->data[i]);
      if (current_value > local_max) {
        local_max = current_value;
        local_index = i;
      }
    }
  }
  *val = local_max;
  if (p)
    *p = local_index;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecMax_MPI(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMax_MPI\n");
#endif
  $vec vec = petscToCivlVec(x);
  int rank, idx = -1;
  PetscReal max = PETSC_MAX_REAL;
  MPI_Comm_rank(x->comm, &rank);
  if (rank == 0) {
    int N = vec.len;
    if (N != 0) {
      idx = 0;
      max = PetscRealPart(vec.data[0]);
      for (int i = 1; i < N; i++) {
        PetscReal a = PetscRealPart(vec.data[i]);
        if (a > max) {
          max = a;
          idx = i;
        }
      }
    }
  }
  MPI_Bcast(&max, 1, MPIU_REAL, 0, x->comm);
  MPI_Bcast(&idx, 1, MPI_INT, 0, x->comm);
  if (p)
    *p = idx;
  *val = max;
  return 0;
}

PetscErrorCode VecMax_MPI_alt(Vec x, PetscInt *p, PetscReal *val) {
  $assert(x != NULL);
  $assert(val != NULL);
  // Compute local max and index using VecMax_Seq
  VecMax_Seq(x, p, val);
  if (p) {
    PetscReal local_max = *val;
    PetscReal global_max;
    int global_index;
    // Find global maximum value
    MPI_Allreduce(&local_max, &global_max, 1, MPIU_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
    // Adjust local index to global using rstart if it is the global max
    int local_idx =
        (local_max == global_max) ? *p + x->map->rstart : PETSC_MAX_INT;
    // Find the smallest global index among processes with the global max
    MPI_Allreduce(&local_idx, &global_index, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    *val = global_max;
    *p = (global_index != PETSC_MAX_INT) ? global_index : -1;
  } else {
    // Only reduce the value if index is not needed
    MPI_Allreduce(val, val, 1, MPIU_REAL, MPI_MAX, MPI_COMM_WORLD);
  }
  return 0;
}

/*
  Returns the value PETSC_MIN_REAL and negative p if the vector is of
  length 0.

  Returns the smallest index with the maximum value.
*/
PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMax\n");
#endif
  switch (x->type) {
  case VECSEQ:
    return VecMax_Seq(x, p, val);
  case VECMPI:
    return VecMax_MPI(x, p, val);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecMax_MPI(x, p, val) : VecMax_Seq(x, p, val);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode VecMin_Seq(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMin_Seq\n");
#endif
  $assert(x != NULL);
  $assert(val != NULL);
  int n = x->map->n;
  int local_index = -1;
  PetscReal local_max = PETSC_MAX_REAL;
  if (n > 0) {
    local_index = 0;
    local_max = PetscRealPart(x->data[local_index]);
    for (int i = 1; i < n; ++i) {
      PetscReal current_value = PetscRealPart(x->data[i]);
      if (current_value < local_max) {
        local_max = current_value;
        local_index = i;
      }
    }
  }
  *val = local_max;
  if (p)
    *p = local_index;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecMin_MPI(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMin_MPI\n");
#endif
  $assert(x != NULL);
  $assert(val != NULL);
  // Compute local max and index using VecMax_Seq
  VecMin_Seq(x, p, val);
  if (p) {
    PetscReal local_max = *val;
    PetscReal global_max;
    int global_index;
    // Find global maximum value
    MPI_Allreduce(&local_max, &global_max, 1, MPIU_REAL, MPI_MIN,
                  MPI_COMM_WORLD);
    // Adjust local index to global using rstart if it is the global max
    int local_idx =
        (local_max == global_max) ? *p + x->map->rstart : PETSC_MAX_INT;
    // Find the smallest global index among processes with the global max
    MPI_Allreduce(&local_idx, &global_index, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    *val = global_max;
    *p = (global_index != PETSC_MAX_INT) ? global_index : -1;
  } else {
    // Only reduce the value if index is not needed
    MPI_Allreduce(val, val, 1, MPIU_REAL, MPI_MIN, MPI_COMM_WORLD);
  }
  return 0;
}

PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMin\n");
#endif
  $assert(x != NULL);
  switch (x->type) {
  case VECSEQ:
    return VecMin_Seq(x, p, val);
  case VECMPI:
    return VecMin_MPI(x, p, val);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecMin_MPI(x, p, val) : VecMin_Seq(x, p, val);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode PetscLogFlops(PetscLogDouble n) {
  $assert(n >= 0);
  return PetscAddLogDouble(&petsc_TotalFlops, &petsc_TotalFlops_th,
                           PETSC_FLOPS_PER_OP * n);
}

PetscErrorCode VecScale_Seq(Vec x, PetscScalar alpha) {
  $assert(x->read_lock_count == 0,
          "Cannot scale values: Vector is locked for reading.");
  int n = x->map->n;
  for (int i = 0; i < n; i++)
    x->data[i] = scalar_mul(alpha, x->data[i]);
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
#ifdef DEBUG
  $print("DEBUG: Spec VecScale called\n");
#endif
  return VecScale_Seq(x, alpha);
}

PetscErrorCode VecMAXPY_Seq(Vec y, PetscInt nv, const PetscScalar alpha[],
                            Vec x[]) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMAXPY_Seq called\n");
#endif
  int n = y->map->n;
  for (int i = 0; i < n; i++) {
    PetscScalar sum = scalar_zero;
    for (int j = 0; j < nv; j++)
      sum = scalar_add(sum, scalar_mul(alpha[j], x[j]->data[i]));
    y->data[i] = scalar_add(y->data[i], sum);
  }
  y->hdr.state++;
  return 0;
}

PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[],
                        Vec x[]) {
#ifdef DEBUG
  $print("DEBUG: Spec VecMAXPY called\n");
#endif
  return VecMAXPY_Seq(y, nv, alpha, x);
}

PetscErrorCode VecAXPY_Seq(Vec y, PetscScalar alpha, Vec x) {
  $assert(y != NULL);
  $assert(x != NULL);
  $assert(y->map->n == x->map->n, "Local sizes of x and y must match.");
  PetscInt n = x->map->n;
  for (PetscInt i = 0; i < n; i++)
    y->data[i] = scalar_add(y->data[i], scalar_mul(alpha, x->data[i]));
  y->hdr.state++;
  return 0;
}

PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x) {
#ifdef DEBUG
  $print("DEBUG: Spec VecAXPY called\n");
#endif
  return VecAXPY_Seq(y, alpha, x);
}

PetscErrorCode VecAXPBY_Seq(Vec y, PetscScalar alpha, PetscScalar beta, Vec x) {
#ifdef DEBUG
  $print("DEBUG: Spec VecAXPBY_Seq called\n");
#endif
  $assert(y != NULL);
  $assert(x != NULL);
  int n = y->map->n;
  for (int i = 0; i < n; i++) {
#ifdef USE_COMPLEX
    y->data[i] =
        scalar_add(scalar_mul(alpha, x->data[i]), scalar_mul(beta, y->data[i]));
#else
    y->data[i] = alpha * x->data[i] + beta * y->data[i];
#endif
  }
  y->hdr.state++;
  return 0;
}

PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x) {
#ifdef DEBUG
  $print("DEBUG: Spec VecAXPBY called\n");
#endif
  return VecAXPBY_Seq(y, alpha, beta, x);
}

PetscErrorCode VecAXPBYPCZ_Seq(Vec z, PetscScalar alpha, PetscScalar beta,
                               PetscScalar gamma, Vec x, Vec y) {
  PetscInt n = z->map->n;
  for (PetscInt i = 0; i < n; i++) {
    z->data[i] = scalar_add(
        scalar_add(scalar_mul(alpha, x->data[i]), scalar_mul(beta, y->data[i])),
        scalar_mul(gamma, z->data[i]));
  }
  z->hdr.state++;
  return 0;
}

PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Spec VecAXPBYPCZ called\n");
#endif
  return VecAXPBYPCZ_Seq(z, alpha, beta, gamma, x, y);
}

PetscErrorCode VecMAXPBY(Vec y, PetscInt nv, const PetscScalar alpha[],
                         PetscScalar beta, Vec x[]) {
#ifdef DEBUG
  $print("DEBUG: spec VecMAXPBY called\n");
#endif
  $assert(y->read_lock_count == 0,
          "Cannot MAXPBY values: Vector is locked for reading.");
  $assert(nv >= 0, "Number of vectors cannot be negative");
  if (nv > 0)
    $assert(y && x && alpha, "Input vectors and scalars must not be NULL.");

  // Vector compatibility checks
  for (int v = 0; v < nv; v++) {
    $assert(y->map->n == x[v]->map->n,
            "VecMAXPBY_spec: All vectors must have the same size.");
    $assert(y != x[v], "Input vectors cannot contain y itself");
  }
  $vec c_y = petscToCivlVec(y);
  if (scalar_eq(scalar_of(0.0), beta)) {
    $vec c_z = $vec_zero(c_y.len);
    for (int i = 0; i < c_y.len; i++)
      $vec_set(c_z, i, scalar_of(0.0));
    civlToPetscVecCopy(c_z, y);
  } else {
    civlToPetscVecCopy($vec_scalar_mul(beta, c_y), y);
  }
  VecMAXPY(y, nv, alpha, x);
  return 0;
}

PetscErrorCode VecSwap(Vec x, Vec y) {
  $assert(x->read_lock_count == 0 && y->read_lock_count == 0,
          "Cannot swap values: Vector is locked for reading.");
  $assert(x->map->n == y->map->n);
  for (int i = 0; i < x->map->n; i++) {
    PetscScalar temp = x->data[i];
    x->data[i] = y->data[i];
    y->data[i] = temp;
  }
  $assert(x != NULL && y != NULL);
  return 0;
}

PetscErrorCode VecWAXPY_Seq(Vec w, PetscScalar alpha, Vec x, Vec y) {
  int n = x->map->n;
  $assert(n == y->map->n, "Vectors x and y must have the same local size");
  $assert(n == w->map->n, "Vectors w and x must have the same local size");
  for (int i = 0; i < n; i++)
    w->data[i] = scalar_add(y->data[i], scalar_mul(alpha, x->data[i]));
  w->hdr.state++;
  return 0;
}

PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Spec VecWAXPY called\n");
#endif
  if (scalar_eq(scalar_of(0), alpha))
    return VecCopy(y, w);
  else
    return VecWAXPY_Seq(w, alpha, x, y);
}

PetscErrorCode VecAYPX_Seq(Vec y, PetscScalar beta, Vec x) {
  PetscInt n = y->map->n;
  $assert(n == x->map->n, "Local size mismatch between vectors");
  for (PetscInt i = 0; i < n; i++)
    y->data[i] = scalar_add(x->data[i], scalar_mul(beta, y->data[i]));
  y->hdr.state++;
  return 0;
}

PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x) {
#ifdef DEBUG
  $print("DEBUG: Spec VecAYPX called\n");
#endif
  return VecAYPX_Seq(y, beta, x);
}

PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y) {
  $assert(w && x && y);
  $assert(w->map->n == x->map->n && w->map->n == y->map->n);
  for (int i = 0; i < w->map->n; i++)
    w->data[i] = scalar_mul(x->data[i], y->data[i]);
  return 0;
}

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec x, Vec y, PetscReal *max) {
  $assert(x != NULL);
  $assert(y != NULL);
  $assert(max != NULL);
  int n = x->map->n;
  PetscReal m = 0.0;

  for (int i = 0; i < n; i++) {
    // Check if y->data[i] is zero to avoid division by zero.
    PetscReal v = scalar_eq(y->data[i], scalar_zero)
                      ? scalar_abs(x->data[i])
                      : scalar_abs(scalar_div(x->data[i], y->data[i]));
    m = max(m, v);
  }
  *max = m;
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecMaxPointwiseDivide(Vec xin, Vec yin, PetscReal *max) {
  return VecMaxPointwiseDivide_Seq(xin, yin, max);
}

PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y) {
  $assert(w && x && y, "Vectors cannot be NULL.");
  $assert(w->map->n == x->map->n && w->map->n == y->map->n,
          "Vectors Local sizes mismatch.");

  // Compute w[i] = x[i] / y[i] component-wise
  for (int i = 0; i < w->map->n; i++) {
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
  int local_size = v->map->n;
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
  for (int i = 0; i < m; i++) {
    PetscErrorCode ierr = VecDuplicate(v, &(*V)[i]);
    if (ierr != 0) {
      // Handle error, free allocated memory
      for (int j = 0; j < i; j++)
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
  for (int i = 0; i < m; i++) {
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
  int id = NormIds[type];           // Map NormType to identifier
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

char *$petsc_norm_name(NormType type) {
  switch (type) {
  case NORM_1:
    return "NORM_1";
  case NORM_2:
    return "NORM_2";
  case NORM_FROBENIUS:
    return "NORM_FROBENIUS";
  case NORM_INFINITY:
    return "NORM_INFINITY";
  case NORM_1_AND_2:
    return "NORM_1_AND_2";
  default:
    $assert(0);
  }
  return NULL;
}

void $petsc_norm($vec vec, NormType type, PetscReal *result) {
  switch (type) {
  case NORM_1:
    result[0] = $vec_norm(vec, 1);
    break;
  case NORM_FROBENIUS:
  case NORM_2:
    result[0] = $vec_norm(vec, 2);
    break;
  case NORM_INFINITY:
    result[0] = $vec_norm(vec, $norm_infty);
    break;
  case NORM_1_AND_2:
    result[0] = $vec_norm(vec, 1);
    result[1] = $vec_norm(vec, 2);
    break;
  default:
    $assert(0, "Invalid norm type");
  }
}

PetscErrorCode VecNorm_Seq(Vec x, NormType type, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecNorm_Seq\n");
#endif
  $assert(x != NULL);
  $assert(val != NULL);
  int n = x->map->n;
  $vec vec_loc = n == 0 ? $vec_zero(0) : $vec_make_from_dense(n, x->data);
  $petsc_norm(vec_loc, type, val);
  x->hdr.state++;
  //  Store the computed norm
  if (type != NORM_1_AND_2) {
    int id = NormIds[(int)type];
    if (id >= 0 && id < NUM_NORM_TYPES) {
      x->hdr.realcomposeddata[id] = val[0];
      x->hdr.realcomposedstate[id] = x->hdr.state;
    }
  }
  return 0;
}

PetscErrorCode VecNorm_MPI(Vec x, NormType type, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecNorm_MPI\n");
#endif
  int rank;
  MPI_Comm_rank(x->comm, &rank);
  $vec vec = petscToCivlVec(x);
  if (rank == 0)
    $petsc_norm(vec, type, val);
  MPI_Bcast(val, (type == NORM_1_AND_2 ? 2 : 1), MPIU_REAL, 0, x->comm);
  if (type != NORM_1_AND_2) {
    int id = NormIds[(int)type];
    if (id >= 0 && id < NUM_NORM_TYPES) {
      x->hdr.realcomposeddata[id] = val[0];
      x->hdr.realcomposedstate[id] = x->hdr.state;
    }
  }
  return 0;
}

/* Alternative spec of VecNorm_MPI */
PetscErrorCode VecNorm_MPI_alt(Vec x, NormType type, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecNorm_MPI\n");
#endif
  PetscReal local_val[2] = {0.0, 0.0}, global_val[2] = {0.0, 0.0};
  int size;
  MPI_Comm_size(x->comm, &size);
  // Compute local norm using VecNorm_Seq
  VecNorm_Seq(x, type, local_val);

  switch (type) {
  case NORM_1:
    // Sum the absolute values across all processes
    MPI_Allreduce(&local_val[0], &global_val[0], 1, MPIU_REAL, MPI_SUM,
                  x->comm);
    *val = global_val[0];
    break;
  case NORM_FROBENIUS:
  case NORM_2:
    local_val[0] *= local_val[0];
    // Sum the squares of local norms and take the square root
    MPI_Allreduce(&local_val[0], &global_val[0], 1, MPIU_REAL, MPI_SUM,
                  x->comm);
    *val = PetscSqrtReal(global_val[0]);
    break;
  case NORM_INFINITY:
    // Compute max norm across all processes
    MPI_Allreduce(&local_val[0], &global_val[0], 1, MPIU_REAL, MPI_MAX,
                  x->comm);
    *val = global_val[0];
    break;
  case NORM_1_AND_2:
    local_val[1] *= local_val[1];
    // Compute both 1-norm and 2-norm reductions
    MPI_Allreduce(local_val, global_val, 2, MPIU_REAL, MPI_SUM, x->comm);
    *val = global_val[0];                      // 1-norm
    *(val + 1) = PetscSqrtReal(global_val[1]); // 2-norm
    break;
  default:
    $assert(0, "Invalid norm type");
  }
  return 0;
}

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
#ifdef DEBUG
  $print("DEBUG: Spec VecNorm\n");
#endif
  switch (x->type) {
  case VECSEQ:
    VecNorm_Seq(x, type, val);
    break;
  case VECMPI:
    VecNorm_MPI(x, type, val);
    break;
  case VECSTANDARD:
    (x->nproc > 1) ? VecNorm_MPI(x, type, val) : VecNorm_Seq(x, type, val);
    break;
  default:
    $assert(0, "Invalid norm type");
  }
  return 0;
}

PetscErrorCode VecStrideNorm(Vec x, PetscInt start, NormType ntype,
                             PetscReal *val) {
  int n, stride;
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
    for (int i = start; i < n; i += stride)
      local_val[0] += PetscAbsScalar(x->data[i]);
    MPI_Allreduce(local_val, val, 1, PETSC_REAL, MPI_SUM, PETSC_COMM_WORLD);
    break;

  case NORM_FROBENIUS:
  case NORM_2:
    for (int i = start; i < n; i += stride) {
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
    for (int i = start; i < n; i += stride) {
#ifdef USE_COMPLEX
      local_val[0] = fmax(local_val[0], $cabs(x->data[i]));
#else
      local_val[0] = fmax(local_val[0], fabs(x->data[i]));
#endif
    }
    MPI_Allreduce(local_val, val, 1, PETSC_REAL, MPI_MAX, PETSC_COMM_WORLD);
    break;

  case NORM_1_AND_2:
    for (int i = start; i < n; i += stride) {
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
  int local_row = row - v->map->rstart;
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

PetscErrorCode VecSetValues_Seq(Vec x, PetscInt ni, const PetscInt ix[],
                                const PetscScalar y[], InsertMode iora) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  $assert(x && x->data);
  $assert(ix && y);
  $assert(ni >= 0);

  /* Only process 0 performs the update on its local copy of the global vector
   */
  if (rank == 0) {
    for (int i = 0; i < ni; i++) {
      int local_index = ix[i] - x->map->rstart;
      if (local_index < 0 || local_index >= x->map->n)
        continue; // Skip indices not in the local range
#ifdef USE_COMPLEX
      if (iora == INSERT_VALUES) {
        x->data[local_index].real = y[i].real;
        x->data[local_index].imag = y[i].imag;
      } else if (iora == ADD_VALUES) {
        x->data[local_index].real += y[i].real;
        x->data[local_index].imag += y[i].imag;
      } else {
        return 1; // Unsupported InsertMode
      }
#else
      if (iora == INSERT_VALUES)
        x->data[local_index] = y[i];
      else if (iora == ADD_VALUES)
        x->data[local_index] += y[i];
      else
        return 1;
#endif
    }
    x->hdr.state++;
  }
  return 0;
}

PetscErrorCode VecSetValues_MPI(Vec x, PetscInt ni, const PetscInt ix[],
                                const PetscScalar y[], InsertMode iora) {
  PetscErrorCode ierr;
  int rank, size;
  MPI_Comm comm = x->comm;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int N = x->map->N;
  int localSize = x->map->n;

  /* First, get the local sizes from all processes */
  PetscInt *recvcounts = (PetscInt *)malloc(size * sizeof(PetscInt));
  PetscInt *displs = (PetscInt *)malloc(size * sizeof(PetscInt));
  MPI_Allgather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
  displs[0] = 0;
  for (int i = 1; i < size; i++)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

#ifdef USE_COMPLEX
  /* Allocate buffers for the real and imaginary parts on each process */
  double *localReal = (double *)malloc(localSize * sizeof(double));
  double *localImag = (double *)malloc(localSize * sizeof(double));
  for (int i = 0; i < localSize; i++) {
    localReal[i] = x->data[i].real;
    localImag[i] = x->data[i].imag;
  }
  double *globalReal = NULL, *globalImag = NULL;
  if (rank == 0) {
    globalReal = (double *)malloc(N * sizeof(double));
    globalImag = (double *)malloc(N * sizeof(double));
  }
  /* Gather the real parts and imaginary parts from all processes */
  MPI_Gatherv(localReal, localSize, MPI_DOUBLE, globalReal, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  MPI_Gatherv(localImag, localSize, MPI_DOUBLE, globalImag, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  free(localReal);
  free(localImag);

  // On process 0, update the gathered global arrays according to indices ix[]
  if (rank == 0) {
    for (int i = 0; i < ni; i++) {
      PetscInt global_index = ix[i];
      if (global_index < 0 || global_index >= N)
        continue;
      if (iora == INSERT_VALUES) {
        globalReal[global_index] = y[i].real;
        globalImag[global_index] = y[i].imag;
      } else if (iora == ADD_VALUES) {
        globalReal[global_index] += y[i].real;
        globalImag[global_index] += y[i].imag;
      } else {
        free(globalReal);
        free(globalImag);
        free(recvcounts);
        free(displs);
        return 1;
      }
    }
  }

  /* Now scatter the updated global data back to each process */
  double *recvReal = (double *)malloc(localSize * sizeof(double));
  double *recvImag = (double *)malloc(localSize * sizeof(double));
  MPI_Scatterv(globalReal, recvcounts, displs, MPI_DOUBLE, recvReal, localSize,
               MPI_DOUBLE, 0, comm);
  MPI_Scatterv(globalImag, recvcounts, displs, MPI_DOUBLE, recvImag, localSize,
               MPI_DOUBLE, 0, comm);
  for (int i = 0; i < localSize; i++)
    x->data[i] = scalar_make(recvReal[i], recvImag[i]);
  free(recvReal);
  free(recvImag);
  if (rank == 0) {
    free(globalReal);
    free(globalImag);
  }
#else
  /* Real case: allocate a buffer for the global vector */
  PetscScalar *globalData = (PetscScalar *)malloc(N * sizeof(PetscScalar));
  MPI_Gatherv(x->data, localSize, MPI_DOUBLE, globalData, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  if (rank == 0) {
    for (int i = 0; i < ni; i++) {
      PetscInt global_index = ix[i];
      if (global_index < 0 || global_index >= N)
        continue;
      if (iora == INSERT_VALUES) {
        globalData[global_index] = y[i];
      } else if (iora == ADD_VALUES) {
        globalData[global_index] += y[i];
      } else {
        free(globalData);
        free(recvcounts);
        free(displs);
        return 1;
      }
    }
  }
  MPI_Scatterv(globalData, recvcounts, displs, MPI_DOUBLE, x->data, localSize,
               MPI_DOUBLE, 0, comm);
  free(globalData);
#endif

  free(recvcounts);
  free(displs);
  return 0;
}

PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            const PetscScalar y[], InsertMode iora) {
  switch (x->type) {
  case VECSEQ:
    return VecSetValues_Seq(x, ni, ix, y, iora);
  case VECMPI:
    return VecSetValues_MPI(x, ni, ix, y, iora);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecSetValues_MPI(x, ni, ix, y, iora)
                          : VecSetValues_Seq(x, ni, ix, y, iora);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode VecSetValuesBlocked_Seq(Vec x, PetscInt ni, const PetscInt ix[],
                                       const PetscScalar y[], InsertMode iora) {
  $assert(x->read_lock_count == 0,
          "Cannot set values: Vector is locked for reading.");
  $assert(x && x->data && ix && y && ni >= 0);
  int bs = x->map->bs;
  $assert(bs > 0, "VecSetValuesBlocked: block size must be positive.");

  for (int i = 0; i < ni; i++) {
    /* Convert global block index to local block index */
    int local_block = ix[i] - (x->map->rstart / bs);
    if (local_block < 0 || local_block >= (x->map->n / bs))
      continue; /* Skip blocks not owned locally */
    for (int j = 0; j < bs; j++) {
      int elem_index = bs * local_block + j;
#ifdef USE_COMPLEX
      if (iora == INSERT_VALUES) {
        x->data[elem_index].real = y[bs * i + j].real;
        x->data[elem_index].imag = y[bs * i + j].imag;
      } else if (iora == ADD_VALUES) {
        x->data[elem_index].real += y[bs * i + j].real;
        x->data[elem_index].imag += y[bs * i + j].imag;
      } else {
        return 1; /* Unsupported mode */
      }
#else
      if (iora == INSERT_VALUES)
        x->data[elem_index] = y[bs * i + j];
      else if (iora == ADD_VALUES)
        x->data[elem_index] += y[bs * i + j];
      else
        return 1;
#endif
    }
  }
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecSetValuesBlocked_MPI(Vec x, PetscInt ni, const PetscInt ix[],
                                       const PetscScalar y[], InsertMode iora) {
  PetscErrorCode ierr;
  int rank, size;
  MPI_Comm comm = x->comm;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int N = x->map->N; /* global number of elements */
  int localSize = x->map->n;
  int bs = x->map->bs;
  $assert(bs > 0, "VecSetValuesBlocked_MPI: block size must be positive.");

  /* Gather local sizes from all processes */
  PetscInt *recvcounts = (PetscInt *)malloc(size * sizeof(PetscInt));
  PetscInt *displs = (PetscInt *)malloc(size * sizeof(PetscInt));
  MPI_Allgather(&localSize, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
  displs[0] = 0;
  for (int i = 1; i < size; i++)
    displs[i] = displs[i - 1] + recvcounts[i - 1];

#ifdef USE_COMPLEX
  /* Allocate buffers for real and imaginary parts on each process */
  double *localReal = (double *)malloc(localSize * sizeof(double));
  double *localImag = (double *)malloc(localSize * sizeof(double));
  for (int i = 0; i < localSize; i++) {
    localReal[i] = x->data[i].real;
    localImag[i] = x->data[i].imag;
  }
  double *globalReal = NULL, *globalImag = NULL;
  if (rank == 0) {
    globalReal = (double *)malloc(N * sizeof(double));
    globalImag = (double *)malloc(N * sizeof(double));
  }
  MPI_Gatherv(localReal, localSize, MPI_DOUBLE, globalReal, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  MPI_Gatherv(localImag, localSize, MPI_DOUBLE, globalImag, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  free(localReal);
  free(localImag);

  /* Process 0 updates the global array using block indices.
     Note: on process 0, x->map->rstart is assumed 0 so that the global block
     index is simply ix[i]. */
  if (rank == 0) {
    for (int i = 0; i < ni; i++) {
      int block = ix[i]; /* global block index */
      if (block < 0 || block >= (N / bs))
        continue;
      for (int j = 0; j < bs; j++) {
        int elem_index = bs * block + j;
#ifdef USE_COMPLEX
        if (iora == INSERT_VALUES) {
          globalReal[elem_index] = y[bs * i + j].real;
          globalImag[elem_index] = y[bs * i + j].imag;
        } else if (iora == ADD_VALUES) {
          globalReal[elem_index] += y[bs * i + j].real;
          globalImag[elem_index] += y[bs * i + j].imag;
        } else {
          free(globalReal);
          free(globalImag);
          free(recvcounts);
          free(displs);
          return 1;
        }
#else
        if (iora == INSERT_VALUES) {
          globalData[elem_index] = y[bs * i + j];
        } else if (iora == ADD_VALUES) {
          globalData[elem_index] += y[bs * i + j];
        } else {
          free(globalData);
          free(recvcounts);
          free(displs);
          return 1;
        }
#endif
      }
    }
  }
  /* Now scatter the updated global real/imaginary arrays back to each process
   */
  double *recvReal = (double *)malloc(localSize * sizeof(double));
  double *recvImag = (double *)malloc(localSize * sizeof(double));
  MPI_Scatterv(globalReal, recvcounts, displs, MPI_DOUBLE, recvReal, localSize,
               MPI_DOUBLE, 0, comm);
  MPI_Scatterv(globalImag, recvcounts, displs, MPI_DOUBLE, recvImag, localSize,
               MPI_DOUBLE, 0, comm);
  for (int i = 0; i < localSize; i++)
    x->data[i] = scalar_make(recvReal[i], recvImag[i]);
  free(recvReal);
  free(recvImag);
  if (rank == 0) {
    free(globalReal);
    free(globalImag);
  }
#else
  /* Real case: allocate global buffer */
  PetscScalar *globalData = (PetscScalar *)malloc(N * sizeof(PetscScalar));
  MPI_Gatherv(x->data, localSize, MPI_DOUBLE, globalData, recvcounts, displs,
              MPI_DOUBLE, 0, comm);
  if (rank == 0) {
    for (int i = 0; i < ni; i++) {
      int block = ix[i];
      if (block < 0 || block >= (N / bs))
        continue;
      for (int j = 0; j < bs; j++) {
        int elem_index = bs * block + j;
        if (iora == INSERT_VALUES) {
          globalData[elem_index] = y[bs * i + j];
        } else if (iora == ADD_VALUES) {
          globalData[elem_index] += y[bs * i + j];
        } else {
          free(globalData);
          free(recvcounts);
          free(displs);
          return 1;
        }
      }
    }
  }
  MPI_Scatterv(globalData, recvcounts, displs, MPI_DOUBLE, x->data, localSize,
               MPI_DOUBLE, 0, comm);
  free(globalData);
#endif
  free(recvcounts);
  free(displs);
  x->hdr.state++;
  return 0;
}

PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora) {
  switch (x->type) {
  case VECSEQ:
    return VecSetValuesBlocked_Seq(x, ni, ix, y, iora);
  case VECMPI:
    return VecSetValuesBlocked_MPI(x, ni, ix, y, iora);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecSetValuesBlocked_MPI(x, ni, ix, y, iora)
                          : VecSetValuesBlocked_Seq(x, ni, ix, y, iora);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

PetscErrorCode ISGetSize(IS is, PetscInt *size) {
  $assert(size, "size is NULL\n");
  *size = is->map->N;
  return 0;
}

PetscErrorCode ISCreateGeneral(MPI_Comm comm, PetscInt n, const PetscInt idx[],
                               PetscCopyMode mode, IS *is) {
  // Allocate the IS structure
  struct _p_IS *newis = (struct _p_IS *)malloc(sizeof(struct _p_IS));
  $assert(newis != NULL, "Allocation for newis failed");

  // Set min and max indices
  newis->comm = comm;
  newis->min = (n > 0) ? idx[0] : -1;
  newis->max = (n > 0) ? idx[n - 1] : -1;
  newis->local_offset = (n > 0) ? idx[0] : 0;

  // Allocate and copy index data based on the mode
  if (mode == PETSC_COPY_VALUES) {
    newis->data = (PetscInt *)malloc(n * sizeof(PetscInt));
    $assert(newis->data != NULL, "Allocation for newis->data failed");
    for (int i = 0; i < n; i++)
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
    for (int i = 0; i < n; i++)
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
  newis->map = (SimpleMap)malloc(sizeof(*newis->map));
  $assert(newis->map != NULL, "Allocation for newis->map failed");

  MPI_Comm_size(comm, &(newis->map->nproc));
  MPI_Comm_rank(comm, &(newis->map->rstart)); // Use rank for offset calculation

  newis->map->bs = -1;
  newis->map->n = n; // Local size
  newis->map->rstart = first;
  newis->map->rend = first + step * n;

  /* Compute global N using MPI */
  int global_N;
  MPI_Allreduce(&n, &global_N, 1, MPI_INT, MPI_SUM, comm);
  newis->map->N = global_N; // Set the global index set size

  /* Initialize the remaining IS fields */
  newis->comm = comm;
  newis->min = first;
  newis->max = (n > 0) ? (first + step * (n - 1)) : first;
  newis->local_offset = first;

  /* Allocate and fill in the local index (data) array:
     data[i] = first + i * step, for 0 <= i < n. */
  if (n > 0) {
    newis->data = (PetscInt *)malloc(n * sizeof(PetscInt));
    $assert(newis->data != NULL, "Allocation for newis->data failed");
    for (int i = 0; i < n; i++)
      ((PetscInt *)newis->data)[i] = first + i * step;
  } else {
    newis->data = NULL;
  }

  /* Allocate and fill the total array with the same values as the data array.
   */
  if (n > 0) {
    newis->total = malloc(n * sizeof(PetscInt));
    $assert(newis->total != NULL, "Allocation for newis->total failed");
    for (int i = 0; i < n; i++)
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

  // Build a concatenated $vec using only the local portions.
  $vec big = $vec_zero(0);
  for (int i = 0; i < nx; i++) {
    int localSize = X[i]->map->n; // use local size (n) not global size (N)
    // Construct a $vec from the local data array.
    $vec tmp = $vec_make_from_dense(localSize, X[i]->data);
    big = $vec_concat(big, tmp);
  }
  // Convert the concatenated $vec back to a PETSc Vec.
  Vec newVec =
      civlToPetscVec(big, PETSC_DECIDE, PetscObjectComm((PetscObject)X[0]));
  *Y = newVec;

  // For this spec stub, we do not build an index set.
  if (x_is)
    *x_is = NULL;
  return 0;
}

PetscErrorCode VecGetValues_MPI(Vec xin, PetscInt ni, const PetscInt ix[],
                                PetscScalar y[]) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int start = xin->map->rstart; // First index owned by current process
  const int local_size = xin->map->n; // Local size of the vector

  for (int i = 0; i < ni; i++) {
    if (ix[i] < 0)
      continue;
    const int tmp = ix[i] - start;
    $assert(tmp >= 0 && tmp < local_size, "Index %d out of range", ix[i]);
    y[i] = xin->data[tmp];
  }
  return 0;
}

PetscErrorCode VecGetValues_Seq(Vec xin, PetscInt ni, const PetscInt ix[],
                                PetscScalar y[]) {
  for (int i = 0; i < ni; i++) {
    if (ix[i] < 0)
      continue; // Skip negative indices if they are to be ignored
    $assert(ix[i] >= 0, "Negative index %d", ix[i]);
    $assert(ix[i] < xin->map->n, "Index %d exceeds vector size %d", ix[i],
            xin->map->n);
    y[i] = xin->data[ix[i]];
  }
  return 0;
}

PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            PetscScalar y[]) {
  $assert(x != NULL);
  $assert(ix != NULL);
  $assert(y != NULL);
  switch (x->type) {
  case VECSEQ:
    return VecGetValues_Seq(x, ni, ix, y);
  case VECMPI:
    return VecGetValues_MPI(x, ni, ix, y);
  case VECSTANDARD:
    return (x->nproc > 1) ? VecGetValues_MPI(x, ni, ix, y)
                          : VecGetValues_Seq(x, ni, ix, y);
  default:
    $assert(0, "Invalid vector type");
  }
  return 0;
}

static inline PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n) {
  if (!a || !b || n == 0)
    return 1;
  const size_t scalar_len = n / sizeof(PetscScalar);
  const PetscScalar *x = (PetscScalar *)b;
  PetscScalar *y = (PetscScalar *)a;
  for (size_t i = 0; i < scalar_len; i++)
    y[i] = x[i];
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
    out->comm = MPI_COMM_WORLD;
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

PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x,
                     const PetscBLASInt *sx, const PetscScalar *y,
                     const PetscBLASInt *sy) {
  PetscScalar sum = scalar_zero;
  int i, j, k;
  if (*sx == 1 && *sy == 1) {
    for (i = 0; i < *n; i++)
      sum = scalar_add(sum, scalar_mul(PetscConj(x[i]), y[i]));
  } else {
    for (i = 0, j = 0, k = 0; i < *n; i++, j += *sx, k += *sy) {
      // sum += PetscConj(x[j]) * y[k];
      sum = scalar_add(sum, scalar_mul(PetscConj(x[j]), y[k]));
    }
  }
  return sum;
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

PetscErrorCode BLASscal_(const PetscBLASInt *n, const PetscScalar *alpha,
                         PetscScalar *x, const PetscBLASInt *incx) {
  int i, j;
  if (*incx == 1) {
    // Unit stride: process elements consecutively
    for (i = 0; i < *n; i++)
      x[i] = scalar_mul(*alpha, x[i]);
  } else {
    // Non-unit stride: process elements with specified increment
    for (i = 0, j = 0; i < *n; i++, j += *incx)
      x[j] = scalar_mul(*alpha, x[j]);
  }
}

PetscErrorCode BLASaxpy_(const PetscBLASInt *n, const PetscScalar *alpha,
                         const PetscScalar *x, const PetscBLASInt *incx,
                         PetscScalar *y, const PetscBLASInt *incy) {
  int i, j, k;
  if (*incx == 1 && *incy == 1)
    for (i = 0; i < *n; i++)
      y[i] = scalar_add(y[i], scalar_mul(*alpha, x[i]));
  else
    for (i = 0, j = 0, k = 0; i < *n; i++, j += *incx, k += *incy)
      y[k] = scalar_add(y[k], scalar_mul(*alpha, x[j]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

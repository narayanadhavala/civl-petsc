#ifndef _PETSCVEC_H
#define _PETSCVEC_H
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CIVL_RTYPE double
#include "civlcomplex.cvh"
#ifdef USE_COMPLEX
#define CIVL_COMPLEX
#else
#undef CIVL_COMPLEX
#endif
#include "civlvec.cvh"

// Basic types...
typedef int PetscInt;
typedef int PetscMPIInt;
typedef double PetscReal;
#define PETSC_REAL MPI_DOUBLE // used in MPI communication

// Operations on complex numbers that also make sense for reals...
#ifdef USE_COMPLEX
typedef STYPE PetscScalar;
#define PetscConj(a) scalar_conj(x)
#define PetscImaginaryPart(a) ((a).imag)
#define PetscRealPart(a) ((a).real)
#define PetscAbsScalar(a) scalar_abs(a)
#define $cmake(x, y) $make_complex(x, y)
#else
typedef PetscReal PetscScalar;
#define PetscConj(a) scalar_conj(x)
#define PetscImaginaryPart(a) ((PetscReal)(0))
#define PetscRealPart(a) (a)
#define PetscAbsScalar(a) fabs(a)
#define $cmake(x, y) (x)
#endif

// PETSc's Boolean values
#define PETSC_FALSE 0
#define PETSC_TRUE 1

// PetscBLASInt is an integer datatype used for BLAS operations
typedef int PetscBLASInt;

// PetscErrorCode is an error code datatype used for error handling
typedef int PetscErrorCode;

typedef int PetscClassId;

// PetscOptions is a datatype representing a set of PETSc options
typedef struct PetscOptions_s *PetscOptions;

// PetscLayout is a datatype representing the layout of a PETSc object
typedef struct _n_PetscLayout *PetscLayout;

// PetscBool is a boolean datatype
typedef _Bool PetscBool;

// PetscLogDouble is a datatype representing a double precision
// floating point number used for logging
typedef double PetscLogDouble;

// The vector "types" --- the kind of vector (sequential, MPI,
// standard, ...)
typedef int VecType;
#define VECSEQ 1
#define VECMPI 2
#define VECSTANDARD 3

typedef enum {
  PETSC_COPY_VALUES,
  PETSC_OWN_POINTER,
  PETSC_USE_POINTER
} PetscCopyMode;

#define NUM_NORM_TYPES 5

typedef enum {
  IS_INFO_UNKNOWN = 0,
  IS_INFO_FALSE = 1,
  IS_INFO_TRUE = 2
} ISInfoBool;

// ISInfo - Info that may either be computed or set as known for an index set
typedef enum {
  IS_INFO_MIN = -1,
  IS_SORTED = 0,
  IS_UNIQUE = 1,
  IS_PERMUTATION = 2,
  IS_INTERVAL = 3,
  IS_IDENTITY = 4,
  IS_INFO_MAX = 5
} ISInfo;

// Forward declarations to resolve mutual dependencies
struct _p_PetscObject;
typedef struct _p_PetscObject *PetscObject;

struct _n_PetscObjectList;
typedef struct _n_PetscObjectList *PetscObjectList;

// Definition of struct _n_PetscObjectList
struct _n_PetscObjectList {
  char name[256];
  PetscBool skipdereference; /* when the PetscObjectList is destroyed do not
                                call PetscObjectDereference() on this object */
  PetscObject obj;
  PetscObjectList next;
};

// Definition of struct _p_PetscObject
struct _p_PetscObject {
  PetscClassId classid;
  const char *type_name;
  const char *class_name;
  PetscObjectList olist;
  PetscInt state;              // current state
  PetscInt *realcomposedstate; // Array of states
  PetscReal *realcomposeddata; // Array of data
};

#define PETSCHEADER(ObjectOps)                                                 \
  struct _p_PetscObject hdr;                                                   \
  ObjectOps ops[1]

// A component of a Vec struct specifying how the vector is
// distributed across processes.
typedef struct map_s {
  PetscInt n;            // local_size
  PetscInt N;            // global_size
  PetscInt rstart, rend; // local start, local end + 1
  PetscInt bs;           // for now assuming the block size as 1
} *SimpleMap;

// Now we can define Vec_s using PETSCHEADER
typedef struct Vec_s {
  PETSCHEADER(struct _VecOps);
  MPI_Comm comm;
  PetscScalar *data;
  SimpleMap map;
  VecType type;
  PetscInt nproc;
  int read_lock_count;
} *Vec;

// PetscViewer is a datatype representing an object used for viewing PETSc
// objects
// TODO: why the strange name _p_...?
typedef struct _p_PetscViewer *PetscViewer;

// PetscViewerFormat is an enum representing different formats for PetscViewer
typedef enum {
  PETSC_VIEWER_DEFAULT,
  PETSC_VIEWER_STDOUT_SELF
} PetscViewerFormat;

// TODO why strange name?
struct _p_PetscViewer {
  PetscViewerFormat format;
  int iformat;
  void *data;
};

typedef struct {
  PetscReal val;
  PetscInt index;
} VecLocation;

struct _p_PetscDeviceContext {
  PETSCHEADER(struct _DeviceContextOps);
  void *data; /* solver contexts, event, stream */
  PetscInt
      numChildren; /* how many children does this context expect to destroy */
  PetscInt maxNumChildren; /* how many children can this context have room for
                              without realloc'ing */
  PetscBool setup;
  PetscBool usersetdevice;
};

// PETSC_SUCCESS represents a successful PETSc operation
#define PETSC_SUCCESS ((PetscErrorCode)0)

// PETSC_ERR_ARG_OUTOFRANGE represents an error code for out-of-range input
// arguments
#define PETSC_ERR_ARG_OUTOFRANGE ((PetscErrorCode)63)

// PETSC_ERR_ARG_SIZ represents an error code for nonconforming object sizes
// used in PETSc operations
#define PETSC_ERR_ARG_SIZ ((PetscErrorCode)60)

// PETSC_ERR_SUP represents an error code indicating no support for the
// requested operation in PETSc
#define PETSC_ERR_SUP ((PetscErrorCode)56)

#define PETSC_ERR_ARG_NULL ((PetscErrorCode)85)
#define PETSC_ERR_ARG_CORRUPT ((PetscErrorCode)64)
#define PETSC_ERR_ARG_WRONG ((PetscErrorCode)62)
#define PETSC_ERR_ARG_WRONGSTATE ((PetscErrorCode)73)
#define PETSCFREEDHEADER (-1)
#define PETSC_ERR_RETURN ((PetscErrorCode)99)
#define PETSC_ERR_ARG_IDN ((PetscErrorCode)61)
#define PETSC_ERR_ARG_INCOMP ((PetscErrorCode)75)
#define PETSC_ERROR_INITIAL 0

// PETSC_DECIDE represents a constant used in place of an integer argument
// when you want PETSc to choose the value for that argument
#ifndef PETSC_DECIDE
#define PETSC_DECIDE (-1)
#endif

// PETSC_DETERMINE is like PETSC_DECIDE.  // TODO: why do we need both?
#ifndef PETSC_DETERMINE
#define PETSC_DETERMINE PETSC_DECIDE
#endif

// PETSC_DEFAULT represents a default value
#define PETSC_DEFAULT (-2)

// PETSC_COMM_WORLD is the MPI communiator used for PETSc communiation
#define PETSC_COMM_WORLD MPI_COMM_WORLD

#ifndef MPI_COMM_SELF
#define MPI_COMM_SELF ((MPI_Comm)0x44000000)
#endif

#ifndef __isfinitef
#define __isfinitef(x) 1
#endif

#define PETSC_COMM_SELF MPI_COMM_SELF

// PetscInt_FMT is a format specifier for PetscInt used in formatted output
#define PetscInt_FMT "d"

// PetscSqrtReal computes the square root of a real number.
#define PetscSqrtReal(a) sqrt(a)

// an ordered pair: (real, int) used in MPI operations
#define PETSC_REAL_INT MPI_DOUBLE_INT

// PETSC_SMALL represents a small value used for numerical comparison
#define PETSC_SMALL 1.e-10

// PETSC_MAX_REAL represents the maximum real number value
#define PETSC_MAX_REAL 1.7976931348623157e+308

// PETSC_MIN_REAL represents the minimum real number value
#define PETSC_MIN_REAL (-PETSC_MAX_REAL)

// Enumeration of different types of norms used in PETSc
typedef enum NORM_TYPE {
  NORM_1 = 0,
  NORM_2 = 1,
  NORM_FROBENIUS = 2,
  NORM_INFINITY = 3,
  NORM_1_AND_2 = 4
} NormType;

extern PetscInt NormIds[5];

// Enumeration of different insert modes used in PETSc
typedef enum INSERT_MODE {
  NOT_SET_VALUES,
  INSERT_VALUES,
  ADD_VALUES,
  MAX_VALUES,
  MIN_VALUES,
  INSERT_ALL_VALUES,
  ADD_ALL_VALUES,
  INSERT_BC_VALUES,
  ADD_BC_VALUES
} InsertMode;

// PetscCall is a macro used to wrap calls to PETSc functions
#define PetscCall(a) a

// PetscFunctionBeginUser marks the beginning of a user-defined function
#define PetscFunctionBeginUser

/*
  Calculates the first global index owned by a given process.
  Parameters:
  - v: The Vec object containing the vector information.
  - p: The rank of the process.

  Returns: The first global index owned by process p.
 */
#define FIRST(v, p)                                                            \
  ((v->map->N / v->nproc) * (p) +                                              \
   ((p) < (v->map->N % v->nproc) ? (p) : (v->map->N % v->nproc)))

/*
   Calculates the number of elements owned by a given process.
   Parameters:
   - v: The Vec object containing the vector information.
   - p: The rank of the process.

   Returns: The number of elements owned by process p.
*/
#define NUM_OWNED(v, p)                                                        \
  ((v->map->N / v->nproc) + ((p) < (v->map->N % v->nproc) ? 1 : 0))

/*
    Determines the owner process of a given global index.
    Parameters:
    - v: The Vec object containing the vector information.
    - i: The global index.

    Returns: The rank of the process that owns the element at global index i.
*/
#define OWNER(v, i)                                                            \
  ((i) < ((v->map->N / v->nproc) + 1) * (v->map->N % v->nproc)                 \
       ? (i) / ((v->map->N / v->nproc) + 1)                                    \
       : ((i) - ((v->map->N / v->nproc) + 1) * (v->map->N % v->nproc)) /       \
                 (v->map->N / v->nproc) +                                      \
             (v->map->N % v->nproc))

/*
  Converts a global index to its corresponding local index.
  Parameters:
  - v: The Vec object containing the vector information.
  - i: The global index.

  Returns: The local index corresponding to the given global index i.
 */
#define LOCAL_INDEX(v, i) ((i) - FIRST(v, OWNER(v, i)))

/*
  Converts a local index to its corresponding global index.
  Parameters:
  - v: The Vec object containing the vector information.
  - p: The rank of the process.
  - j: The local index.

  Returns: The global index corresponding to the local index j on process p.
 */
#define GLOBAL_INDEX(v, p, j) (FIRST(v, p) + (j))

/*
  Prints formatted output, but only from the first (rank 0) process in the
  communicator.

  Parameters:
  - comm      MPI communicator that defines the group of processes.
  - format    A string that specifies how subsequent arguments are converted for
  output.
  - ...       (Variable arguments) Additional arguments specifying data to be
  printed. These correspond to the conversion specifiers in the format string.

  Behavior:
  - Determines the rank of the calling process within the communicator.
  - If the rank is 0 (i.e., it's the "main" or "root" process):
    - Uses vprintf to print the formatted string with the provided arguments.
  - If the rank is not 0, the function does nothing for now.

  Returns:
  - PetscErrorCode  Always returns 0 in this implementation, indicating success.
*/
#define PetscPrintf(comm, ...)                                                 \
  do {                                                                         \
    int __rank;                                                                \
    MPI_Comm_rank(comm, &__rank);                                              \
    if (__rank == 0) {                                                         \
      printf(__VA_ARGS__);                                                     \
    }                                                                          \
  } while (0)

#define VEC_CLASSID 123

PetscErrorCode PetscError(MPI_Comm comm, int line, const char *func,
                          const char *file, PetscErrorCode n, int p,
                          const char *mess, ...);

#define PetscUnlikely(x) (x)

#define SETERRQ(comm, ierr, ...)                                               \
  do {                                                                         \
    PetscErrorCode ierr_seterrq_petsc_ =                                       \
        PetscError(comm, __LINE__, __func__, __FILE__, ierr,                   \
                   PETSC_ERROR_INITIAL, __VA_ARGS__);                          \
    return ierr_seterrq_petsc_ ? ierr_seterrq_petsc_ : PETSC_ERR_RETURN;       \
  } while (0)

#define PetscCheck(cond, comm, ierr, ...)                                      \
  do {                                                                         \
    if (PetscUnlikely(!(cond))) {                                              \
      SETERRQ(comm, ierr, __VA_ARGS__);                                        \
    }                                                                          \
  } while (0)

PetscErrorCode PetscValidHeaderSpecific(void *x, PetscClassId cid, int arg);

#define PetscValidType(a, arg) ((void)0)

#define PetscLogEventBegin(e, o1, o2, o3, o4) ((void)0)

#define PetscLogEventEnd(e, o1, o2, o3, o4) ((void)0)

#define PetscAssertPointer(h, arg) $assert(h != NULL)

#define PETSC_FIRST_ARG_(N, ...) N

#define PETSC_FIRST_ARG(args) PETSC_FIRST_ARG_ args

#define PetscStringize_(...) #__VA_ARGS__
/*
  PetscStringize - Stringize a token

  Synopsis:
  #include <petscmacros.h>
  const char* PetscStringize(x)

  No Fortran Support

  Input Parameter:
. x - The token you would like to stringize

  Output Parameter:
. <return-value> - The string representation of `x`
*/
#define PetscStringize(...) PetscStringize_(__VA_ARGS__)

#define PETSC_SELECT_16TH(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12,   \
                          a13, a14, a15, a16, ...)                             \
  a16

#define PETSC_NUM(...)                                                         \
  PETSC_SELECT_16TH(__VA_ARGS__, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,   \
                    TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,     \
                    TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE, TWOORMORE,     \
                    ONE, throwaway)

#define PETSC_REST_HELPER_TWOORMORE(first, ...) , __VA_ARGS__
#define PETSC_REST_HELPER_ONE(first)

#define PETSC_REST_HELPER2(qty, ...) PETSC_REST_HELPER_##qty(__VA_ARGS__)
#define PETSC_REST_HELPER(qty, ...) PETSC_REST_HELPER2(qty, __VA_ARGS__)

#define PETSC_REST_ARG(...)                                                    \
  PETSC_REST_HELPER(PETSC_NUM(__VA_ARGS__), __VA_ARGS__)

#define PetscUseTypeMethod(obj, ...)                                           \
  do {                                                                         \
    PetscCheck((obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused)),             \
               PetscObjectComm((PetscObject)obj), PETSC_ERR_SUP,               \
               "No method %s for %s of type %s",                               \
               PetscStringize(PETSC_FIRST_ARG((__VA_ARGS__, unused))),         \
               ((PetscObject)obj)->class_name, ((PetscObject)obj)->type_name); \
    PetscCall(((obj)->ops->PETSC_FIRST_ARG((__VA_ARGS__, unused)))(            \
        obj PETSC_REST_ARG(__VA_ARGS__)));                                     \
  } while (0)

#define PetscValidLogicalCollectiveInt(a, b, arg)                              \
  do {                                                                         \
    PetscInt b0 = (b), b1[2], b2[2];                                           \
    b1[0] = -b0;                                                               \
    b1[1] = b0;                                                                \
    PetscCall(MPIU_Allreduce(b1, b2, 2, MPI_INT, MPI_MAX,                      \
                             PetscObjectComm((PetscObject)(a))));              \
    PetscCheck(-b2[0] == b2[1], PetscObjectComm((PetscObject)(a)),             \
               PETSC_ERR_ARG_WRONG,                                            \
               "Int value must be same on all processes, argument # %d", arg); \
  } while (0)

#define PetscValidFunction(f, arg)                                             \
  PetscCheck((f), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,                         \
             "Null Function Pointer: Parameter # %d", arg)

#define VecSetErrorIfLocked(x, args)                                           \
  do {                                                                         \
    $assert((x), "Error: Null vector passed to VecSetErrorIfLocked.\n");       \
    $assert((x)->read_lock_count == 0, "Vector was locked for access.\n");     \
  } while (0)

#define PetscMalloc1(m1, r1)                                                   \
  (*(r1) = malloc((m1) * sizeof(**(r1))), PETSC_SUCCESS)

PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj, PetscInt id,
                                              PetscReal *data, PetscBool *flag);

PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj, PetscInt id,
                                              PetscReal data);

PetscErrorCode PetscInfo_Private(PetscObject obj, const char message[]);

#define PetscInfo(A, ...) PetscInfo_Private(((PetscObject)A), __VA_ARGS__)

PetscBool PetscIsInfOrNanReal(PetscReal v);

int __isfinited(double x);

bool $is_scalar_zero(PetscScalar alpha);

bool $is_scalar_one(PetscScalar alpha);

bool $is_lessthan(PetscScalar alpha, double n);

bool $is_greaterthan(PetscScalar alpha, double n);

#define VecCheckAssembled(a) ((void)0)

#define PetscFunctionBeginHot

#define PetscValidLogicalCollectiveEnum(a, b, arg) ((void)0)

// PetscPrintf0 is a modified macro for PetscPrintf used to print formatted
// output with no arguments
#define PetscPrintf0(comm, format) (printf(format))

// PetscFunctionBegin marks the beginning of a Petsc function
#define PetscFunctionBegin PetscErrorCode __ierr = 0;

// PetscFunctionReturn returns an error code from a Petsc function
#define PetscFunctionReturn(a) return a

#ifdef USE_COMPLEX
#define PETSC_USE_COMPLEX 1
#else
#define PETSC_USE_COMPLEX 0
#endif

#ifndef PETSC_USE_DEBUG
#define PETSC_USE_DEBUG 0
#else
#define PETSC_USE_DEBUG 1
#endif

#ifndef PETSC_USE_REAL___FP16
#define PETSC_USE_REAL___FP16 1
#else
#define PETSC_USE_REAL___FP16 0
#endif

// PetscDefined_Internal checks if a macro is defined internally
#define PetscDefined_Internal(x) (x)

// PetscDefined checks if a macro is defined
#define PetscDefined(def) PetscDefined_Internal(PETSC_##def)

// PETSC_EXTERN specifies an external linkage for a variable or function
#define PETSC_EXTERN extern

// PETSC_EXTERN_TLS specifies an external linkage for a thread-local variable
#define PETSC_EXTERN_TLS PETSC_EXTERN

typedef int PetscLogEvent;

PetscBool PetscIsInfReal(PetscReal a);

PetscBool PetscIsNanReal(PetscReal a);

// PetscCallBLAS calls a BLAS function
#define PetscCallBLAS(x, X) X

#define MPIU_Allreduce(a, b, c, d, e, fcomm)                                   \
  do {                                                                         \
    int ierr = MPI_Allreduce((a), (b), (c), (d), (e), (fcomm));                \
    if (ierr != MPI_SUCCESS) {                                                 \
      fprintf(stderr, "Error in MPI_Allreduce: %d\n", ierr);                   \
      return ierr;                                                             \
    }                                                                          \
  } while (0)

// PetscArraycpy copies elements from one array (str1) to another (str2)
#define PetscArraycpy(str1, str2, cnt)                                         \
  ((sizeof(*(str1)) == sizeof(*(str2)))                                        \
       ? PetscMemcpy((str1), (str2), (size_t)(cnt) * sizeof(*(str1)))          \
       : PETSC_ERR_ARG_SIZ)

// Macro to check if two vectors have the same type
#define PetscCheckSameType(a, arga, b, argb)                                   \
  $assert((a) && (b), "Error: Null pointer passed to PetscCheckSameType.");    \
  $assert((a)->type == (b)->type, "Error: Vectors have different types.")

// Macro to check if two vectors have the same communicator
#define PetscCheckSameComm(a, arga, b, argb)                                   \
  $assert((a) && (b), "Error: Null pointer passed to PetscCheckSameComm.");    \
  $assert((a)->comm == (b)->comm,                                              \
          "Error: Vectors have different communicators.")

#define PetscCheckSameTypeAndComm(a, arga, b, argb)                            \
  do {                                                                         \
    PetscCheckSameType(a, arga, b, argb);                                      \
    PetscCheckSameComm(a, arga, b, argb);                                      \
  } while (0)

// Macro to check if two vectors have the same global size or not
#define VecCheckSameSize(a, arga, b, argb)                                     \
  $assert((a) && (b), "Error: Null pointer passed to VecCheckSameSize.");      \
  $assert((a)->map->N == (b)->map->N,                                          \
          "Error: Vectors have different global sizes.")
/*
  VecLockReadPush - Pushes a read-only lock on a vector to prevent it from being
  written to.

  Parameters:
  - x The vector to lock for reading.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
*/
#define VecLockReadPush(x)                                                     \
  do {                                                                         \
    $assert((x) != NULL, "VecLockReadPush: Vector pointer is NULL.");          \
    (x)->read_lock_count++;                                                    \
  } while (0)

#define VecLockReadPop(x)                                                      \
  do {                                                                         \
    $assert((x) != NULL, "VecLockReadPop: Vector pointer is NULL.");           \
    $assert((x)->read_lock_count > 0,                                          \
            "VecLockReadPop: read_lock_count is already zero.");               \
    (x)->read_lock_count--;                                                    \
  } while (0)

#define PetscObjectStateIncrease(obj) ((obj)->state++, PETSC_SUCCESS)

#define VecAsyncFnName(Base) VEC_##Base##_ASYNC_FN_NAME

#define PetscObjectQueryFunction(obj, name, fptr) ((void)0)

#define VecMethodDispatch(v, dctx, async_name, name, async_arg_types, ...)     \
  do {                                                                         \
    PetscErrorCode(*_8_f) async_arg_types = NULL;                              \
    if (dctx)                                                                  \
      PetscCall(                                                               \
          PetscObjectQueryFunction((PetscObject)(v), async_name, &_8_f));      \
    if (_8_f) {                                                                \
      PetscCall((*_8_f)(v, __VA_ARGS__, dctx));                                \
    } else {                                                                   \
      PetscUseTypeMethod(v, name, __VA_ARGS__);                                \
    }                                                                          \
  } while (0)

#define MPIU_REAL MPI_DOUBLE

#define MPIU_MAX MPI_MAX

#define PetscValidLogicalCollectiveScalar(a, b, arg)                           \
  do {                                                                         \
    PetscScalar b0 = (b);                                                      \
    PetscReal b1[5], b2[5];                                                    \
    if (PetscIsNanScalar(b0)) {                                                \
      b1[4] = 1;                                                               \
    } else {                                                                   \
      b1[4] = 0;                                                               \
    };                                                                         \
    b1[0] = -PetscRealPart(b0);                                                \
    b1[1] = PetscRealPart(b0);                                                 \
    b1[2] = -PetscImaginaryPart(b0);                                           \
    b1[3] = PetscImaginaryPart(b0);                                            \
    PetscCall(MPIU_Allreduce(b1, b2, 5, MPIU_REAL, MPIU_MAX,                   \
                             PetscObjectComm((PetscObject)(a))));              \
    PetscCheck(b2[4] > 0 || (PetscEqualReal(-b2[0], b2[1]) &&                  \
                             PetscEqualReal(-b2[2], b2[3])),                   \
               PetscObjectComm((PetscObject)(a)), PETSC_ERR_ARG_WRONG,         \
               "Scalar value must be same on all processes, argument # %d",    \
               arg);                                                           \
  } while (0)

#define VecCheckSameLocalSize(x, ar1, y, ar2)                                  \
  do {                                                                         \
    PetscCheck(                                                                \
        (x)->map->n == (y)->map->n, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,     \
        "Incompatible vector local lengths parameter # %d local size "         \
        "%" PetscInt_FMT " != parameter # %d local size %" PetscInt_FMT,       \
        ar1, (x)->map->n, ar2, (y)->map->n);                                   \
  } while (0)

PetscBool PetscIsNanScalar(PetscScalar v);

PetscBool PetscEqualReal(PetscReal a, PetscReal b);

/*
  PetscObjectComm - Gets the MPI communicator for any `PetscObject`
  regardless of the type. Parameters:
  - obj Any PETSc object, for example a `Vec`, `Mat`, or `KSP`. It must
  be cast to a (`PetscObject`), for example,
  `PetscObjectComm((PetscObject)mat)`.

  Returns: MPI_Comm (the MPI communicator associated with the object or
  `MPI_COMM_NULL` if `obj` is not valid).

  Note:
    This function returns the MPI communicator associated with the PETSc
  object `obj`. If `obj` is `NULL` or invalid, it returns
  `MPI_COMM_NULL`.
*/
MPI_Comm PetscObjectComm(PetscObject obj);

/*
  Initializes PETSc. The file and help arguments are currently ignored.
  Parameters:
  - argc Pointer to the number of command line arguments.
  - args Pointer to the array of command line arguments.
  - file Optional file name for options; may be NULL.
  - help Optional help string; may be NULL.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
 */
PetscErrorCode PetscInitialize(int *argc, char ***args, const char file[],
                               const char help[]);

/*
  Retrieves an integer value from the PETSc options database.
  Parameters:
  - options PETSc options object.
  - pre Prefix string for the option.
  - name Name of the option.
  - ivalue Pointer to store the retrieved integer value.
  - set Pointer to a boolean indicating if the option was set.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
 */
PetscErrorCode PetscOptionsGetInt(PetscOptions options, const char pre[],
                                  const char name[], PetscInt *ivalue,
                                  PetscBool *set);

/*
  Creates a new empty vector.
  Parameters:
  - comm MPI communicator.
  - vec Pointer to the Vec object to be created.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for the Vec structure and its internal SimpleMap.
        Initializes fields to default values.
 */
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec);

/*
  Computes the absolute value of a real number.
  Parameters:
  - v1 Input real number.

  Returns: The absolute value of v1.
 */
PetscReal PetscAbsReal(PetscReal v1);

/*
  Safely casts a PetscInt to a PetscBLASInt.
  Parameters:
  - a Input PetscInt value.
  - b Pointer to store the casted PetscBLASInt value.

  Returns: PetscErrorCode (0 on success, 1 if out of range).

  Note: Checks for negative values and overflow before casting.
 */
PetscErrorCode PetscBLASIntCast(PetscInt a, PetscBLASInt *b);

/*
  Creates a new vector of the same type as an existing vector.
  Parameters:
  - v Input vector to be duplicated.
  - newv Pointer to the new vector to be created.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for the new vector and copies size information.
 */
PetscErrorCode VecDuplicate(Vec v, Vec *newv);

/*
  Creates multiple vectors of the same type as an existing vector.
  Parameters:
  - v Input vector to be duplicated.
  - m Number of vectors to create.
  - V Pointer to an array of Vec pointers to store the new vectors.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Creates m new vectors by calling VecDuplicate m times.
 */
PetscErrorCode VecDuplicateVecs(Vec v, PetscInt m, Vec *V[]);

/*
  Destroys multiple vectors and frees their memory.
  Parameters:
  - m Number of vectors to destroy.
  - vv Pointer to an array of Vec pointers to be destroyed.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Calls VecDestroy on each vector and frees the array.
 */
PetscErrorCode VecDestroyVecs(PetscInt m, Vec *vv[]);

/*
  Gets the ownership range of a PETSc vector.

  Parameters:
  - x: The PETSc vector whose ownership range is to be retrieved.
  - low: Pointer to store the first local entry owned by the calling process.
  - high: Pointer to store one past the last local entry owned by the calling
  process.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: The ownership range refers to the portion of the vector owned by the
  current process in a parallel computation. The range is of the form [low,
  high), meaning that 'low' is inclusive and 'high' is exclusive.
 */
PetscErrorCode VecGetOwnershipRange(Vec x, PetscInt *low, PetscInt *high);

/*
  Retrieves the ownership ranges of all processes for a PETSc vector.

  Parameters:
  - x: The PETSc vector whose ownership ranges are to be retrieved.
  - ranges: Pointer to an array of size (number of processes + 1) to store the
  ranges. Each element in the array gives the starting index of the portion
  owned by each process.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: The ranges array contains the global indices of the starting points of
  each process's ownership of the vector. The last element of the array is one
  past the end of the vector, so the range for process `p` is [ranges[p],
  ranges[p+1]).
 */
PetscErrorCode VecGetOwnershipRanges(Vec x, const PetscInt *ranges[]);

PetscErrorCode PetscSplitOwnership(MPI_Comm comm, PetscInt *n, PetscInt *N);
/*
  Sets the local and global sizes of a vector.
  Parameters:
  - v Vector to set sizes for.
  - n Local size (or PETSC_DECIDE).
  - N Global size (or PETSC_DETERMINE).

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Sets the local and global sizes in the vector's map.
 */
PetscErrorCode VecSetSizes(Vec v, PetscInt n, PetscInt N);

/*
  Sets the block size of a vector.
  Parameters:
  - v Vector to set block size for.
  - bs Block size to set.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
 */
PetscErrorCode VecSetBlockSize(Vec v, PetscInt bs);

/*
  Configures the vector from options.
  Parameters:
  - vec Vector to configure.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for vector data based on the local size.
 */
PetscErrorCode VecSetFromOptions(Vec vec);

PetscErrorCode VecSetType(Vec vec, VecType newType);

PetscErrorCode VecGetType(Vec vec, VecType *type);

/*
  Sets all components of a vector to a single scalar value.
  Parameters:
  - x Vector to set values in.
  - alpha Scalar value to set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecSet(Vec x, PetscScalar alpha);

/*
  Displays the vector.
  Parameters:
  - vec Vector to view.
  - viewer PetscViewer object.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Prints vector contents to stdout in simplified version.
 */
PetscErrorCode VecView(Vec vec, PetscViewer viewer);

/*
  Adds floating point operations to the global counter.
  Parameters:
  - n Number of flops to add.

  Returns: PetscErrorCode (0 on success, 1 if n is negative).
 */
PetscErrorCode PetscLogFlops(PetscLogDouble n);

/*
  Swaps the values between two vectors.
  Parameters:
  - x First vector.
  - y Second vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecSwap(Vec x, Vec y);

/*
  Computes the dot product of two vectors.
  Parameters:
  - x First vector.
  - y Second vector.
  - val Pointer to store the dot product result.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: In complex mode, val = x · y' where y' is the conjugate transpose of y.
 */
PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val);

PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val);

PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val);

PetscErrorCode VecMXDot_Private(
    Vec x, PetscInt nv, const Vec y[], PetscScalar result[],
    PetscErrorCode (*mxdot)(Vec, PetscInt, const Vec[], PetscScalar[]),
    PetscLogEvent event);

/*
  VecMTDot - Computes indefinite vector multiple dot products, i.e.,

    val[i] = sum_{k=0..n-1}( x[k] * y[i][k] ),

  with NO complex conjugation. For complex vectors, the "transpose" is used,
  not the "conjugate transpose."

  Collective

  Input Parameters:
  + x   - one vector
  . nv  - number of vectors
  - y   - array of vectors

  Output Parameter:
  . val - array of dot products (length nv)

  Note: This is a stub for demonstration and verification. It does not
  perform any parallel reductions, nor check MPI ranks. In a realistic
  PETSc implementation, you would gather partial sums from each rank.
*/
PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]);

/*
  Computes multiple vector dot products.
  Parameters:
  - x Vector to be dotted with others.
  - nv Number of vectors.
  - y Array of vectors to dot with x.
  - val Array to store the results.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: In complex mode, val[i] = x · y[i]' where y[i]' is the conjugate of
  y[i].
 */
PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]);

/*
  Returns the global number of elements in the vector.
  Parameters:
  - x Input vector.
  - size Pointer to store the size.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecGetSize(Vec x, PetscInt *size);

/*
  Returns the number of elements of the vector stored in local memory.
  Parameters:
  - x Input vector.
  - size Pointer to store the local size.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecGetLocalSize(Vec x, PetscInt *size);

/*
  Determines the vector component with maximum real part and its location.
  Parameters:
  - x Input vector.
  - p Pointer to store the index of the maximum element.
  - val Pointer to store the maximum value.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: For complex vectors, considers the real part for comparison.
 */
PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val);

/*
  Determines the vector component with minimum real part and its location.
  Parameters:
  - x Input vector.
  - p Pointer to store the index of the minimum element.
  - val Pointer to store the minimum value.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: For complex vectors, considers the real part for comparison.
 */
PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val);

typedef struct _p_PetscDeviceContext *PetscDeviceContext;

PetscErrorCode VecScaleAsync_Private(Vec x, PetscScalar alpha,
                                     PetscDeviceContext dctx);

PetscErrorCode VecSetAsync_Private(Vec x, PetscScalar alpha,
                                   PetscDeviceContext dctx);

/*
  Scales a vector by multiplying each element by a scalar.
  Parameters:
  - x Vector to scale.
  - alpha Scalar to multiply by.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars.
 */
PetscErrorCode VecScale(Vec x, PetscScalar alpha);

/*
  Compares two vectors for equality.
  Parameters:
  - vec1: First vector to compare.
  - vec2: Second vector to compare.
  - flg: Pointer to a boolean flag that will be set to `PETSC_TRUE` if the
  vectors are equal, `PETSC_FALSE` otherwise.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: This function checks if the vectors have the same dimensions and block
  size, and if their elements are equal. Supports both real and complex vectors.
 */
PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg);

PetscErrorCode VecMAXPYAsync_Private(Vec y, PetscInt nv,
                                     const PetscScalar alpha[], Vec x[],
                                     PetscDeviceContext dctx);

/*
  Computes y = y + sum(alpha[i] * x[i]) for multiple vectors. Updates the
  vector `y` by adding scaled versions of vectors `x[i]` weighted by
  `alpha[i]` for each `i` in the range `[0, nv-1]`. Parameters:
  - y Vector to be updated.
  - nv Number of vectors.
  - alpha Array of scalars.
  - x Array of vectors.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[]);

PetscErrorCode VecMAXPBY(Vec y, PetscInt nv, const PetscScalar alpha[],
                         PetscScalar beta, Vec x[]);

PetscErrorCode VecAXPYAsync_Private(Vec y, PetscScalar alpha, Vec x,
                                    PetscDeviceContext dctx);

/*
  Computes y = alpha * x + y. Updates the vector `y` by adding the vector
  `x` scaled by the scalar `alpha`. Parameters:
  - y Vector to be updated.
  - alpha Scalar multiplier.
  - x Vector to be added.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x);

PetscErrorCode VecAXPBYAsync_Private(Vec y, PetscScalar alpha, PetscScalar beta,
                                     Vec x, PetscDeviceContext dctx);

PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x);

PetscErrorCode VecAXPBYPCZAsync_Private(Vec z, PetscScalar alpha,
                                        PetscScalar beta, PetscScalar gamma,
                                        Vec x, Vec y, PetscDeviceContext dctx);

PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y);

PetscErrorCode VecAYPXAsync_Private(Vec y, PetscScalar beta, Vec x,
                                    PetscDeviceContext dctx);

/*
  Computes y = x + beta * y. Updates the vector `y` by adding the vector `x` to
  `y` scaled by the scalar `beta`. Parameters:
  - y Vector to be updated.
  - beta Scalar multiplier.
  - x Vector to be added.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x);

PetscErrorCode VecWAXPYAsync_Private(Vec w, PetscScalar alpha, Vec x, Vec y,
                                     PetscDeviceContext dctx);

/*
  Computes w = alpha * x + y. Stores the result in the vector `w` by adding
  the vector `y` to `alpha` times the vector `x`. Parameters:
  - w Vector to store the result.
  - alpha Scalar multiplier for vector x.
  - x Vector to be scaled and added.
  - y Vector to be added.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y);

/*
  Computes the component-wise multiplication w[i] = x[i] * y[i]. This operation
  is performed for each element `i` of the vectors `x`, `y`. Parameters:
  - w Vector to store the result.
  - x First input vector.
  - y Second input vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex numbers, where complex multiplication is
  performed element-wise.
 */
PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y);
/*
  Computes the maximum of the componentwise division max = max_i abs(x[i]/y[i]).
  This operation is performed for each element `i` of the vectors `x`, `y`.

  Parameters:
  - x: Vector containing the numerators.
  - y: Vector containing the denominators.
  - max: Pointer to store the maximum result.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note:
  - If `y[i]` is zero, it is treated as 1 for the computation.
  - Supports both real and complex numbers, where the magnitude of the division
  result is considered for complex numbers.
 */
PetscErrorCode VecMaxPointwiseDivide(Vec x, Vec y, PetscReal *max);

/*
  Computes the component-wise division w[i] = x[i] / y[i]. This operation is
  performed for each element `i` of the vectors `x`, `y`. Parameters:
  - w Vector to store the result.
  - x First input vector (numerator).
  - y Second input vector (denominator).

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex numbers. Handles division by zero
  appropriately.
 */
PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y);

/*
  Begins assembling the vector.
  Parameters:
  - vec Vector to begin assembling.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Should be called after completing all calls to VecSetValues().
        Ensures all entries are stored on the correct MPI process.
 */
PetscErrorCode VecAssemblyBegin(Vec vec);

/*
  Completes assembling the vector.
  Parameters:
  - vec Vector to complete assembling.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Should be called after VecAssemblyBegin().
        Finalizes the assembly of the vector.
 */
PetscErrorCode VecAssemblyEnd(Vec vec);

PetscErrorCode VecCopyAsync_Private(Vec x, Vec y, PetscDeviceContext dctx);

/*
  Copies one vector to another.
  Parameters:
  - xin Source vector.
  - yin Destination vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecCopy(Vec xin, Vec yin);

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val);

PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available,
                                PetscReal *val);

PetscErrorCode VecNormalize(Vec x, PetscReal *val);

/*
  Computes the norm of a sequential vector.
  Parameters:
  - xin Input vector.
  - type Type of norm to compute (NORM_1, NORM_2, NORM_INFINITY).
  - z Pointer to store the computed norm value.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: NORM_FROBENIUS is same as L2 norm for vectors.
        NORM_1_AND_2 returns both L1 & L2 norms at same time.
 */
PetscErrorCode VecNorm_Seq(Vec xin, NormType type, PetscReal *z);

/*
  Copies one sequential vector `xin` to another sequential vector `yin` of the
  same size. It ensures that the destination vector `yin` has the same elements
  as the source vector `xin`. Parameters:
  - xin Source vector.
  - yin Destination vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecCopy_Seq(Vec xin, Vec yin);

/*
  Gets a read-only pointer to the vector's data array.
  Parameters:
  - x Input vector.
  - a Pointer to store the read-only array pointer.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a);

/*
  Gets a writable pointer to the vector's data array.
  Parameters:
  - x Input vector.
  - a Pointer to store the writable array pointer.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecGetArray(Vec x, PetscScalar **a);

/*
  Restores the read-only array obtained from VecGetArrayRead.
  Parameters:
  - x Input vector.
  - a Pointer to the array to be restored.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a);

/*
  Restores the array obtained from VecGetArray.
  Parameters:
  - x Input vector.
  - a Pointer to the array to be restored.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a);

/*
  Sets a single entry in a vector.
  Parameters:
  - v Vector to modify.
  - row Index of the entry to set.
  - value Value to set.
  - mode Insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecSetValue(Vec v, PetscInt i, PetscScalar va, InsertMode mode);

/*
  Inserts or adds values into a PETSc vector at specified indices.

  Parameters:
  - x: The PETSc vector where values are to be set.
  - ni: The number of indices at which values will be inserted.
  - ix: Array of indices where values will be inserted.
  - y: Array of values to be inserted.
  - iora: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: The function either inserts or adds values at specified locations in the
  vector. This operation is often used when assembling vectors in parallel
  computations.
 */
PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            const PetscScalar y[], InsertMode iora);

/*
  Inserts or adds blocks of values into a PETSc vector at specified indices.

  Parameters:
  - x: The PETSc vector where blocks of values are to be set.
  - ni: The number of blocks to be inserted or added.
  - ix: Array of block indices (in block count, not element count).
  - y: Array of values to be inserted or added, organized in block format.
  - iora: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Notes:
  - Each block is a contiguous group of elements of size equal to the vector's
  block size.
  - The function updates the vector such that x[bs * ix[i] + j] = y[bs * i + j],
    for j = 0, ..., bs-1, where bs is the block size of the vector.
  - Indices outside the range owned by the local process are ignored.
  - Calls with INSERT_VALUES and ADD_VALUES cannot be mixed without
    intervening calls to VecAssemblyBegin() and VecAssemblyEnd().
  - Negative indices in ix are ignored to facilitate handling of boundary
  conditions.

  Usage:
  This operation is particularly useful in parallel computations when dealing
  with structured data such as blocks of matrix rows or other grouped data
  structures.
 */
PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora);

// PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS
// *x_is[]);

/*
  Retrieves values from specified locations of a PETSc vector.

  Parameters:
  - x: The PETSc vector from which values are to be retrieved.
  - ni: The number of indices to retrieve.
  - ix: Array of indices to retrieve values from (in global 1D numbering).
  - y: Array where retrieved values will be stored.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Notes:
  - The function retrieves `y[i] = x[ix[i]]` for `i = 0,...,ni-1`.
  - Indices outside the local range of the vector result in `y[i]` being set
  to 0.
 */
PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            PetscScalar y[]);
/*
  Conjugates each element of the vector.
  Parameters:
  - xin Vector to be conjugated.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Handles both complex and real vectors depending on the USE_COMPLEX
  macro.
 */
PetscErrorCode VecConjugate_Seq(Vec xin);

// PetscErrorCode ISCreateStride(MPI_Comm comm, PetscInt n, PetscInt first,
//                               PetscInt step, IS *is);

/*
  Computes the norm of a subvector of a vector defined by a starting point
  and a stride. Parameters:
  - v Vector containing the subvector.
  - start Starting index of the subvector.
  - ntype Type of norm to compute (NORM_1, NORM_2, NORM_FROBENIUS,
  NORM_INFINITY, NORM_1_AND_2).
  - nrm Pointer to store the computed norm value.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: NORM_FROBENIUS is same as L2 norm for vectors.
        NORM_1_AND_2 returns both L1 & L2 norms at same time.
 */
PetscErrorCode VecStrideNorm(Vec v, PetscInt start, NormType ntype,
                             PetscReal *nrm);

/*
  Destroys a vector and frees its memory.
  Parameters:
  - v Pointer to the vector to be destroyed.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecDestroy(Vec *v);

/*
  Finalizes PETSc.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
 */
PetscErrorCode PetscFinalize(void);

/*
  Computes the Euclidean norm (L2 norm) of a vector.
  Parameters:
  - n Pointer to the number of elements in the vector.
  - x Pointer to the vector elements.
  - stride Pointer to the stride between elements in the vector.

  Returns: PetscReal The computed Euclidean norm.

  Note: For complex numbers, it computes: sqrt(sum(|x[i]|^2))
        It iterates through the vector elements with a specified `stride` and
  sums the squares of the element values. The final result is the square root of
  this sum. For real numbers, it computes: sqrt(sum(x[i] * x[i])) For complex
  numbers, it computes: sqrt(sum(|x[i]|^2))
 */
PetscReal BLASnrm2_(const PetscBLASInt *n, const PetscScalar *x,
                    const PetscBLASInt *stride);

/*
  Computes the dot product of two vectors `x` and `y`:
      result = sum(x[i] * y[i])
  It iterates through the vectors with specified strides `sx` and `sy`
  respectively. Parameters:
  - n Pointer to the number of elements in the vectors.
  - x Pointer to the first vector.
  - sx Pointer to the stride between elements in the first vector.
  - y Pointer to the second vector.
  - sy Pointer to the stride between elements in the second vector.

  Returns: PetscScalar The computed dot product.

  Note: For complex numbers, it computes: sum(x[ix] * conj(y[iy]))
 */
PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x,
                     const PetscBLASInt *sx, const PetscScalar *y,
                     const PetscBLASInt *sy);

/*
  Computes the sum of absolute values of elements in a vector `dx`:
      result = sum(|dx[i]|)
  It iterates through the vector elements with a specified stride `incx` and
  sums the absolute values of the elements. Parameters:
  - n Pointer to the number of elements in the vector.
  - dx Pointer to the vector elements.
  - incx Pointer to the stride between elements in the vector.

  Returns: PetscReal The computed sum of absolute values.

  Note: Should be called only when the scalar type is real.
 */
PetscReal BLASasum_(const PetscBLASInt *n, const PetscScalar *dx,
                    const PetscBLASInt *incx);

/*
  Copies n bytes from location b to location a.
  Parameters:
  - a Destination pointer.
  - b Source pointer.
  - n Number of bytes to copy.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Returns an error code if either `a` or `b` is NULL.

 */
PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n);

void gather_data_mpi(PetscReal *localReals, PetscReal *localImags,
                     int local_size, int N, PetscReal **globalReals,
                     PetscReal **globalImags, MPI_Comm comm, int rank,
                     int nproc);

// CIVL-specific functions used to model PETSc concepts...

/*
  Extracts the abstract vector represented by the PETSc vector

  Parameters:
  - petscVec: The PETSc vector to be converted.

  Returns:
  - $vec: The corresponding CIVL vector.
*/
$vec petscToCivlVec(Vec petscVec);

void civlToPetscVecCopy($vec in, Vec out);

/*
  Converts a CIVL vector to a PETSc vector representation.

  Parameters:
  - in: The CIVL vector that contains data to populate the PETSc vector.

  Returns:
  - Vec: The corresponding PETSc vector.
*/
Vec civlToPetscVec($vec in, int n, MPI_Comm comm);

/*
  Prints the contents of a sequential vector.
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Prints complex numbers in the form (a + bi) if USE_COMPLEX is
  defined, otherwise prints real numbers.
 */
void vecprint_seq(const char *name, Vec vin);

/*
  Prints the contents of a MPI vector.
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Prints complex numbers in the form (a + bi) if USE_COMPLEX is
  defined, otherwise prints real numbers.
 */
void vecprint_mpi(const char *name, Vec vin);

/*
  Creates a new sequential vector.
  Parameters:
  - n Number of elements in the vector.
  - data Pointer to initial data for the vector. If NULL, vector is initialized
  with zeros.

  Returns: Vec The newly created vector.

  Note: Allocates memory for the vector structure, its map, and its data.
        Handles both real and complex data based on USE_COMPLEX definition.
 */
Vec vec_create_seq(int n, PetscScalar *data);

/*
  Checks if two sequential vectors are equal.
  Parameters:
  - vec1 First vector for comparison.
  - vec2 Second vector for comparison.

  Returns: bool True if vectors are equal, false otherwise.

  Note: Compares vector sizes, block sizes, and all elements.
        For complex numbers, compares both real and imaginary parts.
 */
bool vec_eq_seq(Vec vec1, Vec vec2);

/*
  Destroys a sequential vector and frees its memory.
  Parameters:
  - vec Vector to be destroyed.

  Note: Frees memory for the vector's data, map, and the vector structure
  itself.
 */
void vec_destroy_seq(Vec vec);

// Now that Vec is forward-declared, we can define VecOps
typedef struct _VecOps *VecOps;

struct _VecOps {
  PetscErrorCode (*norm)(Vec, NormType, PetscReal *); /* z = sqrt(x^H * x) */
  PetscErrorCode (*maxpointwisedivide)(Vec, Vec,
                                       PetscReal *); /* m = max abs(x ./ y) */
  PetscErrorCode (*dot)(Vec, Vec, PetscScalar *);    /* z =  */
  PetscErrorCode (*max)(Vec, PetscInt *,
                        PetscReal *); /* z = max(x); idx=index of max(x) */
  PetscErrorCode (*min)(Vec, PetscInt *,
                        PetscReal *); /* z = min(x); idx=index of min(x) */
  PetscErrorCode (*tdot)(Vec, Vec, PetscScalar *); /* x'*y */
  PetscErrorCode (*scale)(Vec, PetscScalar);       /* x = alpha * x   */
  PetscErrorCode (*set)(Vec, PetscScalar);         /* y = alpha  */
  PetscErrorCode (*axpy)(Vec, PetscScalar, Vec);   /* y = y + alpha * x */
  PetscErrorCode (*aypx)(Vec, PetscScalar, Vec);   /* y = x + alpha * y */
  PetscErrorCode (*axpby)(Vec, PetscScalar, PetscScalar,
                          Vec); /* y = alpha * x + beta * y*/
  PetscErrorCode (*axpbypcz)(Vec, PetscScalar, PetscScalar, PetscScalar, Vec,
                             Vec); /* z = alpha * x + beta *y + gamma *z*/
  PetscErrorCode (*waxpy)(Vec, PetscScalar, Vec, Vec); /* w = y + alpha * x */
  PetscErrorCode (*copy)(Vec, Vec);                    /* y = x */
  PetscErrorCode (*setvalues)(Vec, PetscInt, const PetscInt[],
                              const PetscScalar[], InsertMode);
  PetscErrorCode (*getvalues)(Vec, PetscInt, const PetscInt[], PetscScalar[]);
  PetscErrorCode (*setvaluesblocked)(Vec, PetscInt, const PetscInt[],
                                     const PetscScalar[], InsertMode);
  PetscErrorCode (*mtdot)(Vec, PetscInt, const Vec[],
                          PetscScalar *); /* z[j] = x dot y[j] */
  PetscErrorCode (*maxpy)(Vec, PetscInt, const PetscScalar *,
                          Vec *); /* y = y + alpha[j] x[j] */
  PetscErrorCode (*maxpby)(Vec, PetscInt, const PetscScalar *, PetscScalar,
                           Vec *);  /* y = beta y + alpha[j] x[j] */
  PetscErrorCode (*copy)(Vec, Vec); /* y = x */
  // PetscErrorCode (*concatenate)(PetscInt, const Vec[], Vec *, IS *[]);
};

/* struct _ISOps {
  PetscErrorCode (*duplicate)(IS, IS *);
}; */

#endif

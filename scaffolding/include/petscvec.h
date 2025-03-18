#ifndef _PETSCVEC_H
#define _PETSCVEC_H
#include <float.h>
#include <limits.h>
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
#define PetscConj(a) scalar_conj(a)
#define PetscImaginaryPart(a) ((a).imag)
#define PetscRealPart(a) ((a).real)
#define PetscAbsScalar(a) scalar_abs(a)
#else
typedef PetscReal PetscScalar;
#define PetscConj(a) scalar_conj(a)
#define PetscImaginaryPart(a) ((PetscReal)(0))
#define PetscRealPart(a) (a)
#define PetscAbsScalar(a) scalar_abs(a)
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

/*
  PetscCopyMode - Specifies how an array or `PetscObject` is copied or retained
  by an aggregate `PetscObject`.

  Parameters:
  - PETSC_COPY_VALUES  The array values are copied into new space. The user is
  free to reuse or delete the passed-in array.
  - PETSC_OWN_POINTER  The array values are not copied. The object takes
  ownership of the array and will free it later. The user cannot modify or
  delete the array. The array must have been allocated with `PetscMalloc()`.
  - PETSC_USE_POINTER  The array values are not copied. The object uses the
  array but does not take ownership of it. The user must ensure that the array
  remains valid for the object's lifetime and must free it after use.
 */
typedef enum {
  PETSC_COPY_VALUES,
  PETSC_OWN_POINTER,
  PETSC_USE_POINTER
} PetscCopyMode;

#define NUM_NORM_TYPES 5

#define PETSCHEADER(ObjectOps)                                                 \
  struct _p_PetscObject hdr;                                                   \
  ObjectOps ops[1]

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

// A component of a Vec struct specifying how the vector is
// distributed across processes.
typedef struct map_s {
  PetscInt n;            // local_size
  PetscInt N;            // global_size
  PetscInt rstart, rend; // local start, local end + 1
  PetscInt bs;           // for now assuming the block size as 1
  PetscInt nproc;
} *SimpleMap;

typedef struct _p_IS *IS;
/* IS - Abstract PETSc object used for efficient indexing into vector and
 * matrices */
struct _p_IS {
  PETSCHEADER(struct _ISOps);
  SimpleMap map;
  PetscInt max, min; /* range of possible values */
  void *data;
  PetscInt *total, *nonlocal; /* local representation of ALL indices across the
                                 comm as well as the nonlocal part. */
  PetscInt
      local_offset; /* offset to the local part within the total index set */
  IS complement;    /* IS wrapping nonlocal indices. */
  PetscBool info_permanent[2][IS_INFO_MAX]; /* whether local / global properties
                                               are permanent */
  ISInfoBool info[2][IS_INFO_MAX];          /* local / global properties */
  MPI_Comm comm;
};

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

/* In some header or a place before math.h is included: */
#ifndef __isfinitel
#define __isfinitel(x) 1
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
// #define PETSC_MAX_REAL 1'000'000

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

#define IS_CLASSID 124

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
  (*(r1) = malloc((m1) * sizeof(*(r1))), PETSC_SUCCESS)

PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj, PetscInt id,
                                              PetscReal *data, PetscBool *flag);

PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj, PetscInt id,
                                              PetscReal data);

PetscErrorCode PetscInfo_Private(PetscObject obj, const char message[]);

#define PetscInfo(A, ...) PetscInfo_Private(((PetscObject)A), __VA_ARGS__)

PetscBool PetscIsInfOrNanReal(PetscReal v);

int __isfinited(double x);

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

#ifdef PETSC_USE_MIXED_PRECISION
#undef PETSC_USE_MIXED_PRECISION
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

#define PetscCallMPI(x) (x)

#define MPIU_Allreduce(a, b, c, d, e, fcomm)                                   \
  MPI_Allreduce((a), (b), (c), (d), (e), (fcomm))

/* #define MPIU_Allreduce(a, b, c, d, e, fcomm) \
  do {                                                                         \
    int ierr = MPI_Allreduce((a), (b), (c), (d), (e), (fcomm));                \
    if (ierr != MPI_SUCCESS) {                                                 \
      fprintf(stderr, "Error in MPI_Allreduce: %d\n", ierr);                   \
      return ierr;                                                             \
    }                                                                          \
  } while (0) */

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

#define PetscFree(a)                                                           \
  do {                                                                         \
    if (!(a))                                                                  \
      return 0;                                                                \
    free(a);                                                                   \
    (a) = NULL;                                                                \
  } while (0)

#define VecNorm_SeqFn(a, b, c) VecNorm_Seq(a, b, c)

static int VecMax_Seq_GT(PetscReal l, PetscReal r) { return (l > r) ? 1 : 0; }

static int VecMin_Seq_LT(PetscReal l, PetscReal r) { return (l < r) ? 1 : 0; }

#ifndef MPI_IN_PLACE
#define MPI_IN_PLACE (void *)-1
#endif

#define MPIU_SUM MPI_SUM

#define MPIU_SCALAR MPIU_REAL

#define PetscMax(a, b) (((a) < (b)) ? (b) : (a))

#define PetscDesignatedInitializer(name, ...) .name = __VA_ARGS__

#ifdef USE_VEC_MTDOT
#define VecXDot_SeqFn(a, b, c) VecXDot_Seq_Private(a, b, c, BLASdotu_)
#endif

#define BLASfn(a, b, c, d, e) BLASdot_(a, b, c, d, e)

#ifdef USE_VEC_TDOT
/*
  pay close attention!!! a and b are SWAPPED here so that the eventual
  BLAS call is dot(&bn, xa, &one, ya, &one)
*/
#define VecXDot_SeqFn(a, b, c) VecXDot_Seq_Private(b, a, c, BLASdotu_)
#undef BLASfn
#define BLASfn(a, b, c, d, e) BLASdotu_(a, b, c, d, e)
#endif

#define PETSC_HAVE_MPIUNI 0

#ifdef PETSC_USE_FORTRAN_KERNEL_AYPX
#undef PETSC_USE_FORTRAN_KERNEL_AYPX
#endif

#ifdef PETSC_USE_FORTRAN_KERNEL_WAXPY
#undef PETSC_USE_FORTRAN_KERNEL_WAXPY
#endif

#ifdef PETSC_HAVE_PRAGMA_DISJOINT
#undef PETSC_HAVE_PRAGMA_DISJOINT
#endif

#ifndef PETSC_RESTRICT
#define PETSC_RESTRICT restrict
#endif

#define MPIU_REAL_INT MPI_DOUBLE_INT
#define MPIU_MAXLOC MPI_MAXLOC
#define MPIU_MINLOC MPI_MINLOC
#define MPIU_MAX MPI_MAX
#define MPIU_MIN MPI_MIN

#ifndef PETSC_MAX_INT
#define PETSC_MAX_INT INT_MAX
#endif

#define PetscArrayzero(arr, cnt)                                               \
  do {                                                                         \
    size_t _i;                                                                 \
    for (_i = 0; _i < (cnt); _i++) {                                           \
      (arr)[_i] = scalar_zero;                                                 \
    }                                                                          \
  } while (0)

#define PetscKernelAXPY(U, a1, p1, n)                                          \
  do {                                                                         \
    const PetscInt _n = n;                                                     \
    const PetscScalar _a1 = a1;                                                \
    const PetscScalar *PETSC_RESTRICT _p1 = p1;                                \
    PetscScalar *PETSC_RESTRICT _U = U;                                        \
    PetscInt __i;                                                              \
    for (__i = 0; __i < _n - 1; __i += 2) {                                    \
      PetscScalar __s1 = scalar_mul(_a1, _p1[__i]);                            \
      PetscScalar __s2 = scalar_mul(_a1, _p1[__i + 1]);                        \
      __s1 = scalar_add(__s1, _U[__i]);                                        \
      __s2 = scalar_add(__s2, _U[__i + 1]);                                    \
      _U[__i] = __s1;                                                          \
      _U[__i + 1] = __s2;                                                      \
    }                                                                          \
    if (_n & 0x1)                                                              \
      _U[__i] = scalar_add(_U[__i], scalar_mul(_a1, _p1[__i]));                \
  } while (0)

#define PetscKernelAXPY2(U, a1, a2, p1, p2, n)                                 \
  do {                                                                         \
    const PetscInt _n = n;                                                     \
    const PetscScalar _a1 = a1;                                                \
    const PetscScalar _a2 = a2;                                                \
    const PetscScalar *PETSC_RESTRICT _p1 = p1;                                \
    const PetscScalar *PETSC_RESTRICT _p2 = p2;                                \
    PetscScalar *PETSC_RESTRICT _U = U;                                        \
    for (PetscInt __i = 0; __i < _n; __i++) {                                  \
      PetscScalar __s =                                                        \
          scalar_add(scalar_mul(_a1, _p1[__i]), scalar_mul(_a2, _p2[__i]));    \
      _U[__i] = scalar_add(_U[__i], __s);                                      \
    }                                                                          \
  } while (0)

#define PetscKernelAXPY3(U, a1, a2, a3, p1, p2, p3, n)                         \
  do {                                                                         \
    const PetscInt _n = n;                                                     \
    const PetscScalar _a1 = a1;                                                \
    const PetscScalar _a2 = a2;                                                \
    const PetscScalar _a3 = a3;                                                \
    const PetscScalar *PETSC_RESTRICT _p1 = p1;                                \
    const PetscScalar *PETSC_RESTRICT _p2 = p2;                                \
    const PetscScalar *PETSC_RESTRICT _p3 = p3;                                \
    PetscScalar *PETSC_RESTRICT _U = U;                                        \
    for (PetscInt __i = 0; __i < _n; __i++) {                                  \
      PetscScalar __s = scalar_add(                                            \
          scalar_add(scalar_mul(_a1, _p1[__i]), scalar_mul(_a2, _p2[__i])),    \
          scalar_mul(_a3, _p3[__i]));                                          \
      _U[__i] = scalar_add(_U[__i], __s);                                      \
    }                                                                          \
  } while (0)

#define PetscKernelAXPY4(U, a1, a2, a3, a4, p1, p2, p3, p4, n)                 \
  do {                                                                         \
    const PetscInt _n = n;                                                     \
    const PetscScalar _a1 = a1;                                                \
    const PetscScalar _a2 = a2;                                                \
    const PetscScalar _a3 = a3;                                                \
    const PetscScalar _a4 = a4;                                                \
    const PetscScalar *PETSC_RESTRICT _p1 = p1;                                \
    const PetscScalar *PETSC_RESTRICT _p2 = p2;                                \
    const PetscScalar *PETSC_RESTRICT _p3 = p3;                                \
    const PetscScalar *PETSC_RESTRICT _p4 = p4;                                \
    PetscScalar *PETSC_RESTRICT _U = U;                                        \
    for (PetscInt __i = 0; __i < _n; __i++) {                                  \
      PetscScalar __s =                                                        \
          scalar_add(scalar_add(scalar_add(scalar_mul(_a1, _p1[__i]),          \
                                           scalar_mul(_a2, _p2[__i])),         \
                                scalar_mul(_a3, _p3[__i])),                    \
                     scalar_mul(_a4, _p4[__i]));                               \
      _U[__i] = scalar_add(_U[__i], __s);                                      \
    }                                                                          \
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
  VecSetUp - Initializes the vector type and sets up internal data structures.

  Parameters:
  - v Vector to be set up.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: If the vector type is not set, it is initialized based on the number of
  processes.
*/
PetscErrorCode VecSetUp(Vec v);

/*
  VecGetSubVector - Extracts a subvector from a given vector based on an index
  set.

  Parameters:
  - X  Input vector.
  - is Index set defining the portion of `X` to extract.
  - Y  Output subvector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Converts `X` to `$vec`, extracts the required subsequence using
  `$vec_subseq`, and converts the result back to a PETSc `Vec` using
  `civlToPetscVecCopy`.
 */
PetscErrorCode VecGetSubVector(Vec X, IS is, Vec *Y);

/*
  VecRestoreSubVector - Restores a subvector obtained using VecGetSubVector.

  Parameters:
  - X  Original vector from which the subvector was obtained.
  - is Index set representing the subset of `X`.
  - Y  Subvector to be restored.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: If the subvector's state has not changed, this function simply destroys
  it.
 */
PetscErrorCode VecRestoreSubVector(Vec X, IS is, Vec *Y);

/*
  ISDestroy - Destroys an index set and deallocates its resources.

  Parameters:
  - is The index set to be destroyed.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: This function frees the allocated memory for index sets, including
        local and nonlocal arrays.
 */
PetscErrorCode ISDestroy(IS *is);

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

PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscScalar *z);

PetscErrorCode VecDot_MPI(Vec xin, Vec yin, PetscScalar *z);

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

PetscErrorCode VecTDot_MPI(Vec xin, Vec yin, PetscScalar *z);

PetscErrorCode VecTDot_Seq(Vec x, Vec y, PetscScalar *val);

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

PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[],
                            PetscScalar *z);

PetscErrorCode VecMTDot_Seq(Vec xin, PetscInt nv, const Vec y[],
                            PetscScalar *z);

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

PetscErrorCode VecMax_MPI(Vec xin, PetscInt *idx, PetscReal *z);

PetscErrorCode VecMax_Seq(Vec xin, PetscInt *idx, PetscReal *z);

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

PetscErrorCode VecMin_MPI(Vec xin, PetscInt *idx, PetscReal *z);

PetscErrorCode VecMin_Seq(Vec xin, PetscInt *idx, PetscReal *z);

typedef struct _p_PetscDeviceContext *PetscDeviceContext;

PetscErrorCode VecScaleAsync_Private(Vec x, PetscScalar alpha,
                                     PetscDeviceContext dctx);

PetscErrorCode VecSetAsync_Private(Vec x, PetscScalar alpha,
                                   PetscDeviceContext dctx);

PetscErrorCode VecSet_Seq(Vec xin, PetscScalar alpha);

/*
  Scales a vector by multiplying each element by a scalar.
  Parameters:
  - x Vector to scale.
  - alpha Scalar to multiply by.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars.
 */
PetscErrorCode VecScale(Vec x, PetscScalar alpha);

PetscErrorCode VecScale_Seq(Vec xin, PetscScalar alpha);

/*
  Compares two vectors for equality.
  Parameters:
  - vec1: First vector to compare.
  - vec2: Second vector to compare.
  - flg: Pointer to a boolean flag that will be set to `PETSC_TRUE` if the
  vectors are equal, `PETSC_FALSE` otherwise.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: This function checks if the vectors have the same dimensions and
  block size, and if their elements are equal. Supports both real and
  complex vectors.
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

PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv, const PetscScalar *alpha,
                            Vec *y);

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

PetscErrorCode VecAXPY_Seq(Vec yin, PetscScalar alpha, Vec xin);

PetscErrorCode VecAXPBYAsync_Private(Vec y, PetscScalar alpha, PetscScalar beta,
                                     Vec x, PetscDeviceContext dctx);

/*
  Computes the linear combination of two vectors `x` and `y`:
      y = alpha * x + beta * y
  It iterates through the vectors with specified strides `sx` and `sy`
  respectively. Parameters:
  - alpha Scalar multiplier for the first vector `x`.
  - x Pointer to the first vector.
  - beta Scalar multiplier for the second vector `y`.
  - y Pointer to the second vector.

  Returns: This function updates the vector `y` in place.

  Note: This function performs element-wise operations and assumes that the
  vectors are of the same length.
*/
PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x);

PetscErrorCode VecAXPBY_Seq(Vec yin, PetscScalar a, PetscScalar b, Vec xin);

PetscErrorCode VecAXPBYPCZAsync_Private(Vec z, PetscScalar alpha,
                                        PetscScalar beta, PetscScalar gamma,
                                        Vec x, Vec y, PetscDeviceContext dctx);

/*
  Computes the linear combination of three vectors `x`, `y`, and `z`:
      w = alpha * x + beta * y + gamma * z
  It iterates through the vectors with specified strides `sx`, `sy`, and `sz`
  respectively. Parameters:
  - alpha Scalar multiplier for the first vector `x`.
  - x Pointer to the first vector.
  - beta Scalar multiplier for the second vector `y`.
  - y Pointer to the second vector.
  - gamma Scalar multiplier for the third vector `z`.
  - z Pointer to the third vector.

  Returns: This function updates the vector `z` in place.

  Note: This function performs element-wise operations and assumes that the
  vectors are of the same length.
*/
PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y);

PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscScalar alpha, PetscScalar beta,
                               PetscScalar gamma, Vec xin, Vec yin);

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

PetscErrorCode VecAYPX_Seq(Vec yin, PetscScalar alpha, Vec xin);

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

PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha, Vec xin, Vec yin);
/*
  Computes the component-wise multiplication w[i] = x[i] * y[i]. This
  operation is performed for each element `i` of the vectors `x`, `y`.
  Parameters:
  - w Vector to store the result.
  - x First input vector.
  - y Second input vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex numbers, where complex multiplication
  is performed element-wise.
 */
PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y);

PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin, Vec yin, PetscReal *max);
/*
  Computes the maximum of the componentwise division max = max_i
  abs(x[i]/y[i]). This operation is performed for each element `i` of the
  vectors `x`, `y`.

  Parameters:
  - x: Vector containing the numerators.
  - y: Vector containing the denominators.
  - max: Pointer to store the maximum result.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note:
  - If `y[i]` is zero, it is treated as 1 for the computation.
  - Supports both real and complex numbers, where the magnitude of the
  division result is considered for complex numbers.
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

PetscErrorCode VecCopy_Seq(Vec xin, Vec yin);

/* Utility function to compute the PETSc norm of a CIVL vector */
void $petsc_norm($vec vec, NormType type, PetscReal *result);

/* Returns string representation of the PETSc norm type */
char *$petsc_norm_name(NormType type);

PetscErrorCode VecNorm_Seq(Vec xin, NormType type, PetscReal *z);

PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscReal *z);

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
  Gets a read-only pointer to the vector's data array.
  Parameters:
  - x Input vector.
  - a Pointer to store the read-only array pointer.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a);

PetscErrorCode VecGetArrayWrite(Vec x, PetscScalar **a);

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

PetscErrorCode VecRestoreArrayWrite(Vec x, PetscScalar **a);

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

PetscErrorCode VecSetValues_MPI(Vec xin, PetscInt ni, const PetscInt ix[],
                                const PetscScalar y[], InsertMode addv);

PetscErrorCode VecSetValues_Seq(Vec x, PetscInt ni, const PetscInt ix[],
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
  - Each block is a contiguous group of elements of size equal to the
  vector's block size.
  - The function updates the vector such that x[bs * ix[i] + j] = y[bs * i +
  j], for j = 0, ..., bs-1, where bs is the block size of the vector.
  - Indices outside the range owned by the local process are ignored.
  - Calls with INSERT_VALUES and ADD_VALUES cannot be mixed without
    intervening calls to VecAssemblyBegin() and VecAssemblyEnd().
  - Negative indices in ix are ignored to facilitate handling of boundary
  conditions.

  Usage:
  This operation is particularly useful in parallel computations when
  dealing with structured data such as blocks of matrix rows or other
  grouped data structures.
 */
PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora);

PetscErrorCode VecSetValuesBlocked_MPI(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora);

PetscErrorCode VecSetValuesBlocked_Seq(Vec x, PetscInt ni, const PetscInt ix[],
                                       const PetscScalar y[], InsertMode iora);
/*
  ISCreateGeneral - Creates an index set from an array of integers.

  Parameters:
  - comm  The MPI communicator.
  - n     The number of indices.
  - idx   The array of indices.
  - mode  Copy mode (`PETSC_COPY_VALUES`, `PETSC_OWN_POINTER`, or
  `PETSC_USE_POINTER`).
  - is    The newly created index set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: This function allocates and initializes an `IS` structure.
 */
PetscErrorCode ISCreateGeneral(MPI_Comm comm, PetscInt n, const PetscInt idx[],
                               PetscCopyMode mode, IS *is);

PetscErrorCode ISGetSize(IS is, PetscInt *size);
/*
  Creates a data structure for an index set containing a list of evenly
  spaced integers.

  Parameters:
  - comm: the MPI communicator.
  - n: the length of the locally owned portion of the index set.
  - first: the first element of the locally owned portion of the index set.
  - step: the change to the next index.
  - is: the new index set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
*/
PetscErrorCode ISCreateStride(MPI_Comm comm, PetscInt n, PetscInt first,
                              PetscInt step, IS *is);

/*
  Returns the local (processor) length of an index set.

  Parameters:
  - is: The index set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Output:
  - size: The local size.
  */
PetscErrorCode ISGetLocalSize(IS is, PetscInt *size);

/*
  Creates a new vector that is a vertical concatenation of all the given array
  of vectors in the order they appear in the array. The concatenated vector
  resides on the same communicator and is the same type as the source vectors.

  Parameters:
  - nx: Number of vectors to be concatenated.
  - X: Array containing the vectors to be concatenated in the order of
  concatenation.

  Output Parameters:
  - Y: Concatenated vector.
  - x_is: Array of index sets corresponding to the concatenated components of Y
  (pass NULL if not needed).

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[]);

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

PetscErrorCode VecGetValues_MPI(Vec xin, PetscInt ni, const PetscInt ix[],
                                PetscScalar y[]);

PetscErrorCode VecGetValues_Seq(Vec xin, PetscInt ni, const PetscInt ix[],
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
  Scales a vector `x` by a scalar `alpha`:
      x[i] = alpha * x[i]
  It iterates through the vector with a specified stride `sx`.
  Parameters:
  - n Pointer to the number of elements in the vector.
  - alpha Pointer to the scalar value to scale the vector.
  - x Pointer to the vector to be scaled.
  - sx Pointer to the stride between elements in the vector.

  Returns: void

  Note: This function modifies the input vector `x` in place.
*/
PetscErrorCode BLASscal_(const PetscBLASInt *n, const PetscScalar *alpha,
                         PetscScalar *x, const PetscBLASInt *incx);

/*
Computes the operation y := alpha * x + y, where `x` and `y` are vectors and
`alpha` is a scalar. It iterates through the vectors with specified strides `sx`
and `sy` respectively.

Parameters:
- n Pointer to the number of elements in the vectors.
- alpha Pointer to the scalar multiplier for the vector `x`.
- x Pointer to the first vector.
- sx Pointer to the stride between elements in the first vector.
- y Pointer to the second vector.
- sy Pointer to the stride between elements in the second vector.
*/
PetscErrorCode BLASaxpy_(const PetscBLASInt *n, const PetscScalar *alpha,
                         const PetscScalar *x, const PetscBLASInt *incx,
                         PetscScalar *y, const PetscBLASInt *incy);
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

/*Blas routines*/
/*
  Computes the dot product of two vectors `x` and `y`:
      result = sum(PetscConj(x[i]) * y[i])
  It iterates through the vectors with specified strides `sx` and `sy`
  respectively.

  Parameters:
  - n Pointer to the number of elements in the vectors.
  - x Pointer to the first vector.
  - sx Pointer to the stride between elements in the first vector.
  - y Pointer to the second vector.
  - sy Pointer to the stride between elements in the second vector.

  Returns: PetscScalar The computed dot product.

  Note: For complex numbers, it computes: sum(PetscConj(x[ix]) * y[iy])
*/
PetscScalar BLASdot_(const PetscBLASInt *n, const PetscScalar *x,
                     const PetscBLASInt *sx, const PetscScalar *y,
                     const PetscBLASInt *sy);

// Now that Vec is forward-declared, we can define VecOps
typedef struct _VecOps *VecOps;

struct _VecOps {
  PetscErrorCode (*norm)(Vec, NormType, PetscReal *); // z = sqrt(x^H * x)
  PetscErrorCode (*maxpointwisedivide)(Vec, Vec,
                                       PetscReal *); // m = max abs(x ./ y)
  PetscErrorCode (*dot)(Vec, Vec, PetscScalar *);    // z =
  PetscErrorCode (*max)(Vec, PetscInt *,
                        PetscReal *); // z = max(x); idx=index of max(x)
  PetscErrorCode (*min)(Vec, PetscInt *,
                        PetscReal *); // z = min(x); idx=index of min(x)
  PetscErrorCode (*tdot)(Vec, Vec, PetscScalar *); // x'*y
  PetscErrorCode (*scale)(Vec, PetscScalar);       // x = alpha * x
  PetscErrorCode (*set)(Vec, PetscScalar);         // y = alpha
  PetscErrorCode (*axpy)(Vec, PetscScalar, Vec);   // y = y + alpha * x
  PetscErrorCode (*aypx)(Vec, PetscScalar, Vec);   // y = x + alpha * y
  PetscErrorCode (*axpby)(Vec, PetscScalar, PetscScalar,
                          Vec); // y = alpha * x + beta * y
  PetscErrorCode (*axpbypcz)(Vec, PetscScalar, PetscScalar, PetscScalar, Vec,
                             Vec); // z = alpha * x + beta *y + gamma *z
  PetscErrorCode (*waxpy)(Vec, PetscScalar, Vec, Vec); // w = y + alpha * x
  PetscErrorCode (*copy)(Vec, Vec);                    // y = x
  PetscErrorCode (*setvalues)(Vec, PetscInt, const PetscInt[],
                              const PetscScalar[], InsertMode);
  PetscErrorCode (*getvalues)(Vec, PetscInt, const PetscInt[], PetscScalar[]);
  PetscErrorCode (*setvaluesblocked)(Vec, PetscInt, const PetscInt[],
                                     const PetscScalar[], InsertMode);
  PetscErrorCode (*mtdot)(Vec, PetscInt, const Vec[],
                          PetscScalar *); // z[j] = x dot y[j]
  PetscErrorCode (*maxpy)(Vec, PetscInt, const PetscScalar *,
                          Vec *); // y = y + alpha[j] x[j]
  PetscErrorCode (*maxpby)(Vec, PetscInt, const PetscScalar *, PetscScalar,
                           Vec *); // y = beta y + alpha[j] x[j]
  PetscErrorCode (*concatenate)(PetscInt, const Vec[], Vec *, IS *[]);
  PetscErrorCode (*getsubvector)(Vec, IS, Vec *);
  PetscErrorCode (*restorearray)(Vec, PetscScalar **); /* restore data array */
  PetscErrorCode (*restorearraywrite)(Vec, PetscScalar **);
  PetscErrorCode (*getarraywrite)(Vec, PetscScalar **);
};

// Helper that assigns MPI-version function pointers
static inline void SetOps_MPI(Vec vec) {
  vec->ops->norm = VecNorm_MPI;
  vec->ops->maxpointwisedivide = VecMaxPointwiseDivide_Seq;
  vec->ops->dot = VecDot_MPI;
  vec->ops->max = VecMax_MPI;
  vec->ops->min = VecMin_MPI;
  vec->ops->tdot = VecTDot_MPI;
  vec->ops->scale = VecScale_Seq;
  vec->ops->restorearraywrite = VecRestoreArrayWrite;
  vec->ops->getarraywrite = VecGetArrayWrite;
  vec->ops->set = VecSet_Seq;
  vec->ops->axpy = VecAXPY_Seq;
  vec->ops->aypx = VecAYPX_Seq;
  vec->ops->axpby = VecAXPBY_Seq;
  vec->ops->axpbypcz = VecAXPBYPCZ_Seq;
  vec->ops->waxpy = VecWAXPY_Seq;
  vec->ops->copy = VecCopy_Seq;
  vec->ops->mtdot = VecMTDot_MPI;
  vec->ops->maxpy = VecMAXPY_Seq;
  vec->ops->maxpby = NULL;

  vec->ops->concatenate = NULL;
  // vec->ops->getsubvector = NULL;
  vec->ops->setvalues = VecSetValues_MPI;
  vec->ops->setvaluesblocked = VecSetValuesBlocked_MPI;
  vec->ops->getvalues = VecGetValues_MPI;
}

// Helper that assigns sequential-version function pointers
static inline void SetOps_Seq(Vec vec) {
  vec->ops->norm = VecNorm_Seq;
  vec->ops->maxpointwisedivide = VecMaxPointwiseDivide_Seq;
  vec->ops->dot = VecDot_Seq;
  vec->ops->max = VecMax_Seq;
  vec->ops->min = VecMin_Seq;
  vec->ops->tdot = VecTDot_Seq;
  vec->ops->scale = VecScale_Seq;
  vec->ops->restorearraywrite = VecRestoreArrayWrite;
  vec->ops->getarraywrite = VecGetArrayWrite;
  vec->ops->set = VecSet_Seq;
  vec->ops->axpy = VecAXPY_Seq;
  vec->ops->aypx = VecAYPX_Seq;
  vec->ops->axpby = VecAXPBY_Seq;
  vec->ops->axpbypcz = VecAXPBYPCZ_Seq;
  vec->ops->waxpy = VecWAXPY_Seq;
  vec->ops->copy = VecCopy_Seq;
  vec->ops->mtdot = VecMTDot_Seq;
  vec->ops->maxpy = VecMAXPY_Seq;
  vec->ops->maxpby = NULL;

  vec->ops->concatenate = NULL;
  // vec->ops->getsubvector = NULL;
  vec->ops->setvalues = VecSetValues_Seq;
  vec->ops->setvaluesblocked = VecSetValuesBlocked_Seq;
  vec->ops->getvalues = VecGetValues_Seq;
}

// static struct _VecOps DvOps = {PetscDesignatedInitializer(norm,
// VecNorm_MPI)};
typedef struct _ISOps *_ISOps;

struct _ISOps {
  PetscErrorCode (*duplicate)(IS, IS *);
  /*PetscErrorCode (*getindices)(IS, const PetscInt *[]);
  PetscErrorCode (*restoreindices)(IS, const PetscInt *[]);
  PetscErrorCode (*invertpermutation)(IS, PetscInt, IS *);
  PetscErrorCode (*sort)(IS);
  PetscErrorCode (*sortremovedups)(IS);
  PetscErrorCode (*sorted)(IS, PetscBool *);
  PetscErrorCode (*destroy)(IS);
  PetscErrorCode (*view)(IS, PetscViewer);
  PetscErrorCode (*load)(IS, PetscViewer);
  PetscErrorCode (*copy)(IS, IS);
  PetscErrorCode (*togeneral)(IS);
  PetscErrorCode (*oncomm)(IS, MPI_Comm, PetscCopyMode, IS *);
  PetscErrorCode (*setblocksize)(IS, PetscInt);
  PetscErrorCode (*contiguous)(IS, PetscInt, PetscInt, PetscInt *, PetscBool *);
  PetscErrorCode (*locate)(IS, PetscInt, PetscInt *);
  PetscErrorCode (*sortedlocal)(IS, PetscBool *);
  PetscErrorCode (*sortedglobal)(IS, PetscBool *);
  PetscErrorCode (*uniquelocal)(IS, PetscBool *);
  PetscErrorCode (*uniqueglobal)(IS, PetscBool *);
  PetscErrorCode (*permlocal)(IS, PetscBool *);
  PetscErrorCode (*permglobal)(IS, PetscBool *);
  PetscErrorCode (*intervallocal)(IS, PetscBool *);
  PetscErrorCode (*intervalglobal)(IS, PetscBool *);*/
};

#endif

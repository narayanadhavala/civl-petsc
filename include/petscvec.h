#ifndef _PETSCVEC_H
#define _PETSCVEC_H

/*
 * Filename : petscvec.h
 * Author   : Venkata Dhavala
 * Created  : 2024-03-22
 * Modified : 2025-04-13
 *
 * This header file declares the interfaces for PETSc vector operations,
 * providing the function prototypes, macros, and data structure definitions
 * necessary for the PETSc vector functionality.
 *
 * It includes:
 *   - Stub function declarations for key PETSc vector operations.
 *   - Integration points for BLAS routines to support common vector
 * computations.
 *   - CIVL-specific declarations to facilitate the bridging of PETSc vectors
 *     with CIVL data types.
 */

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
typedef STYPE PetscScalar;
#define PetscConj(a) scalar_conj(a)
#define PetscImaginaryPart(a) scalar_imag(a)
#define PetscRealPart(a) scalar_real(a)
#define PetscAbsScalar(a) scalar_abs(a)

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

// Definition of struct _p_PetscObject and its typedef PetscObject
struct _p_PetscObject;
typedef struct _p_PetscObject *PetscObject;

// Definition of PetscObjectList as a pointer to struct _n_PetscObjectList
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

/* Definition of struct _p_PetscObject
   Represents a PETSc object with metadata, state tracking, and composed data
   arrays.
   - Extracted from petscimpl.h */
struct _p_PetscObject {
  PetscClassId classid;
  const char *type_name;
  const char *class_name;
  PetscObjectList olist;
  PetscInt state;              // current state
  PetscInt *realcomposedstate; // Array of states
  PetscReal *realcomposeddata; // Array of data
};

/* Definition of struct map_s
  Represents a simple mapping structure for PETSc vectors, including local
  and global sizes, range, block size, and number of processes.
  - Extracted from vecimpl.h */
typedef struct map_s {
  PetscInt n;            // local_size
  PetscInt N;            // global_size
  PetscInt rstart, rend; // local start, local end + 1
  PetscInt bs;           // for now assuming the block size as 1
  PetscInt nproc;
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

typedef struct _ISOps *_ISOps;

// PetscViewer is a datatype representing an object used for viewing PETSc
// objects
// TODO: why the strange name _p_...?
typedef struct _p_PetscViewer *PetscViewer;

// PetscViewerFormat is an enum representing different formats for PetscViewer
typedef enum {
  PETSC_VIEWER_DEFAULT,
  PETSC_VIEWER_STDOUT_SELF
} PetscViewerFormat;

// Defines the viewer data structure.
struct _p_PetscViewer {
  PetscViewerFormat format;
  int iformat;
  void *data;
};

typedef struct {
  PetscReal val;
  PetscInt index;
} VecLocation;

/*   _p_PetscDeviceContext - Internal structure to manage device context,
  including solver contexts, stream dependencies, and child context management.
   - Extracted from deviceimpl.h */
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

// PetscErrorCodes extracted from petscsystypes.h
#define PETSC_SUCCESS ((PetscErrorCode)0)
#define PETSC_ERR_ARG_OUTOFRANGE ((PetscErrorCode)63)
#define PETSC_ERR_ARG_SIZ ((PetscErrorCode)60)
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

/* PETSC_DECIDE represents a constant used in place of an integer argument
   when you want PETSc to choose the value for that argument
   - Extracted from petscsys.h*/
#ifndef PETSC_DECIDE
#define PETSC_DECIDE (-1)
#endif

/* PETSC_DETERMINE is like PETSC_DECIDE.
   - Extracted from petscsys.h*/
#ifndef PETSC_DETERMINE
#define PETSC_DETERMINE PETSC_DECIDE
#endif

// PETSC_DEFAULT represents a default value
#define PETSC_DEFAULT (-2)

// PETSC_COMM_WORLD is the MPI communiator used for PETSc communiation
#define PETSC_COMM_WORLD MPI_COMM_WORLD

#define PETSC_COMM_SELF MPI_COMM_SELF

/* PetscInt_FMT is a format specifier for PetscInt used in formatted output
   - Extracted from petscsystypes.h */
#define PetscInt_FMT "d"

/* PetscSqrtReal computes the square root of a real number.
   - Extracted from petscmath.h*/
#define PetscSqrtReal(a) sqrt(a)

/* PETSC_MAX_REAL represents the maximum  double real number value
   - Extracted from petscmath.h*/
#define PETSC_MAX_REAL 1.7976931348623157e+308

/* PETSC_MIN_REAL represents the minimum double real number value
   - Extracted from petscmath.h*/
#define PETSC_MIN_REAL (-PETSC_MAX_REAL)

/* Enumeration of different types of norms used in PETSc
   - Extracted from petscvec.h*/
typedef enum NORM_TYPE {
  NORM_1 = 0,
  NORM_2 = 1,
  NORM_FROBENIUS = 2,
  NORM_INFINITY = 3,
  NORM_1_AND_2 = 4
} NormType;

extern PetscInt NormIds[5];

/* Enumeration of different insert modes used in PETSc
   -Extracted from petscsystypes.h*/
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

/* PetscCall is a macro used to wrap calls to PETSc targets
   - Extracted from petscerror.h */
#define PetscCall(a) a

/* PetscFunctionBeginUser marks the beginning of a user-defined function
   - Extracted from petscerror.h */
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

/* A unique id used to identify each Vector class.
   - Extracted from petscvec.h*/
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

/* Macros to test if a PETSc object is valid and if pointers are valid
   - Extracted from petscimpl.h*/
PetscErrorCode PetscValidHeaderSpecific(void *x, PetscClassId cid, int arg);

/* Use this macro to check if the type is set */
#define PetscValidType(a, arg) ((void)0)

#define PetscLogEventBegin(e, o1, o2, o3, o4) ((void)0)

#define PetscLogEventEnd(e, o1, o2, o3, o4) ((void)0)

#define PetscAssertPointer(h, arg) $assert(h != NULL)

// <--- Dependencies for VecAsyncFnName --->
#define VEC_ASYNC_FN_NAME(Base) "Vec" Base "Async_Private_C"
#define VEC_PointwiseDivide_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("PointwiseDivide")
#define VEC_PointwiseMult_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("PointwiseMult")
#define VEC_PointwiseMax_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("PointwiseMax")
#define VEC_PointwiseMin_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("PointwiseMin")
#define VEC_PointwiseMaxAbs_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("PointwiseMaxAbs")
#define VEC_Swap_ASYNC_FN_NAME VEC_ASYNC_FN_NAME("Swap")
#define VecAsyncFnName(Base) VEC_##Base##_ASYNC_FN_NAME

// <--- Dependencies for PetscUseTypeMethod --->
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

/* PetscUseTypeMethod - Call a method on a `PetscObject`, that is a function in
   the objects function table `obj->ops`, error if the method does not exist
   - Extracted from petscimpl.h */
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

/* PetscValidLogicalCollectiveInt - Validates that an integer value is logically
  collective across all processes.
  - Extracted from petscimpl.h */
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

/*   PetscValidFunction - Macro to validate a function pointer, ensuring it is
  not NULL.
  - Extracted from petscimpl.h */
#define PetscValidFunction(f, arg)                                             \
  PetscCheck((f), PETSC_COMM_SELF, PETSC_ERR_ARG_NULL,                         \
             "Null Function Pointer: Parameter # %d", arg)

/* VecSetErrorIfLocked - Macro to check if a vector is null or locked for
  access.
  - Extracted from petscvec.h */
#define VecSetErrorIfLocked(x, args)                                           \
  do {                                                                         \
    $assert((x), "Error: Null vector passed to VecSetErrorIfLocked.\n");       \
    $assert((x)->read_lock_count == 0, "Vector was locked for access.\n");     \
  } while (0)

/* PetscMalloc1 - Allocates memory for an array of given size and assigns it to
  the provided pointer.
  - Extracted from petscsys.h */
#define PetscMalloc1(m1, r1)                                                   \
  (*(r1) = malloc((m1) * sizeof(*(r1))), PETSC_SUCCESS)

/* PetscObjectComposedDataGetReal - Retrieves real-valued composed data
  associated with a PetscObject.
  - Extracted from petscimpl.h */
PetscErrorCode PetscObjectComposedDataGetReal(PetscObject obj, PetscInt id,
                                              PetscReal *data, PetscBool *flag);

/* PetscObjectComposedDataSetReal - Sets real-valued composed data associated
with a PetscObject.
- Extracted from petscimpl.h */
PetscErrorCode PetscObjectComposedDataSetReal(PetscObject obj, PetscInt id,
                                              PetscReal data);

/*   PetscInfo_Private - Logs informational messages associated with a
  PetscObject.
  - Extracted from petsclog.h */
PetscErrorCode PetscInfo_Private(PetscObject obj, const char message[]);

#define PetscInfo(A, ...) PetscInfo_Private(((PetscObject)A), __VA_ARGS__)

int __isfinited(double x);

/* CIVL_Scalar_Bcast - Broadcasts a PetscScalar value across processes in an MPI
  communicator. */
void CIVL_Scalar_Bcast(PetscScalar *val, int count, int root, MPI_Comm comm);

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

/*   PetscIsInfOrNanReal - Checks if a given PetscReal value is either infinite
  or NaN.
  - Extracted from petscmath.h */
PetscBool PetscIsInfOrNanReal(PetscReal v);

// PetscCallBLAS calls a BLAS function
#define PetscCallBLAS(x, X) X

#define PetscCallMPI(x) (x)

/*   MPIU_Allreduce - Wrapper macro for MPI_Allreduce to simplify usage with
  consistent arguments.
  - Extracted from petscsys.h */
#define MPIU_Allreduce(a, b, c, d, e, fcomm)                                   \
  MPI_Allreduce((a), (b), (c), (d), (e), (fcomm))

/* PetscArraycpy copies elements from one array (str1) to another (str2)
  - Extracted from petscstring.h */
#define PetscArraycpy(str1, str2, cnt)                                         \
  ((sizeof(*(str1)) == sizeof(*(str2)))                                        \
       ? PetscMemcpy((str1), (str2), (size_t)(cnt) * sizeof(*(str1)))          \
       : PETSC_ERR_ARG_SIZ)

/* PetscCheckSameType - Macro to check if two vectors have the same type
  - Extracted from petscimpl.h */
#define PetscCheckSameType(a, arga, b, argb)                                   \
  $assert((a) && (b), "Error: Null pointer passed to PetscCheckSameType.");    \
  $assert((a)->type == (b)->type, "Error: Vectors have different types.")

/* PetscCheckSameComm - Macro to check if two vectors have the same communicator
  - Extracted from petscimpl.h */
#define PetscCheckSameComm(a, arga, b, argb)                                   \
  $assert((a) && (b), "Error: Null pointer passed to PetscCheckSameComm.");    \
  $assert((a)->comm == (b)->comm,                                              \
          "Error: Vectors have different communicators.")

/* PetscCheckSameTypeAndComm - Macro to check if two objects have the same
  type and communicator.
  - Extracted from petscimpl.h */
#define PetscCheckSameTypeAndComm(a, arga, b, argb)                            \
  do {                                                                         \
    PetscCheckSameType(a, arga, b, argb);                                      \
    PetscCheckSameComm(a, arga, b, argb);                                      \
  } while (0)

/* VecCheckSameSize - Macro to verify that two vectors have the same global size
  and are not null.
  - Extracted from vecimpl.h */
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
  - Extracted from rvector.c */
#define VecLockReadPush(x)                                                     \
  do {                                                                         \
    $assert((x) != NULL, "VecLockReadPush: Vector pointer is NULL.");          \
    (x)->read_lock_count++;                                                    \
  } while (0)

/*
  VecLockReadPop - Pops a read-only lock from a vector, decreasing its read lock
  count.

  Parameters:
  - x The vector to unlock for reading.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h */
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

/*
  VecMethodDispatch - Dispatches a method call for a vector object, optionally
  using an asynchronous method if a dispatch context is provided.

  Parameters:
  - v: The vector object.
  - dctx: The dispatch context (can be NULL).
  - async_name: The name of the asynchronous method to query.
  - name: The name of the synchronous method to use as fallback.
  - async_arg_types: The argument types for the asynchronous method.
  - ...: Additional arguments to pass to the method.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from vecimpl.h
*/
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

// Define MPIU_REAL as MPI_DOUBLE for real number communication in MPI
#define MPIU_REAL MPI_DOUBLE

/*
  PetscValidLogicalCollectiveScalar - Validates that a scalar value is logically
  collective and consistent across all processes in a communicator.

  Parameters:
  - a The PetscObject (e.g., Vec, Mat) associated with the communicator.
  - b The scalar value to validate.
  - arg The argument position of the scalar value in the calling function.

  Returns: None (throws an error if validation fails).
  - Extracted from petscimpl.h
*/
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

/*
  VecCheckSameLocalSize - Checks if two vectors have the same local size.

  Parameters:
  - x, ar1: The first vector and its parameter number.
  - y, ar2: The second vector and its parameter number.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from vecimpl.h
*/
#define VecCheckSameLocalSize(x, ar1, y, ar2)                                  \
  do {                                                                         \
    PetscCheck(                                                                \
        (x)->map->n == (y)->map->n, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,     \
        "Incompatible vector local lengths parameter # %d local size "         \
        "%" PetscInt_FMT " != parameter # %d local size %" PetscInt_FMT,       \
        ar1, (x)->map->n, ar2, (y)->map->n);                                   \
  } while (0)

/*PetscFree - Frees allocated memory and sets the pointer to NULL.

  Parameters:
  - a The pointer to the memory to be freed.

  Returns: 0 if the pointer is NULL, otherwise frees the memory and sets the
  pointer to NULL.
  - Extracted from petscsys.h */
#define PetscFree(a)                                                           \
  do {                                                                         \
    if (!(a))                                                                  \
      return 0;                                                                \
    free(a);                                                                   \
    (a) = NULL;                                                                \
  } while (0)

/*VecMax_Seq_GT - Compares two PetscReal values and returns 1 if the first is
  greater than the second, otherwise returns 0.

  Parameters:
  - l The first PetscReal value.
  - r The second PetscReal value.

  Returns: int (1 if l > r, 0 otherwise).
  - Extracted from dvec2.c */
static int VecMax_Seq_GT(PetscReal l, PetscReal r) { return (l > r) ? 1 : 0; }

/*VecMin_Seq_LT - Compares two PetscReal values and returns 1 if the first is
  less than the second, otherwise returns 0.

  Parameters:
  - l The first PetscReal value.
  - r The second PetscReal value.

  Returns: int (1 if l < r, 0 otherwise).
  - Extracted from dvec2.c */
static int VecMin_Seq_LT(PetscReal l, PetscReal r) { return (l < r) ? 1 : 0; }

// MPIU_SUM: Alias for MPI_SUM, used for summation in MPI operations.
#define MPIU_SUM MPI_SUM

// MPIU_SCALAR: Alias for MPIU_REAL, representing scalar values in MPI.
#define MPIU_SCALAR MPIU_REAL

// PetscMax(a, b): Macro to compute the maximum of two values.
#define PetscMax(a, b) (((a) < (b)) ? (b) : (a))

// Disabling Fortran kernel and pragma-related macros as they are not being used
// or verified
#ifdef PETSC_USE_FORTRAN_KERNEL_AYPX
#undef PETSC_USE_FORTRAN_KERNEL_AYPX
#endif

#ifdef PETSC_USE_FORTRAN_KERNEL_WAXPY
#undef PETSC_USE_FORTRAN_KERNEL_WAXPY
#endif

#define PETSC_HAVE_MPIUNI 0

#ifdef PETSC_HAVE_PRAGMA_DISJOINT
#undef PETSC_HAVE_PRAGMA_DISJOINT
#endif

// Define PETSC_RESTRICT macro to handle the restrict keyword for portability.
#ifndef PETSC_RESTRICT
#define PETSC_RESTRICT restrict
#endif

// Defines MPI and PETSc-related macros for data types and operations to
// simplify usage in the code.
#define MPIU_REAL_INT MPI_DOUBLE_INT
#define MPIU_MAXLOC MPI_MAXLOC
#define MPIU_MINLOC MPI_MINLOC
#define MPIU_MAX MPI_MAX
#define MPIU_MIN MPI_MIN
#ifndef PETSC_MAX_INT
#define PETSC_MAX_INT INT_MAX
#endif

/* PetscArrayzero - Sets all elements of an array to a scalar zero value.

  Parameters:
  - arr The array to be zeroed.
  - cnt The number of elements in the array.

  Returns: None (macro).
  - Extracted from petscstring.h */
#define PetscArrayzero(arr, cnt)                                               \
  do {                                                                         \
    size_t _i;                                                                 \
    for (_i = 0; _i < (cnt); _i++) {                                           \
      (arr)[_i] = scalar_zero;                                                 \
    }                                                                          \
  } while (0)

/* Dependencies macros for VecAXPY function
   - Extracted from petscaxpy.h */
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

/* PetscIsNanScalar - Checks if a given scalar value is NaN (Not a Number).

  Parameters:
  - v The scalar value to check.

  Returns: PetscBool (PETSC_TRUE if the value is NaN, PETSC_FALSE otherwise).
  - Extracted from petscmath.h
*/
PetscBool PetscIsNanScalar(PetscScalar v);

/* PetscEqualReal - Compares two real numbers for equality.

  Parameters:
  - a The first real number.
  - b The second real number.

  Returns: PetscBool (PETSC_TRUE if the numbers are equal, PETSC_FALSE
  otherwise).
  - Extracted from petscmath.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscEqualReal/#petscequalreal
*/
PetscBool PetscEqualReal(PetscReal a, PetscReal b);

/*
  Copies n bytes from location b to location a.
  Parameters:
  - a Destination pointer.
  - b Source pointer.
  - n Number of bytes to copy.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Returns an error code if either `a` or `b` is NULL.

  - Extracted from petscstring.h
  - URL: https://petsc.org/release/manualpages/Sys/PetscMemcpy/#petscmemcpy
 */
PetscErrorCode PetscMemcpy(void *a, const void *b, size_t n);

/* PetscObjectComm - Gets the MPI communicator for any `PetscObject`
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
  - Extracted from gcomm.c
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscObjectComm/#petscobjectcomm
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
  - Extracted from petscsys.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscInitialize/#petscinitialize */
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
  - Extracted from petscoptions.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscOptionsGetInt/#petscoptionsgetint
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
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecCreate/#veccreate
 */
PetscErrorCode VecCreate(MPI_Comm comm, Vec *vec);

/*
  Computes the absolute value of a real number.
  Parameters:
  - v1 Input real number.

  Returns: The absolute value of v1.
  - Extracted from petscmath.h
  - URL: https://petsc.org/release/manualpages/Sys/PetscAbsReal/#petscabsreal
 */
PetscReal PetscAbsReal(PetscReal v1);

/*
  Safely casts a PetscInt to a PetscBLASInt.
  Parameters:
  - a Input PetscInt value.
  - b Pointer to store the casted PetscBLASInt value.

  Returns: PetscErrorCode (0 on success, 1 if out of range).

  Note: Checks for negative values and overflow before casting.
  - Extracted from petscsys.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscBLASIntCast/#petscblasintcast
 */
PetscErrorCode PetscBLASIntCast(PetscInt a, PetscBLASInt *b);

/*
  Creates a new vector of the same type as an existing vector.
  Parameters:
  - v Input vector to be duplicated.
  - newv Pointer to the new vector to be created.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for the new vector and copies size information.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecDuplicate/#vecduplicate
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
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecDuplicateVecs/#vecduplicatevecs
 */
PetscErrorCode VecDuplicateVecs(Vec v, PetscInt m, Vec *V[]);

/*
  Destroys multiple vectors and frees their memory.
  Parameters:
  - m Number of vectors to destroy.
  - vv Pointer to an array of Vec pointers to be destroyed.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Calls VecDestroy on each vector and frees the array.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecDestroyVecs/#vecdestroyvecs
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
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecGetOwnershipRange/#vecgetownershiprange
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
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecGetOwnershipRanges/#vecgetownershipranges
 */
PetscErrorCode VecGetOwnershipRanges(Vec x, const PetscInt *ranges[]);

/*
  Splits the ownership of a global size across processes in an MPI communicator.

  Parameters:
  - comm: The MPI communicator.
  - n: Pointer to the local size (input/output). If input is PETSC_DECIDE, it is
  computed.
  - N: Pointer to the global size (input/output). If input is PETSC_DECIDE, it
  is computed.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function ensures that the sum of all local sizes equals the global
  size.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscSplitOwnership/#petscsplitownership
 */
PetscErrorCode PetscSplitOwnership(MPI_Comm comm, PetscInt *n, PetscInt *N);
/*
  Sets the local and global sizes of a vector.
  Parameters:
  - v Vector to set sizes for.
  - n Local size (or PETSC_DECIDE).
  - N Global size (or PETSC_DETERMINE).

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Sets the local and global sizes in the vector's map.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecSetSizes/#vecsetsizes
 */
PetscErrorCode VecSetSizes(Vec v, PetscInt n, PetscInt N);

/*
  VecSetUp - Initializes the vector type and sets up internal data structures.

  Parameters:
  - v Vector to be set up.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: If the vector type is not set, it is initialized based on the number of
  processes.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecSetUp/#vecsetup
*/
PetscErrorCode VecSetUp(Vec v);

/*
  Sets the block size of a vector.
  Parameters:
  - v Vector to set block size for.
  - bs Block size to set.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSetBlockSize/#vecsetblocksize
 */
PetscErrorCode VecSetBlockSize(Vec v, PetscInt bs);

/*
  Configures the vector from options.
  Parameters:
  - vec Vector to configure.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for vector data based on the local size.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSetFromOptions/#vecsetfromoptions
 */
PetscErrorCode VecSetFromOptions(Vec vec);

/*
  Sets the type of a PETSc vector.

  Parameters:
  - vec: The PETSc vector whose type is to be set.
  - newType: The type to set for the vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: The type determines the internal implementation of the vector.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSetType/#vecsettype
 */
PetscErrorCode VecSetType(Vec vec, VecType newType);

/*
  Retrieves the type of a PETSc vector.

  Parameters:
  - vec: The PETSc vector whose type is to be retrieved.
  - type: Pointer to a variable to store the type of the vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: The type of the vector determines its implementation, such as
  standard, MPI, or other specialized types.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecGetType/#vecgettype
 */
PetscErrorCode VecGetType(Vec vec, VecType *type);

/*
  Sets all components of a vector to a single scalar value.
  Parameters:
  - x Vector to set values in.
  - alpha Scalar value to set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSet/#vecset
 */
PetscErrorCode VecSet(Vec x, PetscScalar alpha);

/*
  Internal auxiliary function for VecSet to set all elements of a sequential
  vector to a specified scalar value on a single process.

  Parameters:
  - xin: The input vector to be modified.
  - alpha: The scalar value to set for all elements of the vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors and is not intended
  for parallel use.
  - Extracted from dvecimpl.h
 */
PetscErrorCode VecSet_Seq(Vec xin, PetscScalar alpha);

/*
  Displays the vector.
  Parameters:
  - vec Vector to view.
  - viewer PetscViewer object.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Prints vector contents to stdout in simplified version.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecView/#vecview
 */
PetscErrorCode VecView(Vec vec, PetscViewer viewer);

/*
  Adds floating point operations to the global counter.
  Parameters:
  - n Number of flops to add.

  Returns: PetscErrorCode (0 on success, 1 if n is negative).
  - Extracted from petsclog.h
  - URL:
  https://petsc.org/release/manualpages/Log/PetscLogFlops/#petsclogflops
 */
PetscErrorCode PetscLogFlops(PetscLogDouble n);

/*
  Swaps the values between two vectors.
  Parameters:
  - x First vector.
  - y Second vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecSwap/#vecswap
 */
PetscErrorCode VecSwap(Vec x, Vec y);

/*
  Internal auxiliary function for VecDot to compute the dot product of two
  sequential vectors on a single process.

  Parameters:
  - xin: The first input vector.
  - yin: The second input vector.
  - z: Pointer to store the resulting dot product.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors and is not intended
  for parallel use.
  - Extracted from dvecimpl.h
 */
PetscErrorCode VecDot_Seq(Vec xin, Vec yin, PetscScalar *z);

/*
  Computes the dot product of two parallel vectors in an MPI environment.

  Parameters:
  - xin: The first input vector.
  - yin: The second input vector.
  - z: Pointer to store the resulting dot product.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for parallel vectors and utilizes MPI
  communication to compute the global dot product.
  - Extracted from pvecimpl.h
 */
PetscErrorCode VecDot_MPI(Vec xin, Vec yin, PetscScalar *z);

/*
  Computes the dot product of two vectors.
  Parameters:
  - x First vector.
  - y Second vector.
  - val Pointer to store the dot product result.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: In complex mode, val = x · y' where y' is the conjugate transpose of y.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecDot/#vecdot
 */
PetscErrorCode VecDot(Vec x, Vec y, PetscScalar *val);

/*
  Computes the real part of the dot product of two vectors.

  Parameters:
  - x: The first vector.
  - y: The second vector.
  - val: Pointer to store the real part of the dot product.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is intended for use with real-valued vectors.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecDotRealPart/#vecdotrealpart
 */
PetscErrorCode VecDotRealPart(Vec x, Vec y, PetscReal *val);

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
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecTDot/#vectdot
*/
PetscErrorCode VecTDot(Vec x, Vec y, PetscScalar *val);

/*
  Internal auxiliary function for VecTDot to computet the transpose dot product
  of two parallel vectors in an MPI environment.

  Parameters:
  - xin: The first input vector.
  - yin: The second input vector.
  - z: Pointer to store the resulting transpose dot product.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for parallel vectors and utilizes MPI
  communication to compute the global transpose dot product.
  - Extracted from pvecimpl.h
 */
PetscErrorCode VecTDot_MPI(Vec xin, Vec yin, PetscScalar *z);

/*
  Internal auxiliary function for VecTDot to compute the transpose dot product
  of two sequential vectors on a single process.
  Parameters:
  - xin: The first input vector.
  - yin: The second input vector.
  - z: Pointer to store the resulting transpose dot product.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors to compute the global
  transpose dot product.
  - Extracted from dvecimpl.h
 */
PetscErrorCode VecTDot_Seq(Vec xin, Vec yin, PetscScalar *z);

// Internal Auxilary function used in VecMTDot
PetscErrorCode VecMXDot_Private(
    Vec x, PetscInt nv, const Vec y[], PetscScalar result[],
    PetscErrorCode (*mxdot)(Vec, PetscInt, const Vec[], PetscScalar[]),
    PetscLogEvent event);

/*
  VecMTDot - Computes multiple indefinite vector dot products, i.e.,

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
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMTDot/#vecmtdot
*/
PetscErrorCode VecMTDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]);

/*
  Internal auxiliary function for VecMTDot to compute multiple transpose dot
  products of parallel vectors in an MPI environment.

  Parameters:
  - xin: The input vector.
  - nv: The number of vectors.
  - y: Array of input vectors.
  - z: Pointer to store the resulting transpose dot products (length nv).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for parallel vectors and utilizes MPI
  communication to compute the global transpose dot products.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecMTDot_MPI(Vec xin, PetscInt nv, const Vec y[],
                            PetscScalar *z);

/*
  Internal auxiliary function for VecMTDot to compute multiple transpose dot
  products of sequential vectors on a single process.

  Parameters:
  - xin: The input vector.
  - nv: The number of vectors.
  - y: Array of input vectors.
  - z: Pointer to store the resulting transpose dot products (length nv).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors to compute the global
  transpose dot products.
  - Extracted from dvecimpl.h
*/
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
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMDot/#vecmdot
 */
PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]);

/*
  Internal auxiliary function to compute multiple vector dot products.

  The function computes the dot product of the input vector x with each vector
  in the array y, for a total of nv dot products. The result of each dot product
  is stored in the corresponding position in the output array val.

  Parameters:
  - x: The input sequential vector.
  - nv: The number of vectors in the array y.
  - y: An array of sequential vectors to be used for dot product computations.
  - val: Output array where each entry is set to the dot product of x with the
  corresponding vector in y.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecMDot_Seq(Vec x, PetscInt nv, const Vec y[],
                           PetscScalar val[]);

/*
  Internal auxiliary function to compute multiple vector dot products.

  Parameters:
  - x: The primary input vector for which the dot products are computed.
  - nv: The number of dot products to compute, corresponding to the number of
  vectors in y.
  - y: An array of secondary input vectors used for dot product computations.
  - val: An output array where each element is set to the computed dot product
  of x with the corresponding vector in y.

  Returns:
  - PetscErrorCode: 0 on success, or a non-zero error code if an error occurs
  during the computation.

  Note:
  - This function assumes that all vectors are correctly distributed across
  processes in a parallel computing environment.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecMDot_MPI(Vec x, PetscInt nv, const Vec y[],
                           PetscScalar val[]);

/*
  Computes multiple dot products of vector x with an array of vectors y.
  Parameters:
  - x Input vector.
  - nv The number of vectors in the array y.
  - y Array of vectors with which to compute dot products.
  - val Array to store the resulting dot products; must have at least nv
  elements.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMDot/#vecmdot
*/
PetscErrorCode VecMDot(Vec x, PetscInt nv, const Vec y[], PetscScalar val[]);

/*
  Returns the global number of elements in the vector.
  Parameters:
  - x Input vector.
  - size Pointer to store the size.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecGetSize/#vecgetsize
 */
PetscErrorCode VecGetSize(Vec x, PetscInt *size);

/*
  Internal auxiliary function to compute the global size (N) of an
  MPI-distributed vector.

  Parameters:
  - x: The MPI vector whose size is to be determined.
  - size: Pointer to an integer where the global size will be stored.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is intended for internal use only and is optimized for MPI
  environments.
*/
PetscErrorCode VecGetSize_MPI(Vec x, PetscInt *size);

/*
  Internal auxiliary function to retrieve the number of elements in a sequential
  vector on a single process.

  Parameters:
  - x: The sequential vector whose size is being queried.
  - size: Pointer to store the resulting number of elements in the vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors and is not intended
  for parallel use.
*/
PetscErrorCode VecGetSize_Seq(Vec x, PetscInt *size);

/*
  Returns the number of elements of the vector stored in local memory.
  Parameters:
  - x Input vector.
  - size Pointer to store the local size.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecGetLocalSize/
 */
PetscErrorCode VecGetLocalSize(Vec x, PetscInt *size);

/*
  Determines the vector component with the maximum real part and its location.

  Parameters:
  - x: Input vector.
  - p: Pointer to store the index of the maximum element.
  - val: Pointer to store the maximum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: For complex vectors, it considers the real part for comparison.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMax/#vecmax
*/
PetscErrorCode VecMax(Vec x, PetscInt *p, PetscReal *val);

/*
  Internal auxiliary function for VecMax to determine the maximum component
  and its index for parallel vectors using MPI communication.

  Parameters:
  - xin: Input parallel vector.
  - idx: Pointer to store the index of the maximum element.
  - z: Pointer to store the maximum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function uses MPI operations to identify the global maximum across
  parallel processes.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecMax_MPI(Vec xin, PetscInt *idx, PetscReal *z);

/*
  Internal auxiliary function for VecMax to determine the maximum component
  and its index for sequential vectors on a single process.

  Parameters:
  - xin: Input sequential vector.
  - idx: Pointer to store the index of the maximum element.
  - z: Pointer to store the maximum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecMax_Seq(Vec xin, PetscInt *idx, PetscReal *z);

/*
  Determines the vector component with minimum real part and its location.

  Parameters:
  - x: Input vector.
  - p: Pointer to store the index of the minimum element.
  - val: Pointer to store the minimum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: For complex vectors, considers the real part for comparison.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMin/#vecmin
*/
PetscErrorCode VecMin(Vec x, PetscInt *p, PetscReal *val);

/*
  Internal auxiliary function for VecMin to determine the minimum component
  and its index for parallel vectors using MPI communication.

  Parameters:
  - xin: Input parallel vector.
  - idx: Pointer to store the index of the minimum element.
  - z: Pointer to store the minimum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function uses MPI operations to identify the global minimum across
  parallel processes.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecMin_MPI(Vec xin, PetscInt *idx, PetscReal *z);

/*
  Internal auxiliary function for VecMin to determine the minimum component
  and its index for sequential vectors on a single process.

  Parameters:
  - xin: Input sequential vector.
  - idx: Pointer to store the index of the minimum element.
  - z: Pointer to store the minimum value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecMin_Seq(Vec xin, PetscInt *idx, PetscReal *z);

typedef struct _p_PetscDeviceContext *PetscDeviceContext;

// Internal Auxilary function used in VecScale
PetscErrorCode VecScaleAsync_Private(Vec x, PetscScalar alpha,
                                     PetscDeviceContext dctx);

// Internal Auxilary function used in VecSet
PetscErrorCode VecSetAsync_Private(Vec x, PetscScalar alpha,
                                   PetscDeviceContext dctx);

/*
  Scales a vector by multiplying each element by a scalar.
  Parameters:
  - x Vector to scale.
  - alpha Scalar to multiply by.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecScale/#vecscale
 */
PetscErrorCode VecScale(Vec x, PetscScalar alpha);

/*
  Internal auxiliary function for VecScale to scale a sequential vector by a
  scalar on a single process.

  Parameters:
  - xin: The input vector to be scaled.
  - alpha: The scalar value by which to scale the vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is optimized for sequential vectors and is not intended
  for parallel use.
  - Extracted from dvecimpl.h
 */
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
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecEqual/#vecequal
 */
PetscErrorCode VecEqual(Vec vec1, Vec vec2, PetscBool *flg);

// Internal Auxilary function used in VecMAXPY
PetscErrorCode VecMAXPYAsync_Private(Vec y, PetscInt nv,
                                     const PetscScalar alpha[], Vec x[],
                                     PetscDeviceContext dctx);

/*
  Computes y = y + sum(alpha[i] * x[i]) for multiple vectors. Updates the
  vector `y` by adding scaled versions of vectors `x[i]` weighted by
  `alpha[i]` for each `i` in the range `[0, nv-1]`.

  Parameters:
  - y: Vector to be updated.
  - nv: Number of vectors.
  - alpha: Array of scalars.
  - x: Array of vectors.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex scalars and vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMAXPY/#vecmaxpy
*/
PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[]);

/*
  Internal auxiliary function for VecMAXPY optimized for sequential vectors.
  Performs the operation y = y + sum(alpha[i] * x[i]) without parallel
  communication.

  Parameters:
  - xin: Vector to be updated.
  - nv: Number of vectors.
  - alpha: Array of scalars.
  - y: Array of input vectors.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecMAXPY_Seq(Vec xin, PetscInt nv, const PetscScalar *alpha,
                            Vec *y);

/*
  Computes y = beta*y + sum(alpha[i] * x[i]) for multiple vectors. Updates the
  vector `y` by scaling it with `beta` and adding scaled versions of vectors
  `x[i]` weighted by `alpha[i]` for each `i` in the range `[0, nv-1]`.

  Parameters:
  - y: Vector to be updated.
  - nv: Number of vectors.
  - alpha: Array of scalars.
  - beta: Scalar multiplier for vector y.
  - x: Array of vectors.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex scalars and vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecMAXPBY/#vecmaxpby
*/
PetscErrorCode VecMAXPBY(Vec y, PetscInt nv, const PetscScalar alpha[],
                         PetscScalar beta, Vec x[]);

// Internal Auxilary function used in VecAXPY
PetscErrorCode VecAXPYAsync_Private(Vec y, PetscScalar alpha, Vec x,
                                    PetscDeviceContext dctx);

/*
  Computes y = alpha * x + y. Updates the vector `y` by adding the vector
  `x` scaled by the scalar `alpha`.

  Parameters:
  - y: Vector to be updated.
  - alpha: Scalar multiplier.
  - x: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex scalars and vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecAXPY/#vecaxpy
*/
PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x);

/*
  Internal auxiliary function for VecAXPY optimized for sequential vectors.
  Performs the operation y = alpha * x + y without parallel communication.

  Parameters:
  - yin: Vector to be updated.
  - alpha: Scalar multiplier.
  - xin: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecAXPY_Seq(Vec yin, PetscScalar alpha, Vec xin);

// Internal Auxilary function used in VecAXPBY
PetscErrorCode VecAXPBYAsync_Private(Vec y, PetscScalar alpha, PetscScalar beta,
                                     Vec x, PetscDeviceContext dctx);

/*
  Computes the linear combination of two vectors `x` and `y`:
      y = alpha * x + beta * y

  Parameters:
  - y: Vector to be updated.
  - alpha: Scalar multiplier for vector `x`.
  - beta: Scalar multiplier for vector `y`.
  - x: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function performs element-wise operations and assumes that the
  vectors are of the same length. Supports both real and complex scalars and
  vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecAXPBY/#vecaxpby
*/
PetscErrorCode VecAXPBY(Vec y, PetscScalar alpha, PetscScalar beta, Vec x);

/*
  Internal auxiliary function for VecAXPBY optimized for sequential vectors.
  Performs the operation y = a * x + b * y without parallel communication.

  Parameters:
  - yin: Vector to be updated.
  - a: Scalar multiplier for vector `xin`.
  - b: Scalar multiplier for vector `yin`.
  - xin: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecAXPBY_Seq(Vec yin, PetscScalar a, PetscScalar b, Vec xin);

// Internal Auxilary function used in VecAXPBYPCZ
PetscErrorCode VecAXPBYPCZAsync_Private(Vec z, PetscScalar alpha,
                                        PetscScalar beta, PetscScalar gamma,
                                        Vec x, Vec y, PetscDeviceContext dctx);

/*
  Computes the linear combination of three vectors `x`, `y`, and `z`:
      z = alpha * x + beta * y + gamma * z

  Parameters:
  - z: Vector to be updated.
  - alpha: Scalar multiplier for vector `x`.
  - beta: Scalar multiplier for vector `y`.
  - gamma: Scalar multiplier for vector `z`.
  - x: First vector to be added.
  - y: Second vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function performs element-wise operations and assumes that the
  vectors are of the same length. Supports both real and complex scalars and
  vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecAXPBYPCZ/#vecaxpbypcz
*/
PetscErrorCode VecAXPBYPCZ(Vec z, PetscScalar alpha, PetscScalar beta,
                           PetscScalar gamma, Vec x, Vec y);

/*
  Internal auxiliary function for VecAXPBYPCZ optimized for sequential vectors.
  Performs the operation z = alpha * x + beta * y + gamma * z without parallel
  communication.

  Parameters:
  - zin: Vector to be updated.
  - alpha: Scalar multiplier for vector `xin`.
  - beta: Scalar multiplier for vector `yin`.
  - gamma: Scalar multiplier for vector `zin`.
  - xin: First vector to be added.
  - yin: Second vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecAXPBYPCZ_Seq(Vec zin, PetscScalar alpha, PetscScalar beta,
                               PetscScalar gamma, Vec xin, Vec yin);

// Internal Auxilary function used in VecAYPX
PetscErrorCode VecAYPXAsync_Private(Vec y, PetscScalar beta, Vec x,
                                    PetscDeviceContext dctx);

/*
  Computes y = x + beta * y. Updates the vector `y` by adding the vector `x` to
  `y` scaled by the scalar `beta`.

  Parameters:
  - y: Vector to be updated.
  - beta: Scalar multiplier for vector `y`.
  - x: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex scalars and vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecAYPX/#vecaypx
*/
PetscErrorCode VecAYPX(Vec y, PetscScalar beta, Vec x);

/*
  Internal auxiliary function for VecAYPX optimized for sequential vectors.
  Performs the operation y = x + alpha * y without parallel communication.

  Parameters:
  - yin: Vector to be updated.
  - alpha: Scalar multiplier for vector `yin`.
  - xin: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecAYPX_Seq(Vec yin, PetscScalar alpha, Vec xin);

// Internal Auxilary function used in VecWAXPY
PetscErrorCode VecWAXPYAsync_Private(Vec w, PetscScalar alpha, Vec x, Vec y,
                                     PetscDeviceContext dctx);

/*
  Computes w = alpha * x + y. Stores the result in the vector `w` by adding
  the vector `y` to `alpha` times the vector `x`.

  Parameters:
  - w: Vector to store the result.
  - alpha: Scalar multiplier for vector `x`.
  - x: Vector to be scaled and added.
  - y: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex scalars and vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecWAXPY/#vecwaxpy
*/
PetscErrorCode VecWAXPY(Vec w, PetscScalar alpha, Vec x, Vec y);

/*
  Internal auxiliary function for VecWAXPY optimized for sequential vectors.
  Performs the operation w = alpha * x + y without parallel communication.

  Parameters:
  - win: Vector to store the result.
  - alpha: Scalar multiplier for vector `xin`.
  - xin: Vector to be scaled and added.
  - yin: Vector to be added.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecWAXPY_Seq(Vec win, PetscScalar alpha, Vec xin, Vec yin);

/*
  Internal auxiliary function for VecPointwiseMult performs component-wise
  multiplication of two sequential vectors.

  Description:
    Computes the operation w[i] = x[i] * y[i] for each element i in the input
  vectors x and y, storing the result in the output vector w. This function is
  designed for sequential (non-parallel) vectors.

  Input Parameters:
    w  - Sequential vector where the computed product is stored.
    x  - First input sequential vector.
    y  - Second input sequential vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note:
    This function supports both real and complex numbers by performing
  multiplication element-wise. It is part of the PETSc vector operations defined
  for sequential vectors.

  Extracted from: dvecimpl.h
*/
PetscErrorCode VecPointwiseMult_Seq(Vec w, Vec x, Vec y);

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
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecPointwiseMult/#vecpointwisemult
 */
PetscErrorCode VecPointwiseMult(Vec w, Vec x, Vec y);

/*
  Internal auxiliary function to compute the maximum pointwise division of two
  sequential vectors on a single process.

  Parameters:
    - x: The first input vector (numerator).
    - y: The second input vector (denominator).
    - w: The vector to store the result of the pointwise division.

  Returns:
    - PetscErrorCode: 0 on success, non-zero on failure.

  Note:
    This function is optimized for sequential vectors and is not intended for
  parallel use.

  Extracted from dvecimpl.h
*/
PetscErrorCode VecPointwiseDivide_Seq(Vec w, Vec x, Vec y);

/*
  Computes the component-wise division w[i] = x[i] / y[i]. This operation is
  performed for each element `i` of the vectors `x`, `y`. Parameters:
  - w Vector to store the result.
  - x First input vector (numerator).
  - y Second input vector (denominator).

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex numbers. Handles division by zero
  appropriately.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecPointwiseDivide/#vecpointwisedivide
 */
PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y);

/*
  Internal auxiliary function to compute the component-wise maximum of two
  sequential vectors on a single process. The function sets each entry of the
  output vector w to the maximum of the corresponding entries in x and y, i.e.,
  w[i] = max(x[i], y[i]).

  Parameters:
  - w: Output vector which will hold the component-wise maximum values.
  - x: The first input vector.
  - y: The second input vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: For complex numbers, only the real parts are compared to determine the
  maximum.
*/
PetscErrorCode VecPointwiseMax_Seq(Vec w, Vec x, Vec y);

/*
  Computes the component-wise maximum w[i] = max(x[i], y[i]). This operation is
  performed for each element `i` of the vectors `x` and `y`.

  Parameters:
    - w: Vector to store the result.
    - x: First input vector.
    - y: Second input vector.

  Returns:
    - PetscErrorCode (0 on success, non-zero on failure).

  Note:
    -   Note: For complex numbers, only the real parts are compared to determine
        the maximum.
    - Extracted from petscvec.h
    - URL:
      https://petsc.org/release/manualpages/Vec/VecPointwiseMax/#vecpointwisemax
*/
PetscErrorCode VecPointwiseMax(Vec w, Vec x, Vec y);

/*
  Internal auxiliary function to compute the component-wise
  maximum of the absolute values of two sequential vectors on a single process.

  This function sets each entry of the output vector w to:
      w[i] = max(abs(x[i]), abs(y[i]))

  Parameters:
    - w: Output vector where the component-wise maximum absolute values are
  stored.
    - x: The first input vector.
    - y: The second input vector.

  Returns:
    - PetscErrorCode: 0 on success, non-zero on failure.
*/
PetscErrorCode VecPointwiseMaxAbs_Seq(Vec w, Vec x, Vec y);

/*
  Computes the component-wise maximum of the absolute values of two vectors.
  This operation is performed for each element i of the vectors x and y.

  Parameters:
    - w: Vector to store the result.
    - x: First input vector.
    - y: Second input vector.

  Returns:
    - PetscErrorCode (0 on success, non-zero on failure).

  Note:
    - For complex numbers, only the real parts are compared to determine
      the maximum absolute value.
    - Extracted from petscvec.h
    - URL:
      https://petsc.org/release/manualpages/Vec/VecPointwiseMaxAbs/#vecpointwisemaxabs
*/
PetscErrorCode VecPointwiseMaxAbs(Vec w, Vec x, Vec y);

/*
  Internal auxiliary function to compute the component-wise minimum of two
  sequential vectors on a single process. The function sets each entry of
  the output vector w to the minimum of the corresponding entries in x and
  y, i.e., w[i] = min(x[i], y[i]).

  Parameters:
  - w: Output vector which will hold the component-wise minimum values.
  - x: The first input vector.
  - y: The second input vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: For complex numbers, only the real parts are compared to determine
  the minimum.
*/
PetscErrorCode VecPointwiseMin_Seq(Vec w, Vec x, Vec y);

/*
  Computes the component-wise minimum w[i] = min(x[i], y[i]). This operation is
  performed for each element i of the vectors x and y.

  Parameters:
    - w: Vector to store the result.
    - x: First input vector.
    - y: Second input vector.

  Returns:
    - PetscErrorCode (0 on success, non-zero on failure).

  Note:
    - For complex numbers, only the real parts are compared to determine the
  minimum.
    - Extracted from petscvec.h
    - URL:
      https://petsc.org/release/manualpages/Vec/VecPointwiseMin/#vecpointwisemin
*/
PetscErrorCode VecPointwiseMin(Vec w, Vec x, Vec y);

/*
  Internal auxiliary function for VecPointwiseMax to compute the maximum
  pointwise division of two sequential vectors on a single process.

  Parameters:
    - x: The first input vector (numerator).
    - y: The second input vector (denominator).
    - w: The vector to store the result of the pointwise division.

  Returns:
    - PetscErrorCode: 0 on success, non-zero on failure.

  Note:
    This function is optimized for sequential vectors and is not intended
  for parallel use.

  Extracted from dvecimpl.h
*/
PetscErrorCode VecMaxPointwiseDivide_Seq(Vec xin, Vec yin, PetscReal *max);

/*
  Internal auxiliary function for VecMaxPointwiseDivide to compute the maximum
  pointwise division of two parallel vectors using MPI communication.
  Parameters:
    - xin: The first input vector (numerator).
    - yin: The second input vector (denominator).
    - max: Pointer to store the maximum value of the pointwise division.
  Returns:  PetscErrorCode (0 on success, non-zero on failure).
  Note: This function uses MPI operations to compute the global maximum across
  parallel processes.
  Extracted from pvecimpl.h
*/
PetscErrorCode VecMaxPointwiseDivide_MPI(Vec xin, Vec yin, PetscReal *max);

/*
  Computes the maximum pointwise division of two vectors.
  Parameters:
  - xin: The first input vector (numerator).
  - yin: The second input vector (denominator).
  - max: Pointer to store the maximum value of the pointwise division.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: This function handles both sequential and parallel vectors.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecMaxPointwiseDivide/#vecmaxpointwisedivide
 */
PetscErrorCode VecMaxPointwiseDivide(Vec xin, Vec yin, PetscReal *max);

/*
  Begins assembling the vector.
  Parameters:
  - vec Vector to begin assembling.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Should be called after completing all calls to VecSetValues().
        Ensures all entries are stored on the correct MPI process.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecAssemblyBegin/#vecassemblybegin
 */
PetscErrorCode VecAssemblyBegin(Vec vec);

/*
  Completes assembling the vector.
  Parameters:
  - vec Vector to complete assembling.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Should be called after VecAssemblyBegin().
        Finalizes the assembly of the vector.
    - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecAssemblyEnd/#vecassemblyend
 */
PetscErrorCode VecAssemblyEnd(Vec vec);

// Internal Auxilary function used in VecCopy
PetscErrorCode VecCopyAsync_Private(Vec x, Vec y, PetscDeviceContext dctx);

/*
  Copies one vector to another.

  Parameters:
  - xin: Source vector.
  - yin: Destination vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Supports both real and complex vectors.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecCopy/#veccopy
*/
PetscErrorCode VecCopy(Vec xin, Vec yin);

/*
  Internal auxiliary function for VecCopy optimized for sequential vectors.
  Copies the contents of the source vector to the destination vector without
  parallel communication.

  Parameters:
  - xin: Source vector.
  - yin: Destination vector.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecCopy_Seq(Vec xin, Vec yin);

/*
  Internal auxiliary function to swap the entries of two sequential vectors on a
  single process.

  Parameters:
  - x: The first vector
  - y: The second vector

  Returns:
  - PetscErrorCode: 0 on success, non-zero error code if an error occurs.

  Note:
  - Both vectors must be properly allocated and of equal size.
  - This function operates only on sequential (non-parallel) vectors.
*/
PetscErrorCode VecSwap_Seq(Vec x, Vec y);

/*
  Swaps the contents of two vectors.
  Parameters:
  - x First vector to swap.
  - y Second vector to swap.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecSwap/
*/
PetscErrorCode VecSwap(Vec x, Vec y);

/* Utility function to compute the PETSc norm of a CIVL vector */
void $petsc_norm($vec vec, NormType type, PetscReal *result);

/* Returns string representation of the PETSc norm type */
char *$petsc_norm_name(NormType type);

/*
  Internal auxiliary function for VecNorm optimized for sequential vectors.
  Computes the norm of a sequential vector on a single process.

  Parameters:
  - xin: Input sequential vector.
  - type: Type of norm to compute (e.g., NORM_1, NORM_2, NORM_FROBENIUS,
  NORM_INFINITY, NORM_1_AND_2).
  - z: Pointer to store the computed norm value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecNorm_Seq(Vec xin, NormType type, PetscReal *z);

/*
  Internal auxiliary function for VecNorm optimized for parallel vectors.
  Computes the norm of a parallel vector using MPI communication.

  Parameters:
  - xin: Input parallel vector.
  - type: Type of norm to compute (e.g., NORM_1, NORM_2, NORM_FROBENIUS,
  NORM_INFINITY, NORM_1_AND_2).
  - z: Pointer to store the computed norm value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Uses MPI operations to compute the global norm across parallel
  processes.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecNorm_MPI(Vec xin, NormType type, PetscReal *z);

/*
  Computes the norm of a vector (sequential or parallel).

  Parameters:
  - x: Input vector.
  - type: Type of norm to compute (e.g., NORM_1, NORM_2, NORM_FROBENIUS,
  NORM_INFINITY, NORM_1_AND_2).
  - val: Pointer to store the computed norm value.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Automatically handles both sequential and parallel vector cases.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecNorm/#vecnorm
*/
PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val);

/*
  Checks if the specified norm of a vector has already been computed and is
  available.

  Parameters:
  - x: Input vector.
  - type: Type of norm to check.
  - available: Pointer to store the availability status (true or false).
  - val: Pointer to store the previously computed norm value (if available).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Useful for avoiding redundant norm computations.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecNormAvailable/#vecnormavailable
*/
PetscErrorCode VecNormAvailable(Vec x, NormType type, PetscBool *available,
                                PetscReal *val);

/*
  Normalizes a vector to have a specified norm, usually unit length.

  Parameters:
  - x: Vector to normalize.
  - val: Pointer to store the original norm of the vector before
  normalization.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Typically used to scale vectors to unit length.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecNormalize/#vecnormalize
*/
PetscErrorCode VecNormalize(Vec x, PetscReal *val);

/*
  Gets a read-only pointer to the vector's data array.

  Parameters:
  - x: Input vector.
  - a: Pointer to store the read-only array pointer.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Intended for accessing vector data without modification.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecGetArrayRead/#vecgetarrayread
*/
PetscErrorCode VecGetArrayRead(Vec x, const PetscScalar **a);

/*
  Gets a writable pointer to the vector's data array.

  Parameters:
  - x: Input vector.
  - a: Pointer to store the writable array pointer.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Intended for modifying vector data directly.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecGetArrayWrite/#vecgetarraywrite
*/
PetscErrorCode VecGetArrayWrite(Vec x, PetscScalar **a);

/*
  Gets a writable pointer to the vector's data array.

  Parameters:
  - x: Input vector.
  - a: Pointer to store the writable array pointer.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Provides writable access for direct manipulation of vector data.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecGetArray/#vecgetarray
*/
PetscErrorCode VecGetArray(Vec x, PetscScalar **a);

/*
  Restores the read-only array obtained from VecGetArrayRead.

  Parameters:
  - x: Input vector.
  - a: Pointer to the array to be restored.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Should be called after finishing read-only access.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecRestoreArrayRead/#vecrestorearrayread
*/
PetscErrorCode VecRestoreArrayRead(Vec x, const PetscScalar **a);

/*
  Restores the writable array obtained from VecGetArrayWrite.

  Parameters:
  - x: Input vector.
  - a: Pointer to the array to be restored.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Should be called after finishing modifications.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecRestoreArrayWrite/#vecrestorearraywrite
*/
PetscErrorCode VecRestoreArrayWrite(Vec x, PetscScalar **a);

/*
  Restores the writable array obtained from VecGetArray.

  Parameters:
  - x: Input vector.
  - a: Pointer to the array to be restored.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Should be called after finishing modifications.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecRestoreArray/#vecrestorearray
*/
PetscErrorCode VecRestoreArray(Vec x, PetscScalar **a);

/*
  Sets a single entry in a PETSc vector.

  Parameters:
  - v: The PETSc vector to modify.
  - i: The global index of the entry to set.
  - va: The value to set at the specified index.
  - mode: The insertion mode, either INSERT_VALUES or ADD_VALUES.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: This function is typically used to assemble vectors in parallel. The
  insertion mode determines whether the value replaces the existing value
  (INSERT_VALUES) or is added to it (ADD_VALUES).
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSetValue/#vecsetvalue
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

  Note: The function either inserts or adds values at specified locations in
  the vector. This operation is often used when assembling vectors in parallel
  computations.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecSetValues/#vecsetvalues
*/
PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            const PetscScalar y[], InsertMode iora);

/*
  Internal auxiliary function for VecSetValues optimized for parallel vectors.
  Inserts or adds values into a parallel PETSc vector at specified indices
  using MPI communication.

  Parameters:
  - xin: Input parallel vector.
  - ni: Number of indices for insertion.
  - ix: Array of indices for insertion.
  - y: Array of values to be inserted.
  - addv: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Uses MPI to manage insertion across distributed vector components.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecSetValues_MPI(Vec xin, PetscInt ni, const PetscInt ix[],
                                const PetscScalar y[], InsertMode addv);

/*
  Internal auxiliary function for VecSetValues optimized for sequential
  vectors. Inserts or adds values into a sequential PETSc vector at specified
  indices without parallel communication.

  Parameters:
  - x: Input sequential vector.
  - ni: Number of indices for insertion.
  - ix: Array of indices for insertion.
  - y: Array of values to be inserted.
  - iora: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
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
  - Each block is a contiguous group of elements of size equal to the vector's
  block size.
  - Updates the vector such that x[bs * ix[i] + j] = y[bs * i + j], for j = 0,
  ..., bs-1, where bs is the block size.
  - Indices outside the range owned by the local process are ignored.
  - Calls with INSERT_VALUES and ADD_VALUES cannot be mixed without
  intervening calls to VecAssemblyBegin() and VecAssemblyEnd().
  - Negative indices in ix are ignored to facilitate handling of boundary
  conditions.
  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecSetValuesBlocked/#vecsetvaluesblocked
*/
PetscErrorCode VecSetValuesBlocked(Vec x, PetscInt ni, const PetscInt ix[],
                                   const PetscScalar y[], InsertMode iora);

/*
  Internal auxiliary function for VecSetValuesBlocked optimized for parallel
  vectors. Inserts or adds blocks of values into a parallel PETSc vector using
  MPI communication.

  Parameters:
  - x: Input parallel vector.
  - ni: Number of blocks for insertion.
  - ix: Array of block indices.
  - y: Array of values to be inserted or added, in block format.
  - iora: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Notes:
  - Uses MPI to manage insertion across distributed vector components.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecSetValuesBlocked_MPI(Vec x, PetscInt ni, const PetscInt ix[],
                                       const PetscScalar y[], InsertMode iora);

/*
  Internal auxiliary function for VecSetValuesBlocked optimized for sequential
  vectors. Inserts or adds blocks of values into a sequential PETSc vector
  without parallel communication.

  Parameters:
  - x: Input sequential vector.
  - ni: Number of blocks for insertion.
  - ix: Array of block indices.
  - y: Array of values to be inserted or added, in block format.
  - iora: The insertion mode (INSERT_VALUES or ADD_VALUES).

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecSetValuesBlocked_Seq(Vec x, PetscInt ni, const PetscInt ix[],
                                       const PetscScalar y[], InsertMode iora);

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
  - Retrieves y[i] = x[ix[i]] for i = 0,...,ni-1.
  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecGetValues/#vecgetvalues
*/
PetscErrorCode VecGetValues(Vec x, PetscInt ni, const PetscInt ix[],
                            PetscScalar y[]);

/*
  Internal auxiliary function for VecGetValues optimized for parallel vectors.
  Retrieves values from specified indices in a parallel PETSc vector using MPI
  communication.

  Parameters:
  - xin: Input parallel vector.
  - ni: Number of indices to retrieve.
  - ix: Array of global indices.
  - y: Array to store retrieved values.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Notes:
  - Utilizes MPI to access and retrieve data from distributed vector
  components.
  - Extracted from pvecimpl.h
*/
PetscErrorCode VecGetValues_MPI(Vec xin, PetscInt ni, const PetscInt ix[],
                                PetscScalar y[]);

/*
  Internal auxiliary function for VecGetValues optimized for sequential
  vectors. Retrieves values from specified indices in a sequential PETSc
  vector without parallel communication.

  Parameters:
  - xin: Input sequential vector.
  - ni: Number of indices to retrieve.
  - ix: Array of indices.
  - y: Array to store retrieved values.

  Returns:
  - PetscErrorCode: 0 on success, non-zero on failure.

  Note: Optimized specifically for sequential vectors on a single process.
  - Extracted from dvecimpl.h
*/
PetscErrorCode VecGetValues_Seq(Vec xin, PetscInt ni, const PetscInt ix[],
                                PetscScalar y[]);

/*
  Conjugates each element of the given sequential vector.

  Parameters:
  - xin Vector to be conjugated.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Handles both complex and real vectors depending on the USE_COMPLEX
  macro.
  - Extracted from dvecimpl.h
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

  - Extracted from petscvec.h
  - URL:
  https://petsc.org/release/manualpages/Vec/VecStrideNorm/#vecstridenorm
 */
PetscErrorCode VecStrideNorm(Vec v, PetscInt start, NormType ntype,
                             PetscReal *nrm);

/*
  Destroys a vector and frees its memory.
  Parameters:
  - v Pointer to the vector to be destroyed.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  - Extracted from petscvec.h
  - URL: https://petsc.org/release/manualpages/Vec/VecDestroy/#vecdestroy
 */
PetscErrorCode VecDestroy(Vec *v);

/*
  Finalizes PETSc.

  Returns: PetscErrorCode (Always returns 0 in this implementation).
  - Extracted from petscsys.h
  - URL:
  https://petsc.org/release/manualpages/Sys/PetscFinalize/#petscfinalize
 */
PetscErrorCode PetscFinalize(void);

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

/*
  Computes the Euclidean norm (L2 norm) of a vector.
  Parameters:
  - n Pointer to the number of elements in the vector.
  - x Pointer to the vector elements.
  - stride Pointer to the stride between elements in the vector.

  Returns: PetscReal The computed Euclidean norm.

  Note: For complex numbers, it computes: sqrt(sum(|x[i]|^2))
        It iterates through the vector elements with a specified `stride` and
  sums the squares of the element values. The final result is the square root
  of this sum. For real numbers, it computes: sqrt(sum(x[i] * x[i])) For
  complex numbers, it computes: sqrt(sum(|x[i]|^2))
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
`alpha` is a scalar. It iterates through the vectors with specified strides
`sx` and `sy` respectively.

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
Swaps the elements of two vectors.

For each index i from 0 to n-1, exchanges the element x[i * incx] with y[i *
incy].

Parameters:
- n: Pointer to the number of elements to swap between the vectors.
- x: Pointer to the first vector.
- incx: Pointer to the stride (increment) between successive elements in the
first vector.
- y: Pointer to the second vector.
- incy: Pointer to the stride (increment) between successive elements in the
second vector.

Return:
- PetscErrorCode indicating the success or failure of the operation.
*/
PetscErrorCode BLASswap_(const PetscBLASInt *n, PetscScalar *x,
                         const PetscBLASInt *incx, PetscScalar *y,
                         const PetscBLASInt *incy);

// CIVL-specific targets used to model PETSc concepts...

/*
  Extracts the abstract vector represented by the PETSc vector

  Parameters:
  - petscVec: The PETSc vector to be converted.
  - $vec: The corresponding CIVL vector.
*/
$vec CIVL_PetscToCivlVec(Vec petscVec);

/*
  Copies the contents of a CIVL vector to a PETSc vector.

  Parameters:
  - in: Input CIVL vector.
  - out: Output PETSc vector to store the copied data.
 */
void CIVL_CivlToPetscVecCopy($vec in, Vec out);

/*
  Converts a CIVL vector to a PETSc vector representation.

  Parameters:
  - in: The CIVL vector that contains data to populate the PETSc vector.
  - Vec: The corresponding PETSc vector.
*/
Vec CIVL_CivlToPetscVec($vec in, int n, MPI_Comm comm);

/*
  Prints the contents of a vector based on its type (sequential or parallel).
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Calls specific print targets based on the vector type (VECSEQ or
  VECMPI). Prints complex numbers in the form (a + bi) if USE_COMPLEX is
  defined, otherwise prints real numbers.
 */
void CIVL_PrintVec(const char *name, Vec vin);

/*
  Prints the contents of a sequential vector.
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Prints complex numbers in the form (a + bi) if USE_COMPLEX is
  defined, otherwise prints real numbers.
 */
void CIVL_PrintSeqVec(const char *name, Vec vin);

/*
  Prints the contents of a MPI vector.
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Prints complex numbers in the form (a + bi) if USE_COMPLEX is
  defined, otherwise prints real numbers.
 */
void CIVL_PrintMPIVec(const char *name, Vec vin);

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
  PetscErrorCode (*restorearray)(Vec, PetscScalar **); /* restore data array */
  PetscErrorCode (*restorearraywrite)(Vec, PetscScalar **);
  PetscErrorCode (*getarraywrite)(Vec, PetscScalar **);
  PetscErrorCode (*pointwisemult)(Vec, Vec, Vec);
  PetscErrorCode (*pointwisedivide)(Vec, Vec, Vec);
  PetscErrorCode (*getsize)(Vec, PetscInt *);       /* get size of vector */
  PetscErrorCode (*pointwisemax)(Vec, Vec, Vec);    /* w = max(x,y) */
  PetscErrorCode (*pointwisemin)(Vec, Vec, Vec);    /* w = min(x,y) */
  PetscErrorCode (*pointwisemaxabs)(Vec, Vec, Vec); /* w = max(abs(x),abs(y)) */
  PetscErrorCode (*mdot)(Vec, PetscInt, const Vec[],
                         PetscScalar *); // z[j] = x dot y[j]
  PetscErrorCode (*swap)(Vec, Vec);      // x <-> y
};

#endif

#ifndef _PETSCVEC_H
#define _PETSCVEC_H
#include <complex.h>
#include <float.h>
#include <math.h>
#include <mpi.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// PetscInt is an integer datatype
typedef int PetscInt;

// PetscReal is a datatype representing a real number
typedef double PetscReal;

typedef struct {
  PetscReal real;
  PetscReal imag;
} MyComplex;

#define MY_COMPLEX(a, b) ((MyComplex){(a), (b)})

// PetscScalar is a datatype representing either a real or complex number
#ifdef USE_COMPLEX
typedef MyComplex PetscScalar;
#else
typedef PetscReal PetscScalar;
#endif

// Define PETSC_FALSE and PETSC_TRUE as boolean values
#define PETSC_FALSE 0
#define PETSC_TRUE 1

// my_cabs() - compute the absolute value of a complex number
PetscReal my_cabs(MyComplex z);

// Complex number conjugation
MyComplex my_conj(MyComplex z);

// Complex number addition
MyComplex my_cadd(MyComplex x, MyComplex y);

// Complex number subtraction
MyComplex my_csub(MyComplex x, MyComplex y);

// Complex number multiplication
MyComplex my_cmul(MyComplex x, MyComplex y);

#ifdef USE_COMPLEX
#define PetscConj(a) ((PetscScalar){(a).real, -(a).imag})
#else
#define PetscConj(a) (a)
#endif

// PetscBLASInt is an integer datatype used for BLAS operations
typedef int PetscBLASInt;

// PetscErrorCode is an error code datatype used for error handling
typedef int PetscErrorCode;

// PetscOptions is a datatype representing a set of PETSc options
typedef struct PetscOptions_s *PetscOptions;

// PetscLayout is a datatype representing the layout of a PETSc object
typedef struct _n_PetscLayout *PetscLayout;

// PetscBool is a boolean datatype
typedef _Bool PetscBool;

// MPI_Comm is a datatype representing an MPI communicator
// typedef int MPI_Comm;

// PetscLogDouble is a datatype representing a double precision floating point
// number used for logging
typedef double PetscLogDouble;

typedef const char *VecType;
#define VECSEQ "seq"
#define VECMPI "mpi"
#define VECSTANDARD "standard"

typedef enum {
  PETSC_COPY_VALUES,
  PETSC_OWN_POINTER,
  PETSC_USE_POINTER
} PetscCopyMode;

struct _p_ISLocalToGlobalMapping {
  PetscInt n;
  PetscInt bs;
  PetscInt *indices;
};

typedef struct _p_ISLocalToGlobalMapping *ISLocalToGlobalMapping;

// SimpleMap is a mapping structure
typedef struct map_s {
  PetscInt n;
  PetscInt N;
  PetscInt rstart, rend; /* local start, local end + 1 */
  PetscInt bs;           /* block size */
  ISLocalToGlobalMapping mapping;
} *SimpleMap;

// Vec is a struct representing a vector
struct Vec_s {
  MPI_Comm comm;
  PetscInt block_size;
  PetscScalar *data;
  PetscScalar a, b;
  SimpleMap map;
  VecType type;
};

typedef struct Vec_s *Vec;

// PetscViewer is a datatype representing an object used for viewing PETSc
// objects
typedef struct _p_PetscViewer *PetscViewer;

// PetscViewerFormat is an enum representing different formats for PetscViewer
typedef enum {
  PETSC_VIEWER_DEFAULT,
  PETSC_VIEWER_STDOUT_SELF
} PetscViewerFormat;

struct _p_PetscViewer {
  PetscViewerFormat format;
  int iformat;
  void *data;
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

// PETSC_DECIDE represents a constant used for making a decision
#ifndef PETSC_DECIDE
#define PETSC_DECIDE (-1)
#endif

// PETSC_DETERMINE represents a constant used for determining a value
#ifndef PETSC_DETERMINE
#define PETSC_DETERMINE PETSC_DECIDE
#endif

// PETSC_DEFAULT represents a default value
#define PETSC_DEFAULT (-2)

// PETSC_COMM_WORLD represents the MPI communicator for the entire world
#define PETSC_COMM_WORLD MPI_COMM_WORLD

// PetscInt_FMT is a format specifier for PetscInt used in formatted output
#define PetscInt_FMT "d"

// PetscSqrtReal computes the square root of a real number.
#define PetscSqrtReal(a) sqrt(a)

/*
  Retrieves the imaginary part of a complex/scalar.
  Parameters:
  - a: A complex/scalar value.

  Returns: for complex numbers it extracts the imaginary component else it
  returns the Zero, as the imaginary part is not applicable for real numbers.
 */
#ifdef USE_COMPLEX
#define PetscImaginaryPart(a) ((a).imag)
#else
#define PetscImaginaryPart(a) ((PetscReal)(0))
#endif

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

// PetscCall is a macro used to call a function
#define PetscCall(X) X

// PetscFunctionBeginUser marks the beginning of a user-defined function
#define PetscFunctionBeginUser

// PetscPrintf is a macro used to print formatted output, supporting both real
// and complex numbers
#ifdef USE_COMPLEX
#define PetscPrintf(comm, format, ...) (printf(format, __VA_ARGS__))
#else
#define PetscPrintf(comm, format, ...) (printf(format, __VA_ARGS__))
#endif

// PetscPrintf0 is a modified macro for PetscPrintf used to print formatted
// output with no arguments
#define PetscPrintf0(comm, format) (printf(format))

// PetscFunctionBegin marks the beginning of a Petsc function
#define PetscFunctionBegin PetscErrorCode __ierr = 0;

// PetscFunctionReturn returns an error code from a Petsc function
#define PetscFunctionReturn(err)                                               \
  do {                                                                         \
    __ierr = (err);                                                            \
    return __ierr;                                                             \
  } while (0)

#ifdef USE_COMPLEX
#define PETSC_USE_COMPLEX 1
#else
#define PETSC_USE_COMPLEX 0
#endif

#ifndef PETSC_USE_REAL___FP16
#define PETSC_USE_REAL___FP16 0
#else
#define PETSC_USE_REAL___FP16 1
#endif

// PetscDefined_Internal checks if a macro is defined internally
#define PetscDefined_Internal(x) (x)

// PetscDefined checks if a macro is defined
#define PetscDefined(def) PetscDefined_Internal(PETSC_##def)

// PETSC_EXTERN specifies an external linkage for a variable or function
#define PETSC_EXTERN extern

// PETSC_EXTERN_TLS specifies an external linkage for a thread-local variable
#define PETSC_EXTERN_TLS PETSC_EXTERN

// Modify the PetscAbsScalar macro
#ifdef USE_COMPLEX
#define PetscAbsScalar(a) my_cabs(a)
#else
#define PetscAbsScalar(a) fabs(a)
#endif

#ifdef USE_COMPLEX
#define PetscRealPart(a) ((a).real)
#else
#define PetscRealPart(a) (a)
#endif

// PetscCallBLAS calls a BLAS function
#define PetscCallBLAS(x, X) X

// PetscArraycpy copies elements from one array (str1) to another (str2)
#define PetscArraycpy(str1, str2, cnt)                                         \
  ((sizeof(*(str1)) == sizeof(*(str2)))                                        \
       ? PetscMemcpy((str1), (str2), (size_t)(cnt) * sizeof(*(str1)))          \
       : PETSC_ERR_ARG_SIZ)

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

PetscErrorCode VecGetOwnershipRange(Vec x, PetscInt *low, PetscInt *high);

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

PetscErrorCode PetscLayoutSetBlockSize(SimpleMap map, PetscInt bs);

PetscErrorCode ISLocalToGlobalMappingCreate(MPI_Comm comm, PetscInt bs,
                                            PetscInt n,
                                            const PetscInt indices[],
                                            PetscCopyMode mode,
                                            ISLocalToGlobalMapping *mapping);

PetscErrorCode ISLocalToGlobalMappingDestroy(ISLocalToGlobalMapping *mapping);

/*
  Configures the vector from options.
  Parameters:
  - vec Vector to configure.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Allocates memory for vector data based on the local size.
 */
PetscErrorCode VecSetFromOptions(Vec vec);

/*
  Sets all components of a vector to a single scalar value.
  Parameters:
  - x Vector to set values in.
  - alpha Scalar value to set.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecSet(Vec x, PetscScalar alpha);

PetscErrorCode PetscStrcmp(const char a[], const char b[], PetscBool *flg);

PetscErrorCode VecSetType(Vec vec, VecType newType);

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

/*
  Computes y = y + sum(alpha[i] * x[i]) for multiple vectors. Updates the vector
  `y` by adding scaled versions of vectors `x[i]` weighted by `alpha[i]` for
  each `i` in the range `[0, nv-1]`. Parameters:
  - y Vector to be updated.
  - nv Number of vectors.
  - alpha Array of scalars.
  - x Array of vectors.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecMAXPY(Vec y, PetscInt nv, const PetscScalar alpha[], Vec x[]);

/*
  Computes y = alpha * x + y. Updates the vector `y` by adding the vector `x`
  scaled by the scalar `alpha`. Parameters:
  - y Vector to be updated.
  - alpha Scalar multiplier.
  - x Vector to be added.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Supports both real and complex scalars and vectors.
 */
PetscErrorCode VecAXPY(Vec y, PetscScalar alpha, Vec x);

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

/*
  Computes w = alpha * x + y. Stores the result in the vector `w` by adding the
  vector `y` to `alpha` times the vector `x`. Parameters:
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

/*
  Copies one vector to another.
  Parameters:
  - xin Source vector.
  - yin Destination vector.

  Returns: PetscErrorCode (0 on success, non-zero on failure).
 */
PetscErrorCode VecCopy(Vec xin, Vec yin);

/*
  Computes the norm of a vector.
  Parameters:
  - x Vector for which the norm is computed.
  - type Type of norm to compute (NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY,
  NORM_1_AND_2).
  - val Pointer to store the computed norm value.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: NORM_FROBENIUS is same as L2 norm for vectors.
        NORM_1_AND_2 returns both L1 & L2 norms at same time.
 */
PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val);

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

PetscErrorCode VecSetValues(Vec x, PetscInt ni, const PetscInt ix[], const PetscScalar y[], InsertMode iora);

/*
  Prints to standard out, only from the first MPI process in the communicator.
  Parameters:
  - comm MPI communicator.
  - format Format string, similar to printf.
  - ... Additional arguments to format.

  Returns: PetscErrorCode (0 on success, non-zero on failure).

  Note: Calls from other processes are ignored.
        This function is defined as a macro to the standard `printf` function
  for simplicity.
 */
PetscErrorCode PetscPrintf(MPI_Comm comm, const char format[], ...);

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
  Computes the norm of a subvector of a vector defined by a starting point and a
  stride. Parameters:
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

/*
  Prints the contents of a sequential vector.
  Parameters:
  - name Name or label for the vector to be printed.
  - vin Input vector to be printed.

  Note: Prints complex numbers in the form (a + bi) if USE_COMPLEX is defined,
        otherwise prints real numbers.
 */
void vecprint_seq(const char *name, Vec vin);

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

#endif

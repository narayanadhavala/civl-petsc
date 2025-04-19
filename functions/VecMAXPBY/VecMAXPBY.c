#include <petscvec.h>
#undef VecMAXPBY

PetscErrorCode VecMAXPBY(Vec y, PetscInt nv, const PetscScalar alpha[],
                         PetscScalar beta, Vec x[]) {
#ifdef DEBUG
  $print("Target VecMAXPBY: alpha[0]=", alpha[0], " beta=", beta, " nv =", nv,
         "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(y, VEC_CLASSID, 1);
  VecCheckAssembled(y);
  PetscValidLogicalCollectiveInt(y, nv, 2);
  PetscCall(VecSetErrorIfLocked(y, 1));
  PetscCheck(nv >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE,
             "Number of vectors (given %" PetscInt_FMT ") cannot be negative",
             nv);

  PetscValidLogicalCollectiveScalar(y, beta, 4);
  if (y->ops->maxpby) {
    $print("y->ops->maxpby is true");
    PetscInt zeros = 0;

    if (nv) {
      PetscAssertPointer(alpha, 3);
      PetscAssertPointer(x, 5);
    }
    /* Change by Venkata: replaced to avoid direct scalar operations and type
     * casts, which CIVL doesn't support */
    for (PetscInt i = 0; i < nv; ++i) { // scan all alpha[]
      PetscValidLogicalCollectiveScalar(y, alpha[i], 3);
      PetscValidHeaderSpecific(x[i], VEC_CLASSID, 5);
      PetscValidType(x[i], 5);
      PetscCheckSameTypeAndComm(y, 1, x[i], 5);
      VecCheckSameSize(y, 1, x[i], 5);
      PetscCheck(y != x[i], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG,
                 "Array of vectors 'x' cannot contain y, found x[%" PetscInt_FMT
                 "] == y",
                 i);
      VecCheckAssembled(x[i]);
      PetscCall(VecLockReadPush(x[i]));
      zeros += scalar_eq(alpha[i], scalar_zero);
    }

    if (zeros < nv) { // has nonzero alpha
      PetscCall(PetscLogEventBegin(VEC_MAXPY, y, *x, 0, 0));
      PetscUseTypeMethod(y, maxpby, nv, alpha, beta, x);
      PetscCall(PetscLogEventEnd(VEC_MAXPY, y, *x, 0, 0));
      PetscCall(PetscObjectStateIncrease((PetscObject)y));
    } else {
      PetscCall(VecScale(y, beta));
    }
    for (PetscInt i = 0; i < nv; ++i)
      PetscCall(VecLockReadPop(x[i]));
  } else { // no maxpby
           // used the scalar_eq to compare the complex & real numbers
    if (scalar_eq(scalar_of(0.0), beta))
      PetscCall(VecSet(y, scalar_of(0.0)));
    else
      PetscCall(VecScale(y, beta));
    PetscCall(VecMAXPY(y, nv, alpha, x));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecConcatenate

PetscErrorCode VecConcatenate(PetscInt nx, const Vec X[], Vec *Y, IS *x_is[]) {
  MPI_Comm comm;
  VecType vec_type;
  Vec Ytmp, Xtmp;
  IS *is_tmp;
  PetscInt i, shift = 0, Xnl, Xng, Xbegin;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(*X, nx, 1);
  PetscValidHeaderSpecific(*X, VEC_CLASSID, 2);
  PetscValidType(*X, 2);
  PetscAssertPointer(Y, 3);

  if ((*X)->ops->concatenate) {
    /* use the dedicated concatenation function if available */
    /* modified as CIVL fails in explicitly dereferences (*X)->ops->concatenate)
     * before calling it */
    PetscCall(((*X)->ops->concatenate)(nx, X, Y, x_is));
  } else {
    /* loop over vectors and start creating IS */
    comm = PetscObjectComm((PetscObject)(*X));
    PetscCall(VecGetType(*X, &vec_type));
    PetscCall(PetscMalloc1(nx, &is_tmp));
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSize(X[i], &Xng));
      PetscCall(VecGetLocalSize(X[i], &Xnl));
      PetscCall(VecGetOwnershipRange(X[i], &Xbegin, NULL));
      PetscCall(ISCreateStride(comm, Xnl, shift + Xbegin, 1, &is_tmp[i]));
      shift += Xng;
    }
    /* create the concatenated vector */
    PetscCall(VecCreate(comm, &Ytmp));
    PetscCall(VecSetType(Ytmp, vec_type));
    PetscCall(VecSetSizes(Ytmp, PETSC_DECIDE, shift));
    PetscCall(VecSetUp(Ytmp));
    /* copy data from X array to Y and return */
    for (i = 0; i < nx; i++) {
      PetscCall(VecGetSubVector(Ytmp, is_tmp[i], &Xtmp));
      PetscCall(VecCopy(X[i], Xtmp));
      PetscCall(VecRestoreSubVector(Ytmp, is_tmp[i], &Xtmp));
    }
    *Y = Ytmp;
    if (x_is) {
      *x_is = is_tmp;
    } else {
      for (i = 0; i < nx; i++)
        PetscCall(ISDestroy(&is_tmp[i]));
      PetscCall(PetscFree(is_tmp));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

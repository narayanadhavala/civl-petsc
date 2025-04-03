#include <petscvec.h>
#undef VecSet

PetscErrorCode VecSetAsync_Private(Vec x, PetscScalar alpha,
                                   PetscDeviceContext dctx) {
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveScalar(x, alpha, 2);
  // PetscCall(VecSetErrorIfLocked(x, 1));

  if (PetscRealPart(alpha) == 0) {
    PetscReal norm;
    PetscBool set;

    PetscCall(VecNormAvailable(x, NORM_2, &set, &norm));
    if (set == PETSC_TRUE && norm == 0)
      PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscLogEventBegin(VEC_Set, x, 0, 0, 0));
  VecMethodDispatch(x, dctx, VecAsyncFnName(Set), set,
                    (Vec, PetscScalar, PetscDeviceContext), alpha);
  PetscCall(PetscLogEventEnd(VEC_Set, x, 0, 0, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)x));

  /*  norms can be simply set (if |alpha|*N not too large) */

  {
    PetscReal val = PetscAbsScalar(alpha);
    const PetscInt N = x->map->N;

    if (N == 0) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_1],
                                               0.0l));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,
                                               NormIds[NORM_INFINITY], 0.0));
      PetscCall(
          PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_2], 0.0));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,
                                               NormIds[NORM_FROBENIUS], 0.0));
    } else if (val > PETSC_MAX_REAL / N) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,
                                               NormIds[NORM_INFINITY], val));
    } else {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_1],
                                               N * val));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,
                                               NormIds[NORM_INFINITY], val));
      val *= PetscSqrtReal((PetscReal)N);
      PetscCall(
          PetscObjectComposedDataSetReal((PetscObject)x, NormIds[NORM_2], val));
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x,
                                               NormIds[NORM_FROBENIUS], val));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSet(Vec x, PetscScalar alpha) {
#ifdef DEBUG
  $print("DEBUG: Target VecSet called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecSetAsync_Private(x, alpha, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

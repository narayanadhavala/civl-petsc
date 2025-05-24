#include <petscvec.h>
#undef VecScale

PetscErrorCode VecScaleAsync_Private(Vec x, PetscScalar alpha,
                                     PetscDeviceContext dctx) {
  PetscReal norms[4];
  PetscBool flgs[4];
  PetscReal one = 1.0;

#ifdef DEBUG
  $print("Target VecScaleAsync_Private: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscCall(VecSetErrorIfLocked(x, 1));
  /* Change by Venkata: replaced to avoid direct scalar operations and type
   * casts, which CIVL doesn't support */
  if (scalar_eq(scalar_of(1), alpha))
    PetscFunctionReturn(PETSC_SUCCESS);

  /* get current stashed norms */
  for (PetscInt i = 0; i < 4; i++) {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i],
                                             &norms[i], &flgs[i]));
  }

  PetscCall(PetscLogEventBegin(VEC_Scale, x, 0, 0, 0));
  VecMethodDispatch(x, dctx, VecAsyncFnName(Scale), scale,
                    (Vec, PetscScalar, PetscDeviceContext), alpha);
  PetscCall(PetscLogEventEnd(VEC_Scale, x, 0, 0, 0));

  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  /* put the scaled stashed norms back into the Vec */
  for (PetscInt i = 0; i < 4; i++) {
    PetscReal ar = PetscAbsScalar(alpha);
    if (flgs[i]) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[i],
                                               ar * norms[i]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecScale(Vec x, PetscScalar alpha) {
#ifdef DEBUG
  $print("Target VecScale: alpha=", alpha, "\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecScaleAsync_Private(x, alpha, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecSwap

PetscErrorCode VecSwapAsync_Private(Vec x, Vec y, PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("DEBUG: Target VecSwapAsync_Private called\n");
#endif
  PetscReal normxs[4], normys[4];
  PetscBool flgxs[4], flgys[4];

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 2);
  PetscValidType(x, 1);
  PetscValidType(y, 2);
  PetscCheckSameTypeAndComm(x, 1, y, 2);
  VecCheckSameSize(x, 1, y, 2);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecSetErrorIfLocked(x, 1));
  PetscCall(VecSetErrorIfLocked(y, 2));

  for (PetscInt i = 0; i < 4; i++) {
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)x, NormIds[i],
                                             &normxs[i], &flgxs[i]));
    PetscCall(PetscObjectComposedDataGetReal((PetscObject)y, NormIds[i],
                                             &normys[i], &flgys[i]));
  }

  PetscCall(PetscLogEventBegin(VEC_Swap, x, y, 0, 0));
  VecMethodDispatch(x, dctx, VecAsyncFnName(Swap), swap,
                    (Vec, Vec, PetscDeviceContext), y);
  PetscCall(PetscLogEventEnd(VEC_Swap, x, y, 0, 0));

  PetscCall(PetscObjectStateIncrease((PetscObject)x));
  PetscCall(PetscObjectStateIncrease((PetscObject)y));
  for (PetscInt i = 0; i < 4; i++) {
    if (flgxs[i]) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)y, NormIds[i],
                                               normxs[i]));
    }
    if (flgys[i]) {
      PetscCall(PetscObjectComposedDataSetReal((PetscObject)x, NormIds[i],
                                               normys[i]));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecSwap(Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Target VecSwap called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecSwapAsync_Private(x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

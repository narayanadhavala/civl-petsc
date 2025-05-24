#include <petscvec.h>
#undef VecPointwiseDivide
PETSC_EXTERN PetscLogEvent VEC_PointwiseDivide;

static PetscErrorCode
VecPointwiseApply_Private(Vec w, Vec x, Vec y, PetscDeviceContext dctx,
                          PetscLogEvent event, const char async_name[],
                          PetscErrorCode (*const pointwise_op)(Vec, Vec, Vec)) {
  PetscErrorCode (*async_fn)(Vec, Vec, Vec, PetscDeviceContext) = NULL;
#ifdef DEBUG
  $print("DEBUG: Target VecPointwiseApply_Private called\n");
#endif
  PetscFunctionBegin;
  PetscValidHeaderSpecific(w, VEC_CLASSID, 1);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 2);
  PetscValidHeaderSpecific(y, VEC_CLASSID, 3);
  PetscValidType(w, 1);
  PetscValidType(x, 2);
  PetscValidType(y, 3);
  PetscCheckSameTypeAndComm(x, 2, y, 3);
  PetscCheckSameTypeAndComm(y, 3, w, 1);
  VecCheckSameSize(w, 1, x, 2);
  VecCheckSameSize(w, 1, y, 3);
  VecCheckAssembled(x);
  VecCheckAssembled(y);
  PetscCall(VecSetErrorIfLocked(w, 1));
  PetscValidFunction(pointwise_op, 5);

  if (dctx)
    PetscCall(PetscObjectQueryFunction((PetscObject)w, async_name, &async_fn));
  if (event)
    PetscCall(PetscLogEventBegin(event, x, y, w, 0));
  if (async_fn)
    PetscCall((*async_fn)(w, x, y, dctx));
  else
    PetscCall((*pointwise_op)(w, x, y));
  if (event)
    PetscCall(PetscLogEventEnd(event, x, y, w, 0));
  PetscCall(PetscObjectStateIncrease((PetscObject)w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecPointwiseDivideAsync_Private(Vec w, Vec x, Vec y,
                                               PetscDeviceContext dctx) {
#ifdef DEBUG
  $print("DEBUG: Target VecPointwiseDivideAsync_Private called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecPointwiseApply_Private(w, x, y, dctx, VEC_PointwiseDivide,
                                      VecAsyncFnName(PointwiseDivide),
                                      w->ops->pointwisedivide));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecPointwiseDivide(Vec w, Vec x, Vec y) {
#ifdef DEBUG
  $print("DEBUG: Target VecPointwiseDivide called\n");
#endif
  PetscFunctionBegin;
  PetscCall(VecPointwiseDivideAsync_Private(w, x, y, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

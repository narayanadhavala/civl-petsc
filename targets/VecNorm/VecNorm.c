#include <petscvec.h>
#undef VecNorm

PetscErrorCode VecNorm(Vec x, NormType type, PetscReal *val) {
#ifdef DEBUG
  $print("Target VecNorm: type=", type, "\n");
#endif
  PetscBool flg = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(x, VEC_CLASSID, 1);
  PetscValidType(x, 1);
  VecCheckAssembled(x);
  PetscValidLogicalCollectiveEnum(x, type, 2);
  PetscAssertPointer(val, 3);

  PetscCall(VecNormAvailable(x, type, &flg, val));
  // check that all MPI processes call this routine together and have same
  // availability
  if (PetscDefined(USE_DEBUG)) {
    PetscMPIInt b0 = (PetscMPIInt)flg, b1[2], b2[2];
    b1[0] = -b0;
    b1[1] = b0;
    PetscCall(MPIU_Allreduce(b1, b2, 2, MPI_INT, MPI_MAX,
                             PetscObjectComm((PetscObject)x)));
    PetscCheck(
        -b2[0] == b2[1], PetscObjectComm((PetscObject)x),
        PETSC_ERR_ARG_WRONGSTATE,
        "Some MPI processes have cached norm, others do not. This may happen "
        "when some MPI processes call VecGetArray() and some others do not.");
  }
  if (flg)
    PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(VecLockReadPush(x));
  PetscCall(PetscLogEventBegin(VEC_Norm, x, 0, 0, 0));
  PetscUseTypeMethod(x, norm, type, val);
  PetscCall(PetscLogEventEnd(VEC_Norm, x, 0, 0, 0));
  PetscCall(VecLockReadPop(x));

  if (type != NORM_1_AND_2) {
    PetscCall(
        PetscObjectComposedDataSetReal((PetscObject)x, NormIds[type], *val));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

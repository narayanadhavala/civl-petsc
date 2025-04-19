#include <petscvec.h>
#undef VecMin_MPI

static inline PetscErrorCode VecMinMax_MPI_Default(
    Vec xin, PetscInt *idx, PetscReal *z,
    PetscErrorCode (*VecMinMax_SeqFn)(Vec, PetscInt *, PetscReal *),
    const MPI_Op ops[2]) {
#ifdef DEBUG
  $print("DEBUG: Target VecMin_MPI called\n");
#endif
  PetscFunctionBegin;
  /* Find the local max */
  PetscCall(VecMinMax_SeqFn(xin, idx, z));
  if (PetscDefined(HAVE_MPIUNI))
    PetscFunctionReturn(PETSC_SUCCESS);
  /* Change by Venkata: removed MPI_IN_PLACE reduction for MPI_REAL_INT as CIVL
   * doesn't support it */
  /* Find the global max */
  if (idx) {
    PetscReal local_val = *z;
    PetscReal global_val;
    /* Reduce the values to obtain the global maximum value */
    PetscCall(MPI_Allreduce(&local_val, &global_val, 1, MPIU_REAL, ops[1],
                            PetscObjectComm((PetscObject)xin)));
    /* Determine a candidate index: if this process’s local value equals the
       global maximum then use its global index, otherwise report a large
       number. (Here PETSC_MAX_INT is assumed to be a very large PetscInt
       constant.) */
    PetscInt local_idx =
        (local_val == global_val) ? (*idx + xin->map->rstart) : PETSC_MAX_INT;
    PetscInt global_idx;
    /* Reduce the indices, choosing the smallest index among those that held
       the global maximum value. */
    PetscCall(MPI_Allreduce(&local_idx, &global_idx, 1, MPI_INT, MPI_MIN,
                            PetscObjectComm((PetscObject)xin)));
    *z = global_val;
    *idx = global_idx;
  } else {
    /* If the user does not need the index, simply reduce the value */
    PetscCall(MPI_Allreduce(z, z, 1, MPIU_REAL, ops[1],
                            PetscObjectComm((PetscObject)xin)));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecMin_MPI(Vec xin, PetscInt *idx, PetscReal *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecMin_MPI called\n");
#endif
  const MPI_Op ops[] = {MPIU_MINLOC, MPIU_MIN};

  PetscFunctionBegin;
  PetscCall(VecMinMax_MPI_Default(xin, idx, z, VecMin_Seq, ops));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petscvec.h>
#undef VecNorm_Seq

PetscErrorCode VecNorm_Seq(Vec xin, NormType type, PetscReal *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecNorm_Seq\n");
#endif
  // use a local variable to ensure compiler doesn't think z aliases any of the
  // other arrays
  PetscReal ztmp[] = {0.0, 0.0};
  const PetscInt n = xin->map->n;

  PetscFunctionBegin;
  if (n) {
    const PetscScalar *xx;
    const PetscBLASInt one = 1;
    PetscBLASInt bn = 0;

    PetscCall(PetscBLASIntCast(n, &bn));
    PetscCall(VecGetArrayRead(xin, &xx));
    if (type == NORM_2 || type == NORM_FROBENIUS) {
    NORM_1_AND_2_DOING_NORM_2:
      if (PetscDefined(USE_REAL___FP16)) {
        PetscCallBLAS("BLASnrm2",
                      ztmp[type == NORM_1_AND_2] = BLASnrm2_(&bn, xx, &one));
      } else {
        PetscCallBLAS("BLASdot",
                      ztmp[type == NORM_1_AND_2] = PetscSqrtReal(
                          PetscRealPart(BLASdot_(&bn, xx, &one, xx, &one))));
      }
      PetscCall(PetscLogFlops(2.0 * n - 1));
    } else if (type == NORM_INFINITY) {
      for (PetscInt i = 0; i < n; ++i) {
        const PetscReal tmp = PetscAbsScalar(xx[i]);

        // check special case of tmp == NaN
        if ((tmp > ztmp[0]) || (tmp != tmp)) {
          ztmp[0] = tmp;
          if (tmp != tmp)
            break;
        }
      }
    } else if (type == NORM_1 || type == NORM_1_AND_2) {
      if (PetscDefined(USE_COMPLEX)) {
        // BLASasum() returns the nonstandard 1 norm of the 1 norm of the
        // complex entries so we provide a custom loop instead
        for (PetscInt i = 0; i < n; ++i)
          ztmp[0] += PetscAbsScalar(xx[i]);
      } else {
        PetscCallBLAS("BLASasum", ztmp[0] = BLASasum_(&bn, xx, &one));
      }
      PetscCall(PetscLogFlops(n - 1.0));
      // slight reshuffle so we can skip getting the array again (but still log
      // the flops) if we do norm2 after this
      if (type == NORM_1_AND_2)
        goto NORM_1_AND_2_DOING_NORM_2;
    }
    PetscCall(VecRestoreArrayRead(xin, &xx));
  }
  z[0] = ztmp[0];
  if (type == NORM_1_AND_2)
    z[1] = ztmp[1];
  PetscFunctionReturn(PETSC_SUCCESS);
}

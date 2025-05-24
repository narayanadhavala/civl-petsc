#include <petscvec.h>
#undef VecMDot_Seq

PetscErrorCode VecMDot_Seq(Vec xin, PetscInt nv, const Vec yin[],
                           PetscScalar *z) {
#ifdef DEBUG
  $print("DEBUG: Target VecMDot_Seq called\n");
#endif
  const PetscInt n = xin->map->n;
  PetscInt i = nv, j = n, nv_rem = nv & 0x3, j_rem;
  PetscScalar sum0 = scalar_zero, sum1 = scalar_zero, sum2 = scalar_zero, sum3,
              x0, x1, x2, x3;
  const PetscScalar *yy0, *yy1, *yy2, *yy3, *x, *xbase;
  const Vec *yy = (Vec *)yin;

  PetscFunctionBegin;
  if (n == 0) {
    PetscCall(PetscArrayzero(z, nv));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetArrayRead(xin, &xbase));
  x = xbase;
  switch (nv_rem) {
  case 3:
    PetscCall(VecGetArrayRead(yy[0], &yy0));
    PetscCall(VecGetArrayRead(yy[1], &yy1));
    PetscCall(VecGetArrayRead(yy[2], &yy2));
    switch (j_rem = j & 0x3) {
    case 3:
      x2 = x[2];
      sum0 = scalar_add(sum0, scalar_mul(x2, PetscConj(yy0[2])));
      sum1 = scalar_add(sum1, scalar_mul(x2, PetscConj(yy1[2])));
      sum2 = scalar_add(sum2,
                        scalar_mul(x2, PetscConj(yy2[2]))); /* fall through */
    case 2:
      x1 = x[1];
      sum0 = scalar_add(sum0, scalar_mul(x1, PetscConj(yy0[1])));
      sum1 = scalar_add(sum1, scalar_mul(x1, PetscConj(yy1[1])));
      sum2 = scalar_add(sum2,
                        scalar_mul(x1, PetscConj(yy2[1]))); /* fall through */
    case 1:
      x0 = x[0];
      sum0 = scalar_add(sum0, scalar_mul(x0, PetscConj(yy0[0])));
      sum1 = scalar_add(sum1, scalar_mul(x0, PetscConj(yy1[0])));
      sum2 = scalar_add(sum2,
                        scalar_mul(x0, PetscConj(yy2[0]))); /* fall through */
    case 0:
      x += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      j -= j_rem;
      break;
    }
    while (j > 0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 = scalar_add(
          sum0, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy0[0])),
                                      scalar_mul(x1, PetscConj(yy0[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy0[2])),
                                      scalar_mul(x3, PetscConj(yy0[3])))));
      yy0 += 4;
      sum1 = scalar_add(
          sum1, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy1[0])),
                                      scalar_mul(x1, PetscConj(yy1[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy1[2])),
                                      scalar_mul(x3, PetscConj(yy1[3])))));
      yy1 += 4;
      sum2 = scalar_add(
          sum2, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy2[0])),
                                      scalar_mul(x1, PetscConj(yy2[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy2[2])),
                                      scalar_mul(x3, PetscConj(yy2[3])))));
      yy2 += 4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    PetscCall(VecRestoreArrayRead(yy[0], &yy0));
    PetscCall(VecRestoreArrayRead(yy[1], &yy1));
    PetscCall(VecRestoreArrayRead(yy[2], &yy2));
    break;
  case 2:
    PetscCall(VecGetArrayRead(yy[0], &yy0));
    PetscCall(VecGetArrayRead(yy[1], &yy1));
    switch (j_rem = j & 0x3) {
    case 3:
      x2 = x[2];
      sum0 = scalar_add(sum0, scalar_mul(x2, PetscConj(yy0[2])));
      sum1 = scalar_add(sum1,
                        scalar_mul(x2, PetscConj(yy1[2]))); /* fall through */
    case 2:
      x1 = x[1];
      sum0 = scalar_add(sum0, scalar_mul(x1, PetscConj(yy0[1])));
      sum1 = scalar_add(sum1,
                        scalar_mul(x1, PetscConj(yy1[1]))); /* fall through */
    case 1:
      x0 = x[0];
      sum0 = scalar_add(sum0, scalar_mul(x0, PetscConj(yy0[0])));
      sum1 = scalar_add(sum1,
                        scalar_mul(x0, PetscConj(yy1[0]))); /* fall through */
    case 0:
      x += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      j -= j_rem;
      break;
    }
    while (j > 0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 = scalar_add(
          sum0, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy0[0])),
                                      scalar_mul(x1, PetscConj(yy0[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy0[2])),
                                      scalar_mul(x3, PetscConj(yy0[3])))));
      yy0 += 4;
      sum1 = scalar_add(
          sum1, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy1[0])),
                                      scalar_mul(x1, PetscConj(yy1[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy1[2])),
                                      scalar_mul(x3, PetscConj(yy1[3])))));
      yy1 += 4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;

    PetscCall(VecRestoreArrayRead(yy[0], &yy0));
    PetscCall(VecRestoreArrayRead(yy[1], &yy1));
    break;
  case 1:
    PetscCall(VecGetArrayRead(yy[0], &yy0));
    switch (j_rem = j & 0x3) {
    case 3:
      x2 = x[2];
      sum0 = scalar_add(sum0,
                        scalar_mul(x2, PetscConj(yy0[2]))); /* fall through */
    case 2:
      x1 = x[1];
      sum0 = scalar_add(sum0,
                        scalar_mul(x1, PetscConj(yy0[1]))); /* fall through */
    case 1:
      x0 = x[0];
      sum0 = scalar_add(sum0,
                        scalar_mul(x0, PetscConj(yy0[0]))); /* fall through */
    case 0:
      x += j_rem;
      yy0 += j_rem;
      j -= j_rem;
      break;
    }
    while (j > 0) {
      sum0 = scalar_add(
          sum0, scalar_add(scalar_add(scalar_mul(x[0], PetscConj(yy0[0])),
                                      scalar_mul(x[1], PetscConj(yy0[1]))),
                           scalar_add(scalar_mul(x[2], PetscConj(yy0[2])),
                                      scalar_mul(x[3], PetscConj(yy0[3])))));
      yy0 += 4;
      j -= 4;
      x += 4;
    }
    z[0] = sum0;

    PetscCall(VecRestoreArrayRead(yy[0], &yy0));
    break;
  case 0:
    break;
  }
  z += nv_rem;
  i -= nv_rem;
  yy += nv_rem;

  while (i > 0) {
    sum0 = scalar_zero;
    sum1 = scalar_zero;
    sum2 = scalar_zero;
    sum3 = scalar_zero;
    PetscCall(VecGetArrayRead(yy[0], &yy0));
    PetscCall(VecGetArrayRead(yy[1], &yy1));
    PetscCall(VecGetArrayRead(yy[2], &yy2));
    PetscCall(VecGetArrayRead(yy[3], &yy3));

    j = n;
    x = xbase;
    switch (j_rem = j & 0x3) {
    case 3:
      x2 = x[2];
      sum0 = scalar_add(sum0, scalar_mul(x2, PetscConj(yy0[2])));
      sum1 = scalar_add(sum1, scalar_mul(x2, PetscConj(yy1[2])));
      sum2 = scalar_add(sum2, scalar_mul(x2, PetscConj(yy2[2])));
      sum3 = scalar_add(sum3,
                        scalar_mul(x2, PetscConj(yy3[2]))); /* fall through */
    case 2:
      x1 = x[1];
      sum0 = scalar_add(sum0, scalar_mul(x1, PetscConj(yy0[1])));
      sum1 = scalar_add(sum1, scalar_mul(x1, PetscConj(yy1[1])));
      sum2 = scalar_add(sum2, scalar_mul(x1, PetscConj(yy2[1])));
      sum3 = scalar_add(sum3,
                        scalar_mul(x1, PetscConj(yy3[1]))); /* fall through */
    case 1:
      x0 = x[0];
      sum0 = scalar_add(sum0, scalar_mul(x0, PetscConj(yy0[0])));
      sum1 = scalar_add(sum1, scalar_mul(x0, PetscConj(yy1[0])));
      sum2 = scalar_add(sum2, scalar_mul(x0, PetscConj(yy2[0])));
      sum3 = scalar_add(sum3,
                        scalar_mul(x0, PetscConj(yy3[0]))); /* fall through */
    case 0:
      x += j_rem;
      yy0 += j_rem;
      yy1 += j_rem;
      yy2 += j_rem;
      yy3 += j_rem;
      j -= j_rem;
      break;
    }
    while (j > 0) {
      x0 = x[0];
      x1 = x[1];
      x2 = x[2];
      x3 = x[3];
      x += 4;

      sum0 = scalar_add(
          sum0, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy0[0])),
                                      scalar_mul(x1, PetscConj(yy0[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy0[2])),
                                      scalar_mul(x3, PetscConj(yy0[3])))));
      yy0 += 4;

      sum1 = scalar_add(
          sum1, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy1[0])),
                                      scalar_mul(x1, PetscConj(yy1[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy1[2])),
                                      scalar_mul(x3, PetscConj(yy1[3])))));
      yy1 += 4;

      sum2 = scalar_add(
          sum2, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy2[0])),
                                      scalar_mul(x1, PetscConj(yy2[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy2[2])),
                                      scalar_mul(x3, PetscConj(yy2[3])))));
      yy2 += 4;

      sum3 = scalar_add(
          sum3, scalar_add(scalar_add(scalar_mul(x0, PetscConj(yy3[0])),
                                      scalar_mul(x1, PetscConj(yy3[1]))),
                           scalar_add(scalar_mul(x2, PetscConj(yy3[2])),
                                      scalar_mul(x3, PetscConj(yy3[3])))));
      yy3 += 4;
      j -= 4;
    }
    z[0] = sum0;
    z[1] = sum1;
    z[2] = sum2;
    z[3] = sum3;
    z += 4;
    i -= 4;
    PetscCall(VecRestoreArrayRead(yy[0], &yy0));
    PetscCall(VecRestoreArrayRead(yy[1], &yy1));
    PetscCall(VecRestoreArrayRead(yy[2], &yy2));
    PetscCall(VecRestoreArrayRead(yy[3], &yy3));
    yy += 4;
  }
  PetscCall(VecRestoreArrayRead(xin, &xbase));
  PetscCall(PetscLogFlops(PetscMax(nv * (2.0 * n - 1), 0.0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

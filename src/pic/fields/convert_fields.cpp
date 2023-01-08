#include "wrapper.h"

#include "fields.h"
#include "pic.h"

namespace ntt {
  template <>
  void PIC<Dim1>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }

  template <>
  void PIC<Dim2>::InterpolateAndConvertFieldsToHat() {
    auto& mblock = this->meshblock;
    auto  ni1    = mblock.Ni1();
    auto  ni2    = mblock.Ni2();
    Kokkos::deep_copy(mblock.bckp, mblock.em);
    Kokkos::parallel_for(
      "int_and_conv", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO };
        e_cntr[0] = INV_2 * (mblock.em(i, j, em::ex1) + mblock.em(i, j + 1, em::ex1));
        e_cntr[1] = INV_2 * (mblock.em(i, j, em::ex2) + mblock.em(i + 1, j, em::ex2));
        e_cntr[2] = INV_4
                    * (mblock.em(i, j, em::ex3) + mblock.em(i + 1, j, em::ex3)
                       + mblock.em(i, j + 1, em::ex3) + mblock.em(i + 1, j + 1, em::ex3));
        b_cntr[0] = INV_2 * (mblock.em(i, j, em::bx1) + mblock.em(i + 1, j, em::bx1));
        b_cntr[1] = INV_2 * (mblock.em(i, j, em::bx2) + mblock.em(i, j + 1, em::bx2));
        b_cntr[2] = mblock.em(i, j, em::bx3);

        // convert to hat
        vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, e_cntr, e_hat);
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, b_cntr, b_hat);
        mblock.bckp(i, j, em::ex1) = e_hat[0];
        mblock.bckp(i, j, em::ex2) = e_hat[1];
        mblock.bckp(i, j, em::ex3) = e_hat[2];
        mblock.bckp(i, j, em::bx1) = b_hat[0];
        mblock.bckp(i, j, em::bx2) = b_hat[1];
        mblock.bckp(i, j, em::bx3) = b_hat[2];
      });
  }

  template <>
  void PIC<Dim3>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }

}    // namespace ntt
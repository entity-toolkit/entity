#include "wrapper.h"

#include "fields.h"
#include "pic.h"

namespace ntt {

  template <>
  void PIC<Dim2>::InterpolateAndConvertFieldsToHat() {
    auto& mblock = this->meshblock;
    auto  ni1    = mblock.Ni1();
    auto  ni2    = mblock.Ni2();

    AssertEmptyContent(mblock.bckp_content);
    Kokkos::deep_copy(mblock.bckp, mblock.em);
    ImposeContent(mblock.bckp_content, mblock.em_content);

    AssertEmptyContent(mblock.buff_content);
    Kokkos::deep_copy(mblock.buff, mblock.cur);
    ImposeContent(mblock.buff_content, mblock.cur_content);
    ImposeEmptyContent(mblock.cur_content);

    AssertContent(mblock.em_content,
                  { Content::ex1_cntrv,
                    Content::ex2_cntrv,
                    Content::ex3_cntrv,
                    Content::bx1_cntrv,
                    Content::bx2_cntrv,
                    Content::bx3_cntrv });
    AssertContent(mblock.buff_content,
                  { Content::jx1_cntrv, Content::jx2_cntrv, Content::jx3_cntrv });

    Kokkos::parallel_for(
      "InterpolateAndConvertFieldsToHat",
      mblock.rangeActiveCells(),
      Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO }, j_cntr { ZERO };
        // from edges
        e_cntr[0] = INV_2 * (mblock.em(i, j, em::ex1) + mblock.em(i, j + 1, em::ex1));
        e_cntr[1] = INV_2 * (mblock.em(i, j, em::ex2) + mblock.em(i + 1, j, em::ex2));
        e_cntr[2] = INV_4
                    * (mblock.em(i, j, em::ex3) + mblock.em(i + 1, j, em::ex3)
                       + mblock.em(i, j + 1, em::ex3) + mblock.em(i + 1, j + 1, em::ex3));
        j_cntr[0] = INV_2 * (mblock.buff(i, j, cur::jx1) + mblock.buff(i, j + 1, cur::jx1));
        j_cntr[1] = INV_2 * (mblock.buff(i, j, cur::jx2) + mblock.buff(i + 1, j, cur::jx2));
        j_cntr[2]
          = INV_4
            * (mblock.buff(i, j, cur::jx3) + mblock.buff(i + 1, j, cur::jx3)
               + mblock.buff(i, j + 1, cur::jx3) + mblock.buff(i + 1, j + 1, cur::jx3));
        // from faces
        b_cntr[0] = INV_2 * (mblock.em(i, j, em::bx1) + mblock.em(i + 1, j, em::bx1));
        b_cntr[1] = INV_2 * (mblock.em(i, j, em::bx2) + mblock.em(i, j + 1, em::bx2));
        b_cntr[2] = mblock.em(i, j, em::bx3);

        // convert to hat
        vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO }, j_hat { ZERO };
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, e_cntr, e_hat);
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, j_cntr, j_hat);
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, b_cntr, b_hat);
        mblock.bckp(i, j, em::ex1) = e_hat[0];
        mblock.bckp(i, j, em::ex2) = e_hat[1];
        mblock.bckp(i, j, em::ex3) = e_hat[2];
        mblock.bckp(i, j, em::bx1) = b_hat[0];
        mblock.bckp(i, j, em::bx2) = b_hat[1];
        mblock.bckp(i, j, em::bx3) = b_hat[2];

        mblock.cur(i, j, cur::jx1) = j_hat[0];
        mblock.cur(i, j, cur::jx2) = j_hat[1];
        mblock.cur(i, j, cur::jx3) = j_hat[2];
      });

    ImposeContent(mblock.bckp_content,
                  { Content::ex1_hat_int,
                    Content::ex2_hat_int,
                    Content::ex3_hat_int,
                    Content::bx1_hat_int,
                    Content::bx2_hat_int,
                    Content::bx3_hat_int });
    ImposeContent(mblock.cur_content,
                  { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int });
    ImposeEmptyContent(mblock.buff_content);
  }

  template <>
  void PIC<Dim1>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }
  template <>
  void PIC<Dim3>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }

}    // namespace ntt
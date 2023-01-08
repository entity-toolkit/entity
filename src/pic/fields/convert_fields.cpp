#include "wrapper.h"

#include "fields.h"
#include "pic.h"

namespace ntt {
  // @TODO: ugly conversion without interpolation
  template <>
  void PIC<Dim1>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
    // auto& mblock = this->meshblock;
    // Kokkos::parallel_for(
    //   "convert_fields_to_hat", mblock.rangeAllCellsOnHost(), Lambda(index_t i) {
    //     real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    //     vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
    //     mblock.metric.v_Cntrv2Hat(
    //       { i_ + HALF },
    //       { mblock.em_h(i, em::ex1), mblock.em_h(i, em::ex2), mblock.em_h(i, em::ex3) },
    //       e_hat);
    //     mblock.metric.v_Cntrv2Hat(
    //       { i_ + HALF },
    //       { mblock.em_h(i, em::bx1), mblock.em_h(i, em::bx2), mblock.em_h(i, em::bx3) },
    //       b_hat);
    //     mblock.em_h(i, em::ex1) = e_hat[0];
    //     mblock.em_h(i, em::ex2) = e_hat[1];
    //     mblock.em_h(i, em::ex3) = e_hat[2];
    //     mblock.em_h(i, em::bx1) = b_hat[0];
    //     mblock.em_h(i, em::bx2) = b_hat[1];
    //     mblock.em_h(i, em::bx3) = b_hat[2];
    //   });
  }

  template <>
  void PIC<Dim2>::InterpolateAndConvertFieldsToHat() {
    auto& mblock = this->meshblock;
    Kokkos::deep_copy(mblock.bckp, mblock.em);
    Kokkos::parallel_for(
      "int_and_conv",
      mblock.rangeCells({
        {0, 1},
        {0, 1}
    }),
      Lambda(index_t i, index_t j) {
        real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
        // interpolate to cell centers
        mblock.bckp(i, j, em::ex1)
          = INV_2 * (mblock.em(i, j, em::ex1) + mblock.em(i, j + 1, em::ex1));
        mblock.bckp(i, j, em::ex2)
          = INV_2 * (mblock.em(i, j, em::ex2) + mblock.em(i + 1, j, em::ex2));
        mblock.bckp(i, j, em::ex3)
          = INV_4
            * (mblock.em(i, j, em::ex3) + mblock.em(i + 1, j, em::ex3)
               + mblock.em(i, j + 1, em::ex3) + mblock.em(i + 1, j + 1, em::ex3));
        mblock.bckp(i, j, em::bx1)
          = INV_2 * (mblock.em(i, j, em::bx1) + mblock.em(i + 1, j, em::bx1));
        mblock.bckp(i, j, em::bx2)
          = INV_2 * (mblock.em(i, j, em::bx2) + mblock.em(i, j + 1, em::bx2));
        mblock.bckp(i, j, em::bx3) = mblock.em(i, j, em::bx3);

        // convert to hat
        vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF },
                                  { mblock.bckp(i, j, em::ex1),
                                    mblock.bckp(i, j, em::ex2),
                                    mblock.bckp(i, j, em::ex3) },
                                  e_hat);
        mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF },
                                  { mblock.bckp(i, j, em::bx1),
                                    mblock.bckp(i, j, em::bx2),
                                    mblock.bckp(i, j, em::bx3) },
                                  b_hat);
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
    // auto& mblock = this->meshblock;
    // Kokkos::parallel_for(
    //   "convert_fields_to_hat",
    //   mblock.rangeAllCellsOnHost(),
    //   Lambda(index_t i, index_t j, index_t k) {
    //     real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
    //     real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
    //     real_t      k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };
    //     vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
    //     mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF, k_ + HALF },
    //                               { mblock.em_h(i, j, k, em::ex1),
    //                                 mblock.em_h(i, j, k, em::ex2),
    //                                 mblock.em_h(i, j, k, em::ex3) },
    //                               e_hat);
    //     mblock.metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF, k_ + HALF },
    //                               { mblock.em_h(i, j, k, em::bx1),
    //                                 mblock.em_h(i, j, k, em::bx2),
    //                                 mblock.em_h(i, j, k, em::bx3) },
    //                               b_hat);
    //     mblock.em_h(i, j, k, em::ex1) = e_hat[0];
    //     mblock.em_h(i, j, k, em::ex2) = e_hat[1];
    //     mblock.em_h(i, j, k, em::ex3) = e_hat[2];
    //     mblock.em_h(i, j, k, em::bx1) = b_hat[0];
    //     mblock.em_h(i, j, k, em::bx2) = b_hat[1];
    //     mblock.em_h(i, j, k, em::bx3) = b_hat[2];
    //   });
  }

}    // namespace ntt
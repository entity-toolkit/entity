#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"

namespace ntt {
  template <>
  void Meshblock<Dim2, PICEngine>::InterpolateAndConvertFieldsToHat() {
    NTTLog();
    auto                 ni1          = Ni1();
    auto                 ni2          = Ni2();

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
          Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
    for (auto i { 0 }; i < 6; ++i) {
      array_filled &= (bckp_content[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(bckp_content);
    Kokkos::deep_copy(bckp, em);
    ImposeContent(bckp_content, em_content);

    auto                 this_em     = this->em;
    auto                 this_bckp   = this->bckp;
    auto                 this_metric = this->metric;

    std::vector<Content> EB_cntrv
      = { Content::ex1_cntrv, Content::ex2_cntrv, Content::ex3_cntrv,
          Content::bx1_cntrv, Content::bx2_cntrv, Content::bx3_cntrv };
    AssertContent(em_content, EB_cntrv);

    Kokkos::parallel_for(
      "InterpolateAndConvertFieldsToHat", rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO };
        // from edges
        e_cntr[0] = INV_2 * (this_em(i, j, em::ex1) + this_em(i, j + 1, em::ex1));
        e_cntr[1] = INV_2 * (this_em(i, j, em::ex2) + this_em(i + 1, j, em::ex2));
        e_cntr[2] = INV_4
                    * (this_em(i, j, em::ex3) + this_em(i + 1, j, em::ex3)
                       + this_em(i, j + 1, em::ex3) + this_em(i + 1, j + 1, em::ex3));
        // from faces
        b_cntr[0] = INV_2 * (this_em(i, j, em::bx1) + this_em(i + 1, j, em::bx1));
        b_cntr[1] = INV_2 * (this_em(i, j, em::bx2) + this_em(i, j + 1, em::bx2));
        b_cntr[2] = this_em(i, j, em::bx3);

        // convert to hat
        vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
        this_metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, e_cntr, e_hat);
        this_metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, b_cntr, b_hat);
        this_bckp(i, j, em::ex1) = e_hat[0];
        this_bckp(i, j, em::ex2) = e_hat[1];
        this_bckp(i, j, em::ex3) = e_hat[2];
        this_bckp(i, j, em::bx1) = b_hat[0];
        this_bckp(i, j, em::bx2) = b_hat[1];
        this_bckp(i, j, em::bx3) = b_hat[2];
      });

    ImposeContent(bckp_content, int_fld_content);
  }    // namespace ntt

  template <>
  void Meshblock<Dim2, PICEngine>::InterpolateAndConvertCurrentsToHat() {
    NTTLog();
    auto                 ni1          = Ni1();
    auto                 ni2          = Ni2();

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int };
    for (auto i { 0 }; i < 3; ++i) {
      array_filled &= (cur_content[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(buff_content);
    Kokkos::deep_copy(buff, cur);
    ImposeContent(buff_content, cur_content);
    std::cout << "buff_content = " << (int)buff_content[0] << " " << (int)buff_content[1]
              << " " << (int)buff_content[2] << std::endl;
    std::cout << "cur_content = " << (int)cur_content[0] << " " << (int)cur_content[1] << " "
              << (int)cur_content[2] << std::endl;
    ImposeEmptyContent(cur_content);

    auto                 this_buff   = this->buff;
    auto                 this_cur    = this->cur;
    auto                 this_metric = this->metric;

    std::vector<Content> J_cntrv
      = { Content::jx1_cntrv, Content::jx2_cntrv, Content::jx3_cntrv };
    AssertContent(buff_content, J_cntrv);

    Kokkos::parallel_for(
      "InterpolateAndConvertCurrentsToHat", rangeActiveCells(), Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> j_cntr { ZERO };
        // from edges
        j_cntr[0] = INV_2 * (this_buff(i, j, cur::jx1) + this_buff(i, j + 1, cur::jx1));
        j_cntr[1] = INV_2 * (this_buff(i, j, cur::jx2) + this_buff(i + 1, j, cur::jx2));
        j_cntr[2] = INV_4
                    * (this_buff(i, j, cur::jx3) + this_buff(i + 1, j, cur::jx3)
                       + this_buff(i, j + 1, cur::jx3) + this_buff(i + 1, j + 1, cur::jx3));

        // convert to hat
        vec_t<Dim3> j_hat { ZERO };
        this_metric.v_Cntrv2Hat({ i_ + HALF, j_ + HALF }, j_cntr, j_hat);

        this_cur(i, j, cur::jx1) = j_hat[0];
        this_cur(i, j, cur::jx2) = j_hat[1];
        this_cur(i, j, cur::jx3) = j_hat[2];
      });

    ImposeContent(cur_content,
                  { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int });
    ImposeEmptyContent(buff_content);
  }

  template <>
  void Meshblock<Dim1, PICEngine>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }
  template <>
  void Meshblock<Dim3, PICEngine>::InterpolateAndConvertFieldsToHat() {
    NTTHostError("Not implemented.");
  }

  template <>
  void Meshblock<Dim1, PICEngine>::InterpolateAndConvertCurrentsToHat() {
    NTTHostError("Not implemented.");
  }
  template <>
  void Meshblock<Dim3, PICEngine>::InterpolateAndConvertCurrentsToHat() {
    NTTHostError("Not implemented.");
  }

}    // namespace ntt
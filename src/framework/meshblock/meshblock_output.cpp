#include "wrapper.h"

#include "meshblock.h"
#include "particle_macros.h"
#include "particles.h"
#include "sim_params.h"
#include "species.h"

namespace ntt {

  template <>
  void Meshblock<Dim1, PICEngine>::PrepareFieldsForOutput(const PrepareOutputFlags& flags) {
    NTTLog();
    auto                 this_em      = this->em;
    auto                 this_bckp    = this->bckp;
    auto                 this_metric  = this->metric;

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
          Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
    for (auto i { 0 }; i < 6; ++i) {
      array_filled &= ((this->bckp_content)[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(this->bckp_content);
    Kokkos::deep_copy(this_bckp, this_em);
    ImposeContent(this->bckp_content, this->em_content);

    std::vector<Content> EB_cntrv
      = { Content::ex1_cntrv, Content::ex2_cntrv, Content::ex3_cntrv,
          Content::bx1_cntrv, Content::bx2_cntrv, Content::bx3_cntrv };
    AssertContent(this->em_content, EB_cntrv);

    Kokkos::parallel_for(
      "InterpolateAndConvertFieldsToHat", this->rangeActiveCells(), Lambda(index_t i) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO };
        if (flags & PrepareOutput_InterpToCellCenter) {
          // from edges
          e_cntr[0] = this_em(i, em::ex1);
          e_cntr[1] = INV_2 * (this_em(i, em::ex2) + this_em(i + 1, em::ex2));
          e_cntr[2] = INV_2 * (this_em(i, em::ex3) + this_em(i + 1, em::ex3));
          // from faces
          b_cntr[0] = INV_2 * (this_em(i, em::bx1) + this_em(i + 1, em::bx1));
          b_cntr[1] = this_em(i, em::bx2);
          b_cntr[2] = this_em(i, em::bx3);
        } else {
          e_cntr[0] = this_em(i, em::ex1);
          e_cntr[1] = this_em(i, em::ex2);
          e_cntr[2] = this_em(i, em::ex3);
          b_cntr[0] = this_em(i, em::bx1);
          b_cntr[1] = this_em(i, em::bx2);
          b_cntr[2] = this_em(i, em::bx3);
        }
        // convert to hat
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        if (flags & PrepareOutput_ConvertToHat) {
          // !TODO: not quite correct when not in cell center
          this_metric.v3_Cntrv2Hat({ i_ + HALF }, e_cntr, e_out);
          this_metric.v3_Cntrv2Hat({ i_ + HALF }, b_cntr, b_out);
        } else {
          e_out[0] = e_cntr[0];
          e_out[1] = e_cntr[1];
          e_out[2] = e_cntr[2];
          b_out[0] = b_cntr[0];
          b_out[1] = b_cntr[1];
          b_out[2] = b_cntr[2];
        }
        this_bckp(i, em::ex1) = e_out[0];
        this_bckp(i, em::ex2) = e_out[1];
        this_bckp(i, em::ex3) = e_out[2];
        this_bckp(i, em::bx1) = b_out[0];
        this_bckp(i, em::bx2) = b_out[1];
        this_bckp(i, em::bx3) = b_out[2];
      });
    ImposeContent(this->bckp_content, int_fld_content);
  }

  template <>
  void Meshblock<Dim2, PICEngine>::PrepareFieldsForOutput(const PrepareOutputFlags& flags) {
    NTTLog();
    auto                 this_em      = this->em;
    auto                 this_bckp    = this->bckp;
    auto                 this_metric  = this->metric;

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
          Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
    for (auto i { 0 }; i < 6; ++i) {
      array_filled &= ((this->bckp_content)[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(this->bckp_content);
    Kokkos::deep_copy(this_bckp, this_em);
    ImposeContent(this->bckp_content, this->em_content);

    std::vector<Content> EB_cntrv
      = { Content::ex1_cntrv, Content::ex2_cntrv, Content::ex3_cntrv,
          Content::bx1_cntrv, Content::bx2_cntrv, Content::bx3_cntrv };
    AssertContent(this->em_content, EB_cntrv);

    Kokkos::parallel_for(
      "InterpolateAndConvertFieldsToHat",
      this->rangeActiveCells(),
      Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO };
        if (flags & PrepareOutput_InterpToCellCenter) {
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
        } else {
          e_cntr[0] = this_em(i, j, em::ex1);
          e_cntr[1] = this_em(i, j, em::ex2);
          e_cntr[2] = this_em(i, j, em::ex3);
          b_cntr[0] = this_em(i, j, em::bx1);
          b_cntr[1] = this_em(i, j, em::bx2);
          b_cntr[2] = this_em(i, j, em::bx3);
        }

        // convert to hat
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        if (flags & PrepareOutput_ConvertToHat) {
          // !TODO: not quite correct when not in cell center
          this_metric.v3_Cntrv2Hat({ i_ + HALF, j_ + HALF }, e_cntr, e_out);
          this_metric.v3_Cntrv2Hat({ i_ + HALF, j_ + HALF }, b_cntr, b_out);
        } else {
          e_out[0] = e_cntr[0];
          e_out[1] = e_cntr[1];
          e_out[2] = e_cntr[2];
          b_out[0] = b_cntr[0];
          b_out[1] = b_cntr[1];
          b_out[2] = b_cntr[2];
        }
        this_bckp(i, j, em::ex1) = e_out[0];
        this_bckp(i, j, em::ex2) = e_out[1];
        this_bckp(i, j, em::ex3) = e_out[2];
        this_bckp(i, j, em::bx1) = b_out[0];
        this_bckp(i, j, em::bx2) = b_out[1];
        this_bckp(i, j, em::bx3) = b_out[2];
      });
    ImposeContent(this->bckp_content, int_fld_content);
  }

  template <>
  void Meshblock<Dim2, GRPICEngine>::PrepareFieldsForOutput(const PrepareOutputFlags& flags) {
    NTTLog();
    auto                 this_em      = this->em;
    auto                 this_bckp    = this->bckp;
    auto                 this_metric  = this->metric;

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
          Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
    for (auto i { 0 }; i < 6; ++i) {
      array_filled &= ((this->bckp_content)[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(this->bckp_content);
    Kokkos::deep_copy(this_bckp, this_em);
    ImposeContent(this->bckp_content, this->em_content);

    std::vector<Content> EB_cntrv
      = { Content::ex1_cntrv, Content::ex2_cntrv, Content::ex3_cntrv,
          Content::bx1_cntrv, Content::bx2_cntrv, Content::bx3_cntrv };
    AssertContent(this->em_content, EB_cntrv);

    // in GRPIC engine these are really D and B (not E and H)

    Kokkos::parallel_for(
      "InterpolateAndConvertFieldsToSph",
      this->rangeActiveCells(),
      Lambda(index_t i, index_t j) {
        real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

        // interpolate to cell centers
        vec_t<Dim3> e_cntr { ZERO }, b_cntr { ZERO };
        if (flags & PrepareOutput_InterpToCellCenter) {
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
        } else {
          e_cntr[0] = this_em(i, j, em::ex1);
          e_cntr[1] = this_em(i, j, em::ex2);
          e_cntr[2] = this_em(i, j, em::ex3);
          b_cntr[0] = this_em(i, j, em::bx1);
          b_cntr[1] = this_em(i, j, em::bx2);
          b_cntr[2] = this_em(i, j, em::bx3);
        }

        // convert to hat
        vec_t<Dim3> e_out { ZERO }, b_out { ZERO };
        if (flags & PrepareOutput_ConvertToSphCntrv) {
          // !TODO: not quite correct when not in cell center
          this_metric.v3_Cntrv2SphCntrv({ i_ + HALF, j_ + HALF }, e_cntr, e_out);
          this_metric.v3_Cntrv2SphCntrv({ i_ + HALF, j_ + HALF }, b_cntr, b_out);
        } else {
          e_out[0] = e_cntr[0];
          e_out[1] = e_cntr[1];
          e_out[2] = e_cntr[2];
          b_out[0] = b_cntr[0];
          b_out[1] = b_cntr[1];
          b_out[2] = b_cntr[2];
        }
        this_bckp(i, j, em::ex1) = e_out[0];
        this_bckp(i, j, em::ex2) = e_out[1];
        this_bckp(i, j, em::ex3) = e_out[2];
        this_bckp(i, j, em::bx1) = b_out[0];
        this_bckp(i, j, em::bx2) = b_out[1];
        this_bckp(i, j, em::bx3) = b_out[2];
      });
    ImposeContent(this->bckp_content, int_fld_content);
  }

  // Currents interpolation
  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::PrepareCurrentsForOutput(const PrepareOutputFlags& flags) {
    NTTLog();
    auto                 this_buff    = this->buff;
    auto                 this_cur     = this->cur;
    auto                 this_metric  = this->metric;

    auto                 array_filled = true;
    std::vector<Content> int_fld_content
      = { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int };
    for (auto i { 0 }; i < 3; ++i) {
      array_filled &= ((this->cur_content)[i] == int_fld_content[i]);
    }
    if (array_filled) {
      // do nothing since the array is already filled with the right quantities
      return;
    }
    AssertEmptyContent(this->buff_content);
    Kokkos::deep_copy(this_buff, this_cur);
    ImposeContent(this->buff_content, this->cur_content);
    ImposeEmptyContent(this->cur_content);

    std::vector<Content> J_cntrv
      = { Content::jx1_cntrv, Content::jx2_cntrv, Content::jx3_cntrv };
    AssertContent(this->buff_content, J_cntrv);

    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "InterpolateAndConvertCurrentsToHat", this->rangeActiveCells(), Lambda(index_t i) {
          real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };

          // interpolate to cell centers
          vec_t<Dim3> j_cntr { ZERO };
          if (flags & PrepareOutput_InterpToCellCenter) {
            // from edges
            j_cntr[0] = this_buff(i, cur::jx1);
            j_cntr[1] = INV_2 * (this_buff(i, cur::jx2) + this_buff(i + 1, cur::jx2));
            j_cntr[2] = INV_2 * (this_buff(i, cur::jx3) + this_buff(i + 1, cur::jx3));
          } else {
            j_cntr[0] = this_buff(i, cur::jx1);
            j_cntr[1] = this_buff(i, cur::jx2);
            j_cntr[2] = this_buff(i, cur::jx3);
          }

          // convert to hat
          vec_t<Dim3> j_hat { ZERO };
          if (flags & PrepareOutput_ConvertToHat) {
            this_metric.v3_Cntrv2Hat({ i_ + HALF }, j_cntr, j_hat);
          } else {
            j_hat[0] = j_cntr[0];
            j_hat[1] = j_cntr[1];
            j_hat[2] = j_cntr[2];
          }

          this_cur(i, cur::jx1) = j_hat[0];
          this_cur(i, cur::jx2) = j_hat[1];
          this_cur(i, cur::jx3) = j_hat[2];
        });
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "InterpolateAndConvertCurrentsToHat",
        this->rangeActiveCells(),
        Lambda(index_t i, index_t j) {
          real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

          // interpolate to cell centers
          vec_t<Dim3> j_cntr { ZERO };
          if (flags & PrepareOutput_InterpToCellCenter) {
            // from edges
            j_cntr[0] = INV_2 * (this_buff(i, j, cur::jx1) + this_buff(i, j + 1, cur::jx1));
            j_cntr[1] = INV_2 * (this_buff(i, j, cur::jx2) + this_buff(i + 1, j, cur::jx2));
            j_cntr[2]
              = INV_4
                * (this_buff(i, j, cur::jx3) + this_buff(i + 1, j, cur::jx3)
                   + this_buff(i, j + 1, cur::jx3) + this_buff(i + 1, j + 1, cur::jx3));
          } else {
            j_cntr[0] = this_buff(i, j, cur::jx1);
            j_cntr[1] = this_buff(i, j, cur::jx2);
            j_cntr[2] = this_buff(i, j, cur::jx3);
          }

          // convert to hat
          vec_t<Dim3> j_hat { ZERO };
          if (flags & PrepareOutput_ConvertToHat) {
            this_metric.v3_Cntrv2Hat({ i_ + HALF, j_ + HALF }, j_cntr, j_hat);
          } else {
            j_hat[0] = j_cntr[0];
            j_hat[1] = j_cntr[1];
            j_hat[2] = j_cntr[2];
          }

          this_cur(i, j, cur::jx1) = j_hat[0];
          this_cur(i, j, cur::jx2) = j_hat[1];
          this_cur(i, j, cur::jx3) = j_hat[2];
        });

    } else if constexpr (D == Dim3) {
      NTTHostError("Not implemented.");
    }

    ImposeContent(this->cur_content,
                  { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int });
    ImposeEmptyContent(this->buff_content);
  }
}    // namespace ntt
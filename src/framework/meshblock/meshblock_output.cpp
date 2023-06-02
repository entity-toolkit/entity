#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"
#include "species.h"

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  template <int N, int M>
  void Meshblock<D, S>::PrepareFieldsForOutput(const ndfield_t<D, N>&    field,
                                               ndfield_t<D, M>&          buffer,
                                               const int&                fx1,
                                               const int&                fx2,
                                               const int&                fx3,
                                               const PrepareOutputFlags& flags) {
    NTTLog();
    NTTHostErrorIf(fx1 >= N || fx2 >= N || fx3 >= N || fx1 >= M || fx2 >= M || fx3 >= M,
                   "Invalid field index");
    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "PrepareFieldsForOutput", this->rangeActiveCells(), ClassLambda(index_t i) {
          real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          vec_t<Dim3> f_int { ZERO }, f_sph { ZERO };
          auto        cell_center = false;
          if (flags & PrepareOutput_InterpToCellCenterFromEdges) {
            f_int[0]    = field(i, fx1);
            f_int[1]    = INV_2 * (field(i, fx2) + field(i + 1, fx2));
            f_int[2]    = INV_2 * (field(i, fx3) + field(i + 1, fx3));
            cell_center = true;
          } else if (flags & PrepareOutput_InterpToCellCenterFromFaces) {
            f_int[0]    = INV_2 * (field(i, fx1) + field(i + 1, fx1));
            f_int[1]    = field(i, fx2);
            f_int[2]    = field(i, fx3);
            cell_center = true;
          } else {
            f_int[0] = field(i, fx1);
            f_int[1] = field(i, fx2);
            f_int[2] = field(i, fx3);
          }

          coord_t<Dim1> xi_field { ZERO };
          if (cell_center) {
            xi_field[0] = i_ + HALF;
          } else {
            xi_field[0] = i_;
          }

          if (flags & PrepareOutput_ConvertToHat) {
            this->metric.v3_Cntrv2Hat(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCntrv) {
            this->metric.v3_Cntrv2PhysCntrv(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCov) {
            this->metric.v3_Cov2PhysCov(xi_field, f_int, f_sph);
          }
          buffer(i, fx1) = f_sph[0];
          buffer(i, fx2) = f_sph[1];
          buffer(i, fx3) = f_sph[2];
        });
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "PrepareFieldsForOutput", this->rangeActiveCells(), ClassLambda(index_t i, index_t j) {
          real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

          vec_t<Dim3> f_int { ZERO }, f_sph { ZERO };
          auto        cell_center = false;
          if (flags & PrepareOutput_InterpToCellCenterFromEdges) {
            f_int[0] = INV_2 * (field(i, j, fx1) + field(i, j + 1, fx1));
            f_int[1] = INV_2 * (field(i, j, fx2) + field(i + 1, j, fx2));
            f_int[2] = INV_4
                       * (field(i, j, fx3) + field(i + 1, j, fx3) + field(i, j + 1, fx3)
                          + field(i + 1, j + 1, fx3));
            cell_center = true;
          } else if (flags & PrepareOutput_InterpToCellCenterFromFaces) {
            f_int[0]    = INV_2 * (field(i, j, fx1) + field(i + 1, j, fx1));
            f_int[1]    = INV_2 * (field(i, j, fx2) + field(i, j + 1, fx2));
            f_int[2]    = field(i, j, fx3);
            cell_center = true;
          } else {
            f_int[0] = field(i, j, fx1);
            f_int[1] = field(i, j, fx2);
            f_int[2] = field(i, j, fx3);
          }

          coord_t<Dim2> xi_field { ZERO };
          if (cell_center) {
            xi_field[0] = i_ + HALF;
            xi_field[1] = j_ + HALF;
          } else {
            xi_field[0] = i_;
            xi_field[1] = j_;
          }

          if (flags & PrepareOutput_ConvertToHat) {
            this->metric.v3_Cntrv2Hat(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCntrv) {
            this->metric.v3_Cntrv2PhysCntrv(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCov) {
            this->metric.v3_Cov2PhysCov(xi_field, f_int, f_sph);
          }
          buffer(i, j, fx1) = f_sph[0];
          buffer(i, j, fx2) = f_sph[1];
          buffer(i, j, fx3) = f_sph[2];
        });
    } else if constexpr (D == Dim3) {
      Kokkos::parallel_for(
        "PrepareFieldsForOutput",
        this->rangeActiveCells(),
        ClassLambda(index_t i, index_t j, index_t k) {
          real_t      i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          real_t      j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
          real_t      k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };

          vec_t<Dim3> f_int { ZERO }, f_sph { ZERO };
          auto        cell_center = false;
          if (flags & PrepareOutput_InterpToCellCenterFromEdges) {
            f_int[0] = INV_4
                       * (field(i, j, k, fx1) + field(i, j + 1, k, fx1)
                          + field(i, j, k + 1, fx1) + field(i, j + 1, k + 1, fx1));
            f_int[1] = INV_4
                       * (field(i, j, k, fx2) + field(i + 1, j, k, fx2)
                          + field(i, j, k + 1, fx2) + field(i + 1, j, k + 1, fx2));
            f_int[2] = INV_4
                       * (field(i, j, k, fx3) + field(i + 1, j, k, fx3)
                          + field(i, j + 1, k, fx3) + field(i + 1, j + 1, k, fx3));
            cell_center = true;
          } else if (flags & PrepareOutput_InterpToCellCenterFromFaces) {
            f_int[0]    = INV_2 * (field(i, j, k, fx1) + field(i + 1, j, k, fx1));
            f_int[1]    = INV_2 * (field(i, j, k, fx2) + field(i, j + 1, k, fx2));
            f_int[2]    = INV_2 * (field(i, j, k, fx3) + field(i, j, k + 1, fx3));
            cell_center = true;
          } else {
            f_int[0] = field(i, j, k, fx1);
            f_int[1] = field(i, j, k, fx2);
            f_int[2] = field(i, j, k, fx3);
          }

          coord_t<Dim3> xi_field { ZERO };
          if (cell_center) {
            xi_field[0] = i_ + HALF;
            xi_field[1] = j_ + HALF;
            xi_field[2] = k_ + HALF;
          } else {
            xi_field[0] = i_;
            xi_field[1] = j_;
            xi_field[2] = k_;
          }

          if (flags & PrepareOutput_ConvertToHat) {
            this->metric.v3_Cntrv2Hat(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCntrv) {
            this->metric.v3_Cntrv2PhysCntrv(xi_field, f_int, f_sph);
          } else if (flags & PrepareOutput_ConvertToPhysCov) {
            this->metric.v3_Cov2PhysCov(xi_field, f_int, f_sph);
          }
          buffer(i, j, k, fx1) = f_sph[0];
          buffer(i, j, k, fx2) = f_sph[1];
          buffer(i, j, k, fx3) = f_sph[2];
        });
    }
  }

  template <>
  void Meshblock<Dim2, GRPICEngine>::ComputeVectorPotential(ndfield_t<Dim2, 6>& buffer,
                                                            const int&          buffer_comp) {
    const auto i2_min = this->i2_min();
    // !TODO: this is quite slow
    Kokkos::parallel_for(
      "ComputeVectorPotential", this->rangeActiveCells(), ClassLambda(index_t i, index_t j) {
        const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        const int    k_min = static_cast<int>(i2_min - N_GHOSTS) + 1;
        const int    k_max = static_cast<int>(j - N_GHOSTS);
        real_t       A3    = ZERO;
        for (auto k { k_min }; k <= k_max; ++k) {
          real_t k_ = static_cast<real_t>(k);
          real_t sqrt_detH_ij1 { this->metric.sqrt_det_h({ i_, k_ - HALF }) };
          real_t sqrt_detH_ij2 { this->metric.sqrt_det_h({ i_, k_ + HALF }) };
          int    k1 { k + N_GHOSTS };
          A3 += HALF
                * (sqrt_detH_ij1 * this->em(i, k1 - 1, em::bx1)
                   + sqrt_detH_ij2 * this->em(i, k1, em::bx1));
        }
        buffer(i, j, buffer_comp) = A3;
      });
  }

}    // namespace ntt
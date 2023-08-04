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
        const auto   k_min = (i2_min - N_GHOSTS) + 1;
        const auto   k_max = (j - N_GHOSTS);
        real_t       A3    = ZERO;
        for (auto k { k_min }; k <= k_max; ++k) {
          real_t k_ = static_cast<real_t>(k);
          real_t sqrt_detH_ij1 { this->metric.sqrt_det_h({ i_, k_ - HALF }) };
          real_t sqrt_detH_ij2 { this->metric.sqrt_det_h({ i_, k_ + HALF }) };
          auto   k1 { k + N_GHOSTS };
          A3 += HALF
                * (sqrt_detH_ij1 * this->em(i, k1 - 1, em::bx1)
                   + sqrt_detH_ij2 * this->em(i, k1, em::bx1));
        }
        buffer(i, j, buffer_comp) = A3;
      });
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeDivergenceED(ndfield_t<D, 3>& buffer, const int& buffer_comp) {
    // divE/D is defined in cell centers

    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "ComputeDivergenceED", this->rangeActiveCells(), ClassLambda(index_t i) {
          const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          const real_t ex1_i { INV_2 * (this->em(i, em::ex1) + this->em(i - 1, em::ex1)) };
          const real_t ex1_iP1 { INV_2 * (this->em(i + 1, em::ex1) + this->em(i, em::ex1)) };

          const real_t sqrt_detH_i { this->metric.sqrt_det_h({ i_ }) };
          const real_t sqrt_detH_iP1 { this->metric.sqrt_det_h({ i_ + ONE }) };
          const real_t one_ovr_sqrt_detH_iP { ONE / this->metric.sqrt_det_h({ i_ + HALF }) };
          buffer(i, buffer_comp)
            = one_ovr_sqrt_detH_iP * (sqrt_detH_iP1 * ex1_iP1 - sqrt_detH_i * ex1_i);
        });
    } else if constexpr (D == Dim2) {
      Kokkos::parallel_for(
        "ComputeDivergenceED", this->rangeActiveCells(), ClassLambda(index_t i, index_t j) {
          const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          const real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

          const real_t ex1_ijP { INV_4
                                 * (this->em(i, j, em::ex1) + this->em(i, j + 1, em::ex1)
                                    + this->em(i - 1, j, em::ex1)
                                    + this->em(i - 1, j + 1, em::ex1)) };
          const real_t ex1_iP1jP { INV_4
                                   * (this->em(i, j, em::ex1) + this->em(i, j + 1, em::ex1)
                                      + this->em(i + 1, j, em::ex1)
                                      + this->em(i + 1, j + 1, em::ex1)) };
          const real_t ex2_iPj { INV_4
                                 * (this->em(i, j, em::ex2) + this->em(i, j - 1, em::ex2)
                                    + this->em(i + 1, j, em::ex2)
                                    + this->em(i + 1, j - 1, em::ex2)) };
          const real_t ex2_iPjP1 { INV_4
                                   * (this->em(i, j, em::ex2) + this->em(i, j + 1, em::ex2)
                                      + this->em(i + 1, j, em::ex2)
                                      + this->em(i + 1, j + 1, em::ex2)) };
          const real_t sqrt_detH_iP1jP { this->metric.sqrt_det_h({ i_ + ONE, j_ + HALF }) };
          const real_t sqrt_detH_ijP { this->metric.sqrt_det_h({ i_, j_ + HALF }) };
          const real_t sqrt_detH_iPjP1 { this->metric.sqrt_det_h({ i_ + HALF, j_ + ONE }) };
          const real_t sqrt_detH_iPj { this->metric.sqrt_det_h({ i_ + HALF, j_ }) };
          const real_t one_ovr_sqrt_detH_iPjP {
            ONE / this->metric.sqrt_det_h({ i_ + HALF, j_ + HALF })
          };
          buffer(i, j, buffer_comp)
            = one_ovr_sqrt_detH_iPjP
              * ((sqrt_detH_iP1jP * ex1_iP1jP - sqrt_detH_ijP * ex1_ijP)
                 + (sqrt_detH_iPjP1 * ex2_iPjP1 - sqrt_detH_iPj * ex2_iPj));
        });
    } else {
      Kokkos::parallel_for(
        "ComputeDivergenceED",
        this->rangeActiveCells(),
        ClassLambda(index_t i, index_t j, index_t k) {
          const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          const real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
          const real_t k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };
          const real_t ex1_ijPkP {
            INV_8
            * (this->em(i, j, k, em::ex1) + this->em(i, j + 1, k, em::ex1)
               + this->em(i - 1, j, k, em::ex1) + this->em(i - 1, j + 1, k, em::ex1)
               + this->em(i, j, k + 1, em::ex1) + this->em(i, j + 1, k + 1, em::ex1)
               + this->em(i - 1, j, k + 1, em::ex1) + this->em(i - 1, j + 1, k + 1, em::ex1))
          };
          const real_t ex1_iP1jPkP {
            INV_8
            * (this->em(i, j, k, em::ex1) + this->em(i, j + 1, k, em::ex1)
               + this->em(i + 1, j, k, em::ex1) + this->em(i + 1, j + 1, k, em::ex1)
               + this->em(i, j, k + 1, em::ex1) + this->em(i, j + 1, k + 1, em::ex1)
               + this->em(i + 1, j, k + 1, em::ex1) + this->em(i + 1, j + 1, k + 1, em::ex1))
          };
          const real_t ex2_iPjkP {
            INV_8
            * (this->em(i, j, k, em::ex2) + this->em(i, j - 1, k, em::ex2)
               + this->em(i + 1, j, k, em::ex2) + this->em(i + 1, j - 1, k, em::ex2)
               + this->em(i, j, k + 1, em::ex2) + this->em(i, j - 1, k + 1, em::ex2)
               + this->em(i + 1, j, k + 1, em::ex2) + this->em(i + 1, j - 1, k + 1, em::ex2))
          };
          const real_t ex2_iPjP1kP {
            INV_8
            * (this->em(i, j, k, em::ex2) + this->em(i, j + 1, k, em::ex2)
               + this->em(i + 1, j, k, em::ex2) + this->em(i + 1, j + 1, k, em::ex2)
               + this->em(i, j, k + 1, em::ex2) + this->em(i, j + 1, k + 1, em::ex2)
               + this->em(i + 1, j, k + 1, em::ex2) + this->em(i + 1, j + 1, k + 1, em::ex2))
          };
          const real_t ex3_iPjPk {
            INV_8
            * (this->em(i, j, k, em::ex3) + this->em(i, j, k - 1, em::ex3)
               + this->em(i + 1, j, k, em::ex3) + this->em(i + 1, j, k - 1, em::ex3)
               + this->em(i, j + 1, k, em::ex3) + this->em(i, j + 1, k - 1, em::ex3)
               + this->em(i + 1, j + 1, k, em::ex3) + this->em(i + 1, j + 1, k - 1, em::ex3))
          };
          const real_t ex3_iPjPkP1 {
            INV_8
            * (this->em(i, j, k, em::ex3) + this->em(i, j, k + 1, em::ex3)
               + this->em(i + 1, j, k, em::ex3) + this->em(i + 1, j, k + 1, em::ex3)
               + this->em(i, j + 1, k, em::ex3) + this->em(i, j + 1, k + 1, em::ex3)
               + this->em(i + 1, j + 1, k, em::ex3) + this->em(i + 1, j + 1, k + 1, em::ex3))
          };

          const real_t sqrt_detH_ijPkP { this->metric.sqrt_det_h(
            { i_, j_ + HALF, k_ + HALF }) };
          const real_t sqrt_detH_iP1jPkP { this->metric.sqrt_det_h(
            { i_ + ONE, j_ + HALF, k_ + HALF }) };
          const real_t sqrt_detH_iPjkP { this->metric.sqrt_det_h(
            { i_ + HALF, j_, k_ + HALF }) };
          const real_t sqrt_detH_iPjP1kP { this->metric.sqrt_det_h(
            { i_ + HALF, j_ + ONE, k_ + HALF }) };
          const real_t sqrt_detH_iPjPk { this->metric.sqrt_det_h(
            { i_ + HALF, j_ + HALF, k_ }) };
          const real_t sqrt_detH_iPjPkP1 { this->metric.sqrt_det_h(
            { i_ + HALF, j_ + HALF, k_ + ONE }) };
          const real_t one_ovr_sqrt_detH_iPjPkP {
            ONE / this->metric.sqrt_det_h({ i_ + HALF, j_ + HALF, k_ + HALF })
          };
          buffer(i, j, k, buffer_comp)
            = one_ovr_sqrt_detH_iPjPkP
              * ((sqrt_detH_iP1jPkP * ex1_iP1jPkP - sqrt_detH_ijPkP * ex1_ijPkP)
                 + (sqrt_detH_iPjP1kP * ex2_iPjP1kP - sqrt_detH_iPjkP * ex2_iPjkP)
                 + (sqrt_detH_iPjPkP1 * ex3_iPjPk - sqrt_detH_iPjPk * ex3_iPjPkP1));
        });
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeMoments(const SimulationParams& params,
                                       const FieldID&          field,
                                       const std::vector<int>& components,
                                       const std::vector<int>& prtl_species,
                                       const int&              buff_ind,
                                       const short&            smooth) {
    NTTLog();
    real_t     weight = ONE / math::pow(2.0 * smooth + 1.0, static_cast<int>(D));
    if (field != FieldID::Nppc) {
      weight /= params.ppc0();
    }
    if constexpr (D == Dim1) {
      Kokkos::deep_copy(Kokkos::subview(this->buff, Kokkos::ALL(), (int)buff_ind), ZERO);
    } else if constexpr (D == Dim2) {
      Kokkos::deep_copy(
        Kokkos::subview(this->buff, Kokkos::ALL(), Kokkos::ALL(), (int)buff_ind), ZERO);
    } else if constexpr (D == Dim3) {
      Kokkos::deep_copy(
        Kokkos::subview(this->buff, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), (int)buff_ind),
        ZERO);
    }

    // if species not specified, use all massive particles
    std::vector<int> out_species = prtl_species;
    if (out_species.size() == 0) {
      for (auto& specs : particles) {
        if (specs.mass() > 0.0) {
          out_species.push_back(specs.index());
        }
      }
    }

    // extract the components so that the kernel could interpret them
    int comp1 { -1 }, comp2 { -1 };
    if (components.size() == 1) {
      NTTHostError("ComputeMoments: only one component for T passed");
    } else if (components.size() == 2) {
      comp1 = components[0];
      comp2 = components[1];
    }

    auto this_metric  = this->metric;
    auto use_weights  = params.useWeights();

    auto scatter_buff = Kokkos::Experimental::create_scatter_view(this->buff);
    for (auto& sp : out_species) {
      auto species = particles[sp - 1];
      auto mass    = species.mass();
      auto charge  = species.charge();
      if ((field == FieldID::Charge) && AlmostEqual(charge, 0.0f)) {
        continue;
      }
      if constexpr (D == Dim1) {
        const int ni1 = this->Ni1();
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.tag(p) == static_cast<short>(ParticleTag::alive)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              real_t x1          = get_prtl_x1(species, p);
              auto   i1_min      = i1 - smooth + N_GHOSTS;
              auto   i1_max      = i1 + smooth + N_GHOSTS;
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy { ((mass == ZERO) ? get_photon_Energy_SR(species, p)
                                                : get_prtl_Gamma_SR(species, p)) };
                contrib = ((mass == ZERO) ? ONE : mass) / energy;
                for (auto& c : { comp1, comp2 }) {
                  if (c == 0) {
                    contrib *= energy;
                  } else if (c == 1) {
                    contrib *= species.ux1(p);
                  } else if (c == 2) {
                    contrib *= species.ux2(p);
                  } else if (c == 3) {
                    contrib *= species.ux3(p);
                  }
                }
              }
              if (field != FieldID::Nppc) {
                contrib *= this_metric.min_cell_volume() / this_metric.sqrt_det_h({ x1 });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                buff_access(i1_, buff_ind) += contrib * weight;
              }
            }
          });
      } else if constexpr (D == Dim2) {
        const int ni1 = this->Ni1(), ni2 = this->Ni2();
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.tag(p) == static_cast<short>(ParticleTag::alive)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              auto   i2          = species.i2(p);
              real_t x1          = get_prtl_x1(species, p);
              real_t x2          = get_prtl_x2(species, p);
              auto   i1_min      = IMIN(IMAX(i1 - smooth, 0), ni1) + N_GHOSTS;
              auto   i1_max      = IMIN(IMAX(i1 + smooth, 0), ni1) + N_GHOSTS;
              auto   i2_min      = IMIN(IMAX(i2 - smooth, 0), ni2) + N_GHOSTS;
              auto   i2_max      = IMIN(IMAX(i2 + smooth, 0), ni2) + N_GHOSTS;
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy { ((mass == ZERO) ? get_photon_Energy_SR(species, p)
                                                : get_prtl_Gamma_SR(species, p)) };
                contrib = ((mass == ZERO) ? ONE : mass) / energy;
#ifdef MINKOWSKI_METRIC
                for (auto& c : { comp1, comp2 }) {
                  if (c == 0) {
                    contrib *= energy;
                  } else if (c == 1) {
                    contrib *= species.ux1(p);
                  } else if (c == 2) {
                    contrib *= species.ux2(p);
                  } else if (c == 3) {
                    contrib *= species.ux3(p);
                  }
                }
#else
                  real_t      phi = species.phi(p);
                  vec_t<Dim3> u_hat;
                  this_metric.v3_Cart2Hat({ x1, x2, phi },
                                         { species.ux1(p), species.ux2(p), species.ux3(p) },
                                         u_hat);
                  for (auto& c : { comp1, comp2 }) {
                    if (c == 0) {
                      contrib *= energy;
                    } else { 
                      contrib *= u_hat[c - 1];
                    }
                  }
#endif
              }
              if (field != FieldID::Nppc) {
                contrib *= this_metric.min_cell_volume() / this_metric.sqrt_det_h({ x1, x2 });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i2_ { i2_min }; i2_ <= i2_max; ++i2_) {
                for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                  buff_access(i1_, i2_, buff_ind) += contrib * weight;
                }
              }
            }
          });
      } else if constexpr (D == Dim3) {
        const int ni1 = this->Ni1(), ni2 = this->Ni2(), ni3 = this->Ni3();
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.tag(p) == static_cast<short>(ParticleTag::alive)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              auto   i2          = species.i2(p);
              auto   i3          = species.i3(p);
              real_t x1          = get_prtl_x1(species, p);
              real_t x2          = get_prtl_x2(species, p);
              real_t x3          = get_prtl_x3(species, p);
              auto   i1_min      = i1 - smooth + N_GHOSTS;
              auto   i1_max      = i1 + smooth + N_GHOSTS;
              auto   i2_min      = i2 - smooth + N_GHOSTS;
              auto   i2_max      = i2 + smooth + N_GHOSTS;
              auto   i3_min      = i3 - smooth + N_GHOSTS;
              auto   i3_max      = i3 + smooth + N_GHOSTS;
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy { ((mass == ZERO) ? get_photon_Energy_SR(species, p)
                                                : get_prtl_Gamma_SR(species, p)) };
                contrib = ((mass == ZERO) ? ONE : mass) / energy;
#ifdef MINKOWSKI_METRIC
                for (auto& c : { comp1, comp2 }) {
                  if (c == 0) {
                    contrib *= energy;
                  } else if (c == 1) {
                    contrib *= species.ux1(p);
                  } else if (c == 2) {
                    contrib *= species.ux2(p);
                  } else if (c == 3) {
                    contrib *= species.ux3(p);
                  }
                }
#else
                vec_t<Dim3> u_hat;
                this_metric.v3_Cart2Hat(
                  { x1, x2, x3 }, { species.ux1(p), species.ux2(p), species.ux3(p) }, u_hat);
                for (auto& c : { comp1, comp2 }) {
                  if (c == 0) {
                    contrib *= energy;
                  } else {
                    contrib *= u_hat[c - 1];
                  }
                }
#endif
              }
              if (field != FieldID::Nppc) {
                contrib
                  *= this_metric.min_cell_volume() / this_metric.sqrt_det_h({ x1, x2, x3 });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i3_ { i3_min }; i3_ <= i3_max; ++i3_) {
                for (auto i2_ { i2_min }; i2_ <= i2_max; ++i2_) {
                  for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                    buff_access(i1_, i2_, i3_, buff_ind) += contrib * weight;
                  }
                }
              }
            }
          });
      }
    }
    Kokkos::Experimental::contribute(this->buff, scatter_buff);
  }

}    // namespace ntt
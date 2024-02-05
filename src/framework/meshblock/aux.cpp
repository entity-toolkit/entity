#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"
#include "species.h"

#include "meshblock/meshblock.h"
#include "meshblock/particles.h"

namespace ntt {

  namespace {
    template <Dimension D, int M>
    void ResetBuffer(ndfield_t<D, M>& buffer, int buff_ind) {
      if constexpr (D == Dim1) {
        Kokkos::deep_copy(Kokkos::subview(buffer, Kokkos::ALL(), buff_ind), ZERO);
      } else if constexpr (D == Dim2) {
        Kokkos::deep_copy(
          Kokkos::subview(buffer, Kokkos::ALL(), Kokkos::ALL(), buff_ind),
          ZERO);
      } else if constexpr (D == Dim3) {
        Kokkos::deep_copy(
          Kokkos::subview(buffer, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), buff_ind),
          ZERO);
      }
    }
  } // namespace

  template <Dimension D, SimulationEngine S>
  template <int N, int M>
  void Meshblock<D, S>::PrepareFieldsForOutput(const ndfield_t<D, N>& field,
                                               ndfield_t<D, M>&       buffer,
                                               const int&             fx1,
                                               const int&             fx2,
                                               const int&             fx3,
                                               const PrepareOutputFlags& flags) {
    NTTLog();
    NTTHostErrorIf(fx1 >= N || fx2 >= N || fx3 >= N || fx1 >= M || fx2 >= M ||
                     fx3 >= M,
                   "Invalid field index");
    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "PrepareFieldsForOutput",
        this->rangeActiveCells(),
        ClassLambda(index_t i) {
          real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
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
        "PrepareFieldsForOutput",
        this->rangeActiveCells(),
        ClassLambda(index_t i, index_t j) {
          real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };

          vec_t<Dim3> f_int { ZERO }, f_sph { ZERO };
          auto        cell_center = false;
          if (flags & PrepareOutput_InterpToCellCenterFromEdges) {
            f_int[0]    = INV_2 * (field(i, j, fx1) + field(i, j + 1, fx1));
            f_int[1]    = INV_2 * (field(i, j, fx2) + field(i + 1, j, fx2));
            f_int[2]    = INV_4 * (field(i, j, fx3) + field(i + 1, j, fx3) +
                                field(i, j + 1, fx3) + field(i + 1, j + 1, fx3));
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
          real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
          real_t j_ { static_cast<real_t>(static_cast<int>(j) - N_GHOSTS) };
          real_t k_ { static_cast<real_t>(static_cast<int>(k) - N_GHOSTS) };

          vec_t<Dim3> f_int { ZERO }, f_sph { ZERO };
          auto        cell_center = false;
          if (flags & PrepareOutput_InterpToCellCenterFromEdges) {
            f_int[0] = INV_4 *
                       (field(i, j, k, fx1) + field(i, j + 1, k, fx1) +
                        field(i, j, k + 1, fx1) + field(i, j + 1, k + 1, fx1));
            f_int[1] = INV_4 *
                       (field(i, j, k, fx2) + field(i + 1, j, k, fx2) +
                        field(i, j, k + 1, fx2) + field(i + 1, j, k + 1, fx2));
            f_int[2] = INV_4 *
                       (field(i, j, k, fx3) + field(i + 1, j, k, fx3) +
                        field(i, j + 1, k, fx3) + field(i + 1, j + 1, k, fx3));
            cell_center = true;
          } else if (flags & PrepareOutput_InterpToCellCenterFromFaces) {
            f_int[0] = INV_2 * (field(i, j, k, fx1) + field(i + 1, j, k, fx1));
            f_int[1] = INV_2 * (field(i, j, k, fx2) + field(i, j + 1, k, fx2));
            f_int[2] = INV_2 * (field(i, j, k, fx3) + field(i, j, k + 1, fx3));
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
                                                            int buff_ind) {
    NTTLog();
    ResetBuffer<Dim2, 6>(buffer, buff_ind);
    const auto i2_min = this->i2_min();
    // !TODO: this is quite slow
    Kokkos::parallel_for(
      "ComputeVectorPotential",
      this->rangeActiveCells(),
      ClassLambda(index_t i, index_t j) {
        const real_t i_ { static_cast<real_t>(static_cast<int>(i) - N_GHOSTS) };
        const auto   k_min = (i2_min - N_GHOSTS) + 1;
        const auto   k_max = (j - N_GHOSTS);
        real_t       A3    = ZERO;
        for (auto k { k_min }; k <= k_max; ++k) {
          real_t k_ = static_cast<real_t>(k);
          real_t sqrt_detH_ij1 { this->metric.sqrt_det_h({ i_, k_ - HALF }) };
          real_t sqrt_detH_ij2 { this->metric.sqrt_det_h({ i_, k_ + HALF }) };
          auto   k1 { k + N_GHOSTS };
          A3 += HALF * (sqrt_detH_ij1 * this->em(i, k1 - 1, em::bx1) +
                        sqrt_detH_ij2 * this->em(i, k1, em::bx1));
        }
        buffer(i, j, buff_ind) = A3;
      });
  }

  // !TODO: capture i2max properly (at the moment, the loop does not include it)
  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeDivergenceED(ndfield_t<D, 3>& buffer, int buff_ind) {
    NTTLog();
    ResetBuffer<D, 3>(buffer, buff_ind);
    // divE/D is defined in the nodes (i, j, k)
    if constexpr (D == Dim1) {
      Kokkos::parallel_for(
        "ComputeDivergenceED",
        this->rangeActiveCells(),
        ClassLambda(index_t i1) {
          const auto i1_ { COORD(i1) };
          buffer(i1, buff_ind) = (this->em(i1, em::ex1) *
                                    this->metric.sqrt_det_h({ i1_ + HALF }) -
                                  this->em(i1 - 1, em::ex1) *
                                    this->metric.sqrt_det_h({ i1_ - HALF })) /
                                 this->metric.sqrt_det_h({ i1_ });
        });
    } else if constexpr (D == Dim2) {
      const auto i2_min = this->i2_min();
      const auto i2_max = this->i2_max();
#ifdef MINKOWSKI_METRIC
      const auto ax_i2min = false;
      const auto ax_i2max = false;
#else
      const auto ax_i2min = (this->boundaries[1][0] == BoundaryCondition::AXIS);
      const auto ax_i2max = (this->boundaries[1][1] == BoundaryCondition::AXIS);
#endif
      Kokkos::parallel_for(
        "ComputeDivergenceED",
        this->rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2) {
          const auto i1_ { COORD(i1) };
          const auto i2_ { COORD(i2) };

          if ((ax_i2min && i2 == i2_min) || (ax_i2max && i2 == i2_max)) {
            // northern axis
            buffer(
              i1,
              i2,
              buff_ind) = (this->em(i1, i2, em::ex1) *
                             this->metric.sqrt_det_h_tilde({ i1_ + HALF, i2_ }) -
                           this->em(i1 - 1, i2, em::ex1) *
                             this->metric.sqrt_det_h_tilde({ i1_ - HALF, i2_ }) +
                           TWO * this->em(i1, i2, em::ex2) *
                             this->metric.sqrt_det_h({ i1_, i2_ + HALF })) /
                          this->metric.sqrt_det_h_tilde({ i1_, i2_ });
          } else {
            buffer(i1,
                   i2,
                   buff_ind) = (this->em(i1, i2, em::ex1) *
                                  this->metric.sqrt_det_h({ i1_ + HALF, i2_ }) -
                                this->em(i1 - 1, i2, em::ex1) *
                                  this->metric.sqrt_det_h({ i1_ - HALF, i2_ }) +
                                this->em(i1, i2, em::ex2) *
                                  this->metric.sqrt_det_h({ i1_, i2_ + HALF }) -
                                this->em(i1, i2 - 1, em::ex2) *
                                  this->metric.sqrt_det_h({ i1_, i2_ - HALF })) /
                               this->metric.sqrt_det_h({ i1_, i2_ });
          }
        });
    } else if constexpr (D == Dim3) {
      const auto i2_min = this->i2_min();
      const auto i2_max = this->i2_max();
#ifdef MINKOWSKI_METRIC
      const auto ax_i2min = false;
      const auto ax_i2max = false;
#else
      const auto ax_i2min = (this->boundaries[1][0] == BoundaryCondition::AXIS);
      const auto ax_i2max = (this->boundaries[1][1] == BoundaryCondition::AXIS);
#endif
      Kokkos::parallel_for(
        "ComputeDivergenceED",
        this->rangeActiveCells(),
        ClassLambda(index_t i1, index_t i2, index_t i3) {
          const auto i1_ { COORD(i1) };
          const auto i2_ { COORD(i2) };
          const auto i3_ { COORD(i3) };

          if ((ax_i2min && i2 == i2_min) || (ax_i2max && i2 == i2_max)) {
            buffer(i1, i2, i3, buff_ind) = ZERO;
          } else {
            buffer(i1, i2, i3, buff_ind) =
              (this->em(i1, i2, i3, em::ex1) *
                 this->metric.sqrt_det_h({ i1_ + HALF, i2_, i3_ }) -
               this->em(i1 - 1, i2, i3, em::ex1) *
                 this->metric.sqrt_det_h({ i1_ - HALF, i2_, i3_ }) +
               this->em(i1, i2, i3, em::ex2) *
                 this->metric.sqrt_det_h({ i1_, i2_ + HALF, i3_ }) -
               this->em(i1, i2 - 1, i3, em::ex2) *
                 this->metric.sqrt_det_h({ i1_, i2_ - HALF, i3_ }) +
               this->em(i1, i2, i3, em::ex3) *
                 this->metric.sqrt_det_h({ i1_, i2_, i3_ + HALF }) -
               this->em(i1, i2, i3 - 1, em::ex3) *
                 this->metric.sqrt_det_h({ i1_, i2_, i3_ - HALF })) /
              this->metric.sqrt_det_h({ i1_, i2_, i3_ });
          }
        });
    }
  }

  // template <Dimension D, SimulationEngine S>
  // void Meshblock<D, S>::ComputeChargeDensity(const SimulationParams& params,
  //                                            ndfield_t<D, 3>&        buffer,
  //                                            const std::vector<int>&
  //                                            prtl_species, int buff_ind) {
  //   NTTLog();
  //   ResetBuffer<D, 3>(this->buff, buff_ind);
  //   // charge density is defined in the nodes

  //   // if species not specified, use all charged particles
  //   std::vector<int> out_species = prtl_species;
  //   if (out_species.size() == 0) {
  //     for (auto& specs : particles) {
  //       if (specs.charge() > 0.0) {
  //         out_species.push_back(specs.index());
  //       }
  //     }
  //   }
  //   auto       this_metric = this->metric;
  //   const auto use_weights = params.useWeights();

  //   auto scatter_buff = Kokkos::Experimental::create_scatter_view(this->buff);
  //   for (auto& sp : out_species) {
  //     auto       species = particles[sp - 1];
  //     const auto q_n0    = species.charge() / params.n0();
  //     if constexpr (D == Dim1) {
  //       Kokkos::parallel_for(
  //         "ComputeChargeDensity",
  //         species.rangeActiveParticles(),
  //         Lambda(index_t p) {
  //           if (species.tag(p) == ParticleTag::alive) {
  //             auto       buff_access = scatter_buff.access();
  //             const auto i1          = species.i1(p);
  //             const auto i1_         = COORD(i1);
  //             const auto dx1         = species.dx1(p);
  //             const auto coeff = (use_weights) ? (q_n0 * species.weight(p))
  //                                              : (q_n0);

  //             buff_access(i1, buff_ind) += coeff * species.weight(p) *
  //                                          (ONE - dx1) /
  //                                          this_metric.sqrt_det_h({ i1_ });
  //             buff_access(i1 + 1,
  //                         buff_ind) += coeff * species.weight(p) * dx1 /
  //                                      this_metric.sqrt_det_h({ i1_ + ONE });
  //           }
  //         });
  //     } else if constexpr (D == Dim2) {
  //       const auto ax_i2min { (this->boundaries.size() > 1) &&
  //                             (this->boundaries[1][0] == BoundaryCondition::AXIS) };
  //       const auto ax_i2max { (this->boundaries.size() > 1) &&
  //                             (this->boundaries[1][1] == BoundaryCondition::AXIS) };
  //       const auto ni2 { (int)(this->Ni2()) };
  //       Kokkos::parallel_for(
  //         "ComputeChargeDensity",
  //         species.rangeActiveParticles(),
  //         Lambda(index_t p) {
  //           if (species.tag(p) == ParticleTag::alive) {
  //             auto       buff_access = scatter_buff.access();
  //             const auto i1          = species.i1(p);
  //             const auto i2          = species.i2(p);
  //             const auto i1_         = COORD(i1);
  //             const auto i2_         = COORD(i2);
  //           }
  //         });
  //     } else if constexpr (D == Dim3) {
  //       //         Kokkos::parallel_for(
  //       //           "ComputeMoments",
  //       //           species.rangeActiveParticles(),
  //       //           Lambda(index_t p) {
  //       //             if (species.tag(p) == ParticleTag::alive) {
  //       //               auto      buff_access = scatter_buff.access();
  //       //               auto      i1          = species.i1(p);
  //       //               auto      i2          = species.i2(p);
  //       //               auto      i3          = species.i3(p);
  //       //               const int i1_min      = i1 - window + N_GHOSTS;
  //       //               const int i1_max      = i1 + window + N_GHOSTS;
  //       //               const int i2_min      = i2 - window + N_GHOSTS;
  //       //               const int i2_max      = i2 + window + N_GHOSTS;
  //       //               const int i3_min      = i3 - window + N_GHOSTS;
  //       //               const int i3_max      = i3 + window + N_GHOSTS;
  //       //               real_t    contrib { ZERO };
  //       //               if (field == FieldID::Rho) {
  //       //                 contrib = ((mass == ZERO) ? ONE : mass);
  //       //               } else if (field == FieldID::Charge) {
  //       //                 contrib = charge;
  //       //               } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
  //       //                 contrib = ONE;
  //       //               } else if (field == FieldID::T) {
  //       //                 real_t energy {
  //       //                   (mass == ZERO)
  //       //                     ? NORM(species.ux1(p), species.ux2(p), species.ux3(p))
  //       //                     : math::sqrt(ONE + NORM_SQR(species.ux1(p),
  //       //                                                 species.ux2(p),
  //       //                                                 species.ux3(p)))
  //       //                 };
  //       //                 contrib = ((mass == ZERO) ? ONE : mass) / energy;
  //       // #ifdef MINKOWSKI_METRIC
  //       //                 for (auto& c : { comp1, comp2 }) {
  //       //                   if (c == 0) {
  //       //                     contrib *= energy;
  //       //                   } else if (c == 1) {
  //       //                     contrib *= species.ux1(p);
  //       //                   } else if (c == 2) {
  //       //                     contrib *= species.ux2(p);
  //       //                   } else if (c == 3) {
  //       //                     contrib *= species.ux3(p);
  //       //                   }
  //       //                 }
  //       // #else
  //       //                 const real_t x1 = get_prtl_x1(species, p);
  //       //                 const real_t x2 = get_prtl_x2(species, p);
  //       //                 const real_t x3 = get_prtl_x3(species, p);
  //       //                 vec_t<Dim3>  u_hat;
  //       //                 this_metric.v3_Cart2Hat(
  //       //                   { x1, x2, x3 },
  //       //                   { species.ux1(p), species.ux2(p), species.ux3(p) },
  //       //                   u_hat);
  //       //                 for (auto& c : { comp1, comp2 }) {
  //       //                   if (c == 0) {
  //       //                     contrib *= energy;
  //       //                   } else {
  //       //                     contrib *= u_hat[c - 1];
  //       //                   }
  //       //                 }
  //       // #endif
  //       //               }
  //       //               if (field != FieldID::Nppc) {
  //       //                 contrib *= inv_n0 / this_metric.sqrt_det_h(
  //       //                                       { static_cast<real_t>(i1) + HALF,
  //       //                                         static_cast<real_t>(i2) + HALF,
  //       //                                         static_cast<real_t>(i3) + HALF });
  //       //                 if (use_weights) {
  //       //                   contrib *= species.weight(p);
  //       //                 }
  //       //               }
  //       //               for (auto i3_ { i3_min }; i3_ <= i3_max; ++i3_) {
  //       //                 for (auto i2_ { i2_min }; i2_ <= i2_max; ++i2_) {
  //       //                   for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
  //       //                     buff_access(i1_, i2_, i3_, buff_ind) += contrib * smooth;
  //       //                   }
  //       //                 }
  //       //               }
  //       //             }
  //       //           });
  //     }
  //   }
  //   Kokkos::Experimental::contribute(this->buff, scatter_buff);
  // }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeMoments(const SimulationParams& params,
                                       const FieldID&          field,
                                       const std::vector<int>& components,
                                       const std::vector<int>& prtl_species,
                                       int                     buff_ind,
                                       short                   window) {
    NTTLog();
    ResetBuffer<D, 3>(this->buff, buff_ind);
    const auto smooth = ONE / math::pow(TWO * window + ONE, static_cast<int>(D));
    const auto inv_n0 = ONE / params.n0();

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

    auto       this_metric = this->metric;
    const auto use_weights = params.useWeights();

    auto scatter_buff = Kokkos::Experimental::create_scatter_view(this->buff);
    for (auto& sp : out_species) {
      NTTHostErrorIf(sp < 0 || sp >= particles.size(), "Invalid species index");
      auto species = particles[sp - 1];
      auto mass    = species.mass();
      auto charge  = species.charge();
      if ((field == FieldID::Charge) && AlmostEqual(charge, 0.0f)) {
        continue;
      }
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "ComputeMoments",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (species.tag(p) == ParticleTag::alive) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              auto   i1_min      = i1 - window + N_GHOSTS;
              auto   i1_max      = i1 + window + N_GHOSTS;
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy {
                  (mass == ZERO)
                    ? NORM(species.ux1(p), species.ux2(p), species.ux3(p))
                    : math::sqrt(ONE + NORM_SQR(species.ux1(p),
                                                species.ux2(p),
                                                species.ux3(p)))
                };
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
                contrib *= inv_n0 / this_metric.sqrt_det_h(
                                      { static_cast<real_t>(i1) + HALF });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                buff_access(i1_, buff_ind) += contrib * smooth;
              }
            }
          });
      } else if constexpr (D == Dim2) {
        const auto ax_i2min { (this->boundaries.size() > 1) &&
                              (this->boundaries[1][0] == BoundaryCondition::AXIS) };
        const auto ax_i2max { (this->boundaries.size() > 1) &&
                              (this->boundaries[1][1] == BoundaryCondition::AXIS) };
        const auto ni2 { (int)(this->Ni2()) };
        Kokkos::parallel_for(
          "ComputeMoments",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (species.tag(p) == ParticleTag::alive) {
              auto      buff_access = scatter_buff.access();
              auto      i1          = species.i1(p);
              auto      i2          = species.i2(p);
              const int i1_min      = i1 - window + N_GHOSTS;
              const int i1_max      = i1 + window + N_GHOSTS;
              const int i2_min      = i2 - window + N_GHOSTS;
              const int i2_max      = i2 + window + N_GHOSTS;
              real_t    contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy {
                  (mass == ZERO)
                    ? NORM(species.ux1(p), species.ux2(p), species.ux3(p))
                    : math::sqrt(ONE + NORM_SQR(species.ux1(p),
                                                species.ux2(p),
                                                species.ux3(p)))
                };
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
                const real_t x1 = get_prtl_x1(species, p);
                const real_t x2 = get_prtl_x2(species, p);
                vec_t<Dim3>  u_hat;
  #ifdef PIC_ENGINE
                const real_t phi = species.phi(p);
                this_metric.v3_Cart2Hat(
                  { x1, x2, phi },
                  { species.ux1(p), species.ux2(p), species.ux3(p) },
                  u_hat);
  #else
                this_metric.v3_Cov2Hat(
                  { x1, x2 },
                  { species.ux1(p), species.ux2(p), species.ux3(p) },
                  u_hat);
  #endif
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
                contrib *= inv_n0 / this_metric.sqrt_det_h(
                                      { static_cast<real_t>(i1) + HALF,
                                        static_cast<real_t>(i2) + HALF });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i2_ { i2_min }; i2_ <= i2_max; ++i2_) {
                for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                  if (ax_i2min && (i2_ - static_cast<int>(N_GHOSTS) < 0)) {
                    // reflect from theta = 0
                    buff_access(i1_,
                                -i2_ + 2 * static_cast<int>(N_GHOSTS),
                                buff_ind) += contrib * smooth;
                  } else if (ax_i2max && (i2_ - static_cast<int>(N_GHOSTS) >= ni2)) {
                    // reflect from theta = pi
                    buff_access(i1_,
                                2 * ni2 - i2_ + 2 * static_cast<int>(N_GHOSTS) - 1,
                                buff_ind) += contrib * smooth;
                  } else {
                    buff_access(i1_, i2_, buff_ind) += contrib * smooth;
                  }
                }
              }
            }
          });
      } else if constexpr (D == Dim3) {
        Kokkos::parallel_for(
          "ComputeMoments",
          species.rangeActiveParticles(),
          Lambda(index_t p) {
            if (species.tag(p) == ParticleTag::alive) {
              auto      buff_access = scatter_buff.access();
              auto      i1          = species.i1(p);
              auto      i2          = species.i2(p);
              auto      i3          = species.i3(p);
              const int i1_min      = i1 - window + N_GHOSTS;
              const int i1_max      = i1 + window + N_GHOSTS;
              const int i2_min      = i2 - window + N_GHOSTS;
              const int i2_max      = i2 + window + N_GHOSTS;
              const int i3_min      = i3 - window + N_GHOSTS;
              const int i3_max      = i3 + window + N_GHOSTS;
              real_t    contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::Charge) {
                contrib = charge;
              } else if ((field == FieldID::N) || (field == FieldID::Nppc)) {
                contrib = ONE;
              } else if (field == FieldID::T) {
                real_t energy {
                  (mass == ZERO)
                    ? NORM(species.ux1(p), species.ux2(p), species.ux3(p))
                    : math::sqrt(ONE + NORM_SQR(species.ux1(p),
                                                species.ux2(p),
                                                species.ux3(p)))
                };
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
                const real_t x1 = get_prtl_x1(species, p);
                const real_t x2 = get_prtl_x2(species, p);
                const real_t x3 = get_prtl_x3(species, p);
                vec_t<Dim3>  u_hat;
                this_metric.v3_Cart2Hat(
                  { x1, x2, x3 },
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
                contrib *= inv_n0 / this_metric.sqrt_det_h(
                                      { static_cast<real_t>(i1) + HALF,
                                        static_cast<real_t>(i2) + HALF,
                                        static_cast<real_t>(i3) + HALF });
                if (use_weights) {
                  contrib *= species.weight(p);
                }
              }
              for (auto i3_ { i3_min }; i3_ <= i3_max; ++i3_) {
                for (auto i2_ { i2_min }; i2_ <= i2_max; ++i2_) {
                  for (auto i1_ { i1_min }; i1_ <= i1_max; ++i1_) {
                    buff_access(i1_, i2_, i3_, buff_ind) += contrib * smooth;
                  }
                }
              }
            }
          });
      }
    }
    Kokkos::Experimental::contribute(this->buff, scatter_buff);
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::CheckOutOfBounds(const std::string& msg,
                                         bool               only_on_debug) {
    if (only_on_debug) {
#ifndef DEBUG
      return;
#endif
    }
#if !defined(MPI_ENABLED)
    const auto ntags = 2;
#else
    const auto ntags = 2 + math::pow(3, (int)D) - 1;
#endif
    for (auto& species : particles) {
      auto found_oob   = array_t<int>("found_oob");
      auto found_oob_h = Kokkos::create_mirror_view(found_oob);
      if constexpr (D == Dim1) {
        const auto ni1 { (int)(this->Ni1()) };
        Kokkos::parallel_for(
          "Check-OutOfBounds",
          species.rangeActiveParticles(),
          ClassLambda(index_t p) {
            auto oob_found = ((species.tag(p) == ParticleTag::alive) &&
                              (species.i1(p) < 0 || species.i1(p) >= ni1)) ||
                             (species.tag(p) < 0 || species.tag(p) >= ntags);
            Kokkos::atomic_fetch_add(&found_oob(), (int)oob_found);
            if (oob_found) {
              printf("OutOfBounds particle at %ld %d (%d) %f (%f) [%d]\n",
                     p,
                     species.i1(p),
                     species.i1_prev(p),
                     species.dx1(p),
                     species.dx1_prev(p),
                     species.tag(p));
            }
          });
      } else if constexpr (D == Dim2) {
        const auto ni1 { (int)(this->Ni1()) }, ni2 { (int)(this->Ni2()) };
        Kokkos::parallel_for(
          "Check-OutOfBounds",
          species.rangeActiveParticles(),
          ClassLambda(index_t p) {
            auto oob_found = ((species.tag(p) == ParticleTag::alive) &&
                              (species.i1(p) < 0 || species.i1(p) >= ni1 ||
                               species.i2(p) < 0 || species.i2(p) >= ni2)) ||
                             (species.tag(p) < 0 || species.tag(p) >= ntags);
            Kokkos::atomic_fetch_add(&found_oob(), (int)oob_found);
            if (oob_found) {
              printf("OutOfBounds particle at %ld %d (%d) %d (%d) %f (%f) %f "
                     "(%f) [%d]\n",
                     p,
                     species.i1(p),
                     species.i1_prev(p),
                     species.i2(p),
                     species.i2_prev(p),
                     species.dx1(p),
                     species.dx1_prev(p),
                     species.dx2(p),
                     species.dx2_prev(p),
                     species.tag(p));
            }
          });
      } else if constexpr (D == Dim3) {
        const auto ni1 { (int)(this->Ni1()) }, ni2 { (int)(this->Ni2()) },
          ni3 { (int)(this->Ni3()) };
        Kokkos::parallel_for(
          "Check-OutOfBounds",
          species.rangeActiveParticles(),
          ClassLambda(index_t p) {
            auto oob_found = ((species.tag(p) == ParticleTag::alive) &&
                              (species.i1(p) < 0 || species.i1(p) >= ni1 ||
                               species.i2(p) < 0 || species.i2(p) >= ni2 ||
                               species.i3(p) < 0 || species.i3(p) >= ni3)) ||
                             (species.tag(p) < 0 || species.tag(p) >= ntags);
            Kokkos::atomic_fetch_add(&found_oob(), (int)oob_found);
            if (oob_found) {
              printf("OutOfBounds particle at %ld %d (%d) %d (%d) %d (%d) %f "
                     "(%f) %f (%f) %f (%f) [%d]\n",
                     p,
                     species.i1(p),
                     species.i1_prev(p),
                     species.i2(p),
                     species.i2_prev(p),
                     species.i3(p),
                     species.i3_prev(p),
                     species.dx1(p),
                     species.dx1_prev(p),
                     species.dx2(p),
                     species.dx2_prev(p),
                     species.dx3(p),
                     species.dx3_prev(p),
                     species.tag(p));
            }
          });
      }
      Kokkos::deep_copy(found_oob_h, found_oob);
      NTTHostErrorIf(found_oob_h() > 0,
                     fmt::format("%s: found %d OutOfBounds particles (%s)",
                                 msg.c_str(),
                                 found_oob_h(),
                                 species.label().c_str()));
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::CheckNaNs(const std::string& msg, CheckNaNFlags flags) {
    (void)msg;
    (void)flags;
#ifdef DEBUG
    if constexpr (D == Dim2) {
      auto found_nan   = array_t<int>("found_nan");
      auto found_nan_h = Kokkos::create_mirror_view(found_nan);
      if (flags & CheckNaN_Fields) {
        Kokkos::parallel_for(
          "Check-NAN-Fields",
          this->rangeActiveCells(),
          ClassLambda(index_t i1, index_t i2) {
            auto inf_found = Kokkos::isinf(this->em(i1, i2, em::ex1)) ||
                             Kokkos::isinf(this->em(i1, i2, em::ex2)) ||
                             Kokkos::isinf(this->em(i1, i2, em::ex3)) ||
                             Kokkos::isinf(this->em(i1, i2, em::bx1)) ||
                             Kokkos::isinf(this->em(i1, i2, em::bx2)) ||
                             Kokkos::isinf(this->em(i1, i2, em::bx3));
            auto nan_found = Kokkos::isnan(this->em(i1, i2, em::ex1)) ||
                             Kokkos::isnan(this->em(i1, i2, em::ex2)) ||
                             Kokkos::isnan(this->em(i1, i2, em::ex3)) ||
                             Kokkos::isnan(this->em(i1, i2, em::bx1)) ||
                             Kokkos::isnan(this->em(i1, i2, em::bx2)) ||
                             Kokkos::isnan(this->em(i1, i2, em::bx3));
            Kokkos::atomic_fetch_add(&found_nan(), (int)nan_found + (int)inf_found);
            if (nan_found || inf_found) {
              printf("NAN in fields at %ld %ld %f %f %f %f %f %f\n",
                     i1 - N_GHOSTS,
                     i2 - N_GHOSTS,
                     this->em(i1, i2, em::ex1),
                     this->em(i1, i2, em::ex2),
                     this->em(i1, i2, em::ex3),
                     this->em(i1, i2, em::bx1),
                     this->em(i1, i2, em::bx2),
                     this->em(i1, i2, em::bx3));
            }
          });
        Kokkos::deep_copy(found_nan_h, found_nan);
        NTTHostErrorIf(
          found_nan_h() > 0,
          fmt::format("%s: found %d NaNs in fields", msg.c_str(), found_nan_h()));
      }
      if (flags & CheckNaN_Currents) {
        Kokkos::parallel_for(
          "Check-NAN-Currents",
          this->rangeActiveCells(),
          ClassLambda(index_t i1, index_t i2) {
            auto nan_found = Kokkos::isnan(this->cur(i1, i2, cur::jx1)) ||
                             Kokkos::isnan(this->cur(i1, i2, cur::jx2)) ||
                             Kokkos::isnan(this->cur(i1, i2, cur::jx3));
            auto inf_found = Kokkos::isinf(this->cur(i1, i2, cur::jx1)) ||
                             Kokkos::isinf(this->cur(i1, i2, cur::jx2)) ||
                             Kokkos::isinf(this->cur(i1, i2, cur::jx3));
            Kokkos::atomic_fetch_add(&found_nan(), (int)nan_found + (int)inf_found);
            if (nan_found || inf_found) {
              printf("NAN in currents at %ld %ld %f %f %f\n",
                     i1 - N_GHOSTS,
                     i2 - N_GHOSTS,
                     this->cur(i1, i2, cur::jx1),
                     this->cur(i1, i2, cur::jx2),
                     this->cur(i1, i2, cur::jx3));
            }
          });
        Kokkos::deep_copy(found_nan_h, found_nan);
        NTTHostErrorIf(
          found_nan_h() > 0,
          fmt::format("%s: found %d NaNs in currents", msg.c_str(), found_nan_h()));
      }
      if (flags & CheckNaN_Particles) {
        for (auto& species : particles) {
          Kokkos::parallel_for(
            "Check-NAN-Particles",
            species.rangeActiveParticles(),
            ClassLambda(index_t p) {
              auto inf_found = Kokkos::isinf(species.ux1(p)) ||
                               Kokkos::isinf(species.ux2(p)) ||
                               Kokkos::isinf(species.ux3(p));
              auto nan_found = Kokkos::isnan(species.ux1(p)) ||
                               Kokkos::isnan(species.ux2(p)) ||
                               Kokkos::isnan(species.ux3(p));
              Kokkos::atomic_fetch_add(&found_nan(),
                                       (int)nan_found + (int)inf_found);
              if (nan_found || inf_found) {
                printf("NAN in particles at %ld %d (%d) %d (%d) %f (%f) %f "
                       "(%f), %f %f %f\n",
                       p,
                       species.i1(p),
                       species.i1_prev(p),
                       species.i2(p),
                       species.i2_prev(p),
                       species.dx1(p),
                       species.dx1_prev(p),
                       species.dx2(p),
                       species.dx2_prev(p),
                       species.ux1(p),
                       species.ux2(p),
                       species.ux3(p));
              }
            });
          Kokkos::deep_copy(found_nan_h, found_nan);
          NTTHostErrorIf(found_nan_h() > 0,
                         fmt::format("%s: found %d NaNs in particles",
                                     msg.c_str(),
                                     found_nan_h()));
        }
      }
    }
#endif
  }

} // namespace ntt

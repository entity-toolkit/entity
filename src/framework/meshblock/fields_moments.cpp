#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "output.h"
#include "particle_macros.h"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeMoments(const SimulationParams& params,
                                       const FieldID&          field,
                                       const std::vector<int>& components,
                                       const std::vector<int>& prtl_species,
                                       const int&              buff_ind,
                                       const short&            smooth) {
    NTTLog();
    // clear the buffer
    AssertEmptyContent({ this->buff_content[buff_ind] });
    std::size_t ni1 = this->Ni1(), ni2 = this->Ni2(), ni3 = this->Ni3();
    real_t weight = (1.0 / params.ppc0()) / math::pow(2.0 * smooth + 1.0, static_cast<int>(D));
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

    auto this_metric = this->metric;

    for (auto& sp : out_species) {
      auto species      = particles[sp - 1];
      auto scatter_buff = Kokkos::Experimental::create_scatter_view(this->buff);
      auto mass         = species.mass();
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (!species.is_dead(p)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              real_t x1          = get_prtl_x1(species, p);
              auto   i1_min      = IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i1_max      = IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::N) {
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
              for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                buff_access(i1_, buff_ind) += contrib * weight * this_metric.min_cell_volume()
                                              / this_metric.sqrt_det_h({ x1 });
              }
            }
          });
      } else if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (!species.is_dead(p)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              auto   i2          = species.i2(p);
              real_t x1          = get_prtl_x1(species, p);
              real_t x2          = get_prtl_x2(species, p);
              auto   i1_min      = IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i1_max      = IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i2_min      = IMIN(IMAX(i2 - smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              auto   i2_max      = IMIN(IMAX(i2 + smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::N) {
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
                  this_metric.v_Cart2Hat({ x1, x2, phi },
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
              for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                  buff_access(i1_, i2_, buff_ind) += contrib * weight
                                                     * this_metric.min_cell_volume()
                                                     / this_metric.sqrt_det_h({ x1, x2 });
                }
              }
            }
          });
      } else if constexpr (D == Dim3) {
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (!species.is_dead(p)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              auto   i2          = species.i2(p);
              auto   i3          = species.i3(p);
              real_t x1          = get_prtl_x1(species, p);
              real_t x2          = get_prtl_x2(species, p);
              real_t x3          = get_prtl_x3(species, p);
              auto   i1_min      = IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i1_max      = IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i2_min      = IMIN(IMAX(i2 - smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              auto   i2_max      = IMIN(IMAX(i2 + smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              auto   i3_min      = IMIN(IMAX(i3 - smooth + N_GHOSTS, 0), ni3 + 2 * N_GHOSTS);
              auto   i3_max      = IMIN(IMAX(i3 + smooth + N_GHOSTS, 0), ni3 + 2 * N_GHOSTS);
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
              } else if (field == FieldID::N) {
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
                this_metric.v_Cart2Hat(
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
              for (int i3_ = i3_min; i3_ <= i3_max; ++i3_) {
                for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                  for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                    buff_access(i1_, i2_, i3_, buff_ind)
                      += contrib * weight * this_metric.min_cell_volume()
                         / this_metric.sqrt_det_h({ x1, x2, x3 });
                  }
                }
              }
            }
          });
      }
    }
  }
}    // namespace ntt

#ifdef PIC_ENGINE
template void ntt::Meshblock<ntt::Dim1, ntt::PICEngine>::ComputeMoments(const SimulationParams&,
                                                                        const FieldID&,
                                                                        const std::vector<int>&,
                                                                        const std::vector<int>&,
                                                                        const int&,
                                                                        const short&);
template void ntt::Meshblock<ntt::Dim2, ntt::PICEngine>::ComputeMoments(const SimulationParams&,
                                                                        const FieldID&,
                                                                        const std::vector<int>&,
                                                                        const std::vector<int>&,
                                                                        const int&,
                                                                        const short&);
template void ntt::Meshblock<ntt::Dim3, ntt::PICEngine>::ComputeMoments(const SimulationParams&,
                                                                        const FieldID&,
                                                                        const std::vector<int>&,
                                                                        const std::vector<int>&,
                                                                        const int&,
                                                                        const short&);
#elif defined(GRPIC_ENGINE)

#endif
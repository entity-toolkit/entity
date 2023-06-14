#include "meshblock/meshblock.h"

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"
#include "species.h"

#include "meshblock/particles.h"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  Meshblock<D, S>::Meshblock(const std::vector<unsigned int>&    res,
                             const std::vector<real_t>&          ext,
                             const real_t*                       params,
                             const std::vector<ParticleSpecies>& species)
    : Mesh<D>(res, ext, params), Fields<D, S>(res) {
    for (auto& part : species) {
      particles.emplace_back(part);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::Verify() {
    NTTHostErrorIf(std::isnan(minCellSize()), "Minimum cell size evaluated to NaN");

    // verifying that the correct particle arrays are allocated for a given dimension ...
    // ... and a given simulation engine
    for (auto& species : particles) {
      if constexpr (D == Dim1) {
        NTTHostErrorIf(
          (species.i2.extent(0) != 0) || (species.i3.extent(0) != 0)
            || (species.dx2.extent(0) != 0) || (species.dx3.extent(0) != 0)
            || (species.i2_prev.extent(0) != 0) || (species.i3_prev.extent(0) != 0)
            || (species.dx2_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
          "Wrong particle arrays allocated for 1D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 1D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 1D mesh MINKOWSKI");
#endif
      } else if constexpr (D == Dim2) {
        NTTHostErrorIf((species.i3.extent(0) != 0) || (species.dx3.extent(0) != 0)
                         || (species.i3_prev.extent(0) != 0)
                         || (species.dx3_prev.extent(0) != 0),
                       "Wrong particle arrays allocated for 2D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0)
                           || (species.i2_prev.extent(0) != 0)
                           || (species.dx2_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      } else {
        if constexpr (S == PICEngine) {
          NTTHostErrorIf(
            (species.i1_prev.extent(0) != 0) || (species.dx1_prev.extent(0) != 0)
              || (species.i2_prev.extent(0) != 0) || (species.dx2_prev.extent(0) != 0)
              || (species.i3_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
            "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      }
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
    std::size_t ni1 = this->Ni1(), ni2 = this->Ni2(), ni3 = this->Ni3();
    real_t      weight = ONE / math::pow(2.0 * smooth + 1.0, static_cast<int>(D));
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
      if constexpr (D == Dim1) {
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.tag(p) == static_cast<short>(ParticleTag::alive)) {
              auto   buff_access = scatter_buff.access();
              auto   i1          = species.i1(p);
              real_t x1          = get_prtl_x1(species, p);
              auto   i1_min      = IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i1_max      = IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
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
              for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                buff_access(i1_, buff_ind) += contrib * weight;
              }
            }
          });
      } else if constexpr (D == Dim2) {
        Kokkos::parallel_for(
          "ComputeMoments", species.rangeActiveParticles(), Lambda(index_t p) {
            if (species.tag(p) == static_cast<short>(ParticleTag::alive)) {
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
              for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                  buff_access(i1_, i2_, buff_ind) += contrib * weight;
                }
              }
            }
          });
      } else if constexpr (D == Dim3) {
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
              auto   i1_min      = IMIN(IMAX(i1 - smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i1_max      = IMIN(IMAX(i1 + smooth + N_GHOSTS, 0), ni1 + 2 * N_GHOSTS);
              auto   i2_min      = IMIN(IMAX(i2 - smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              auto   i2_max      = IMIN(IMAX(i2 + smooth + N_GHOSTS, 0), ni2 + 2 * N_GHOSTS);
              auto   i3_min      = IMIN(IMAX(i3 - smooth + N_GHOSTS, 0), ni3 + 2 * N_GHOSTS);
              auto   i3_max      = IMIN(IMAX(i3 + smooth + N_GHOSTS, 0), ni3 + 2 * N_GHOSTS);
              real_t contrib { ZERO };
              if (field == FieldID::Rho) {
                contrib = ((mass == ZERO) ? ONE : mass);
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
              for (int i3_ = i3_min; i3_ <= i3_max; ++i3_) {
                for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                  for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
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

  template <Dimension D, SimulationEngine S>
  auto Meshblock<D, S>::RemoveDeadParticles(const double& max_dead_frac)
    -> std::vector<double> {
    std::vector<double> dead_fractions = {};
    for (auto& species : particles) {
      auto npart_tag = species.CountTaggedParticles();
      auto dead_fraction
        = species.npart() > 0
            ? (double)(npart_tag[(short)(ParticleTag::dead)]) / (double)(species.npart())
            : 0.0;
      if ((species.npart() > 0) && (dead_fraction >= (double)max_dead_frac)) {
        species.ReshuffleByTags();
        species.setNpart(npart_tag[(short)(ParticleTag::alive)]);
      }
      dead_fractions.push_back(dead_fraction);
    }
    return dead_fractions;
  }
}    // namespace ntt
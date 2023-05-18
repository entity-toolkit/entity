#include "meshblock.h"

#include "wrapper.h"

#include "particle_macros.h"
#include "particles.h"
#include "sim_params.h"
#include "species.h"

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
    // clear the buffer
    std::vector<Content> A = { this->buff_content[buff_ind] };
    AssertEmptyContent(A);
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
  void Meshblock<D, S>::PrepareFieldsForOutput(const PrepareOutputFlags& flags) {
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

    if constexpr (D == Dim1) {
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
          vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
          if (flags & PrepareOutput_ConvertToHat) {
            // !TODO: not quite correct when not in cell center
            this_metric.v3_Cntrv2Hat({ i_ + HALF }, e_cntr, e_hat);
            this_metric.v3_Cntrv2Hat({ i_ + HALF }, b_cntr, b_hat);
          } else {
            e_hat[0] = e_cntr[0];
            e_hat[1] = e_cntr[1];
            e_hat[2] = e_cntr[2];
            b_hat[0] = b_cntr[0];
            b_hat[1] = b_cntr[1];
            b_hat[2] = b_cntr[2];
          }
          this_bckp(i, em::ex1) = e_hat[0];
          this_bckp(i, em::ex2) = e_hat[1];
          this_bckp(i, em::ex3) = e_hat[2];
          this_bckp(i, em::bx1) = b_hat[0];
          this_bckp(i, em::bx2) = b_hat[1];
          this_bckp(i, em::bx3) = b_hat[2];
        });
    } else if constexpr (D == Dim2) {
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
          vec_t<Dim3> e_hat { ZERO }, b_hat { ZERO };
          if (flags & PrepareOutput_ConvertToHat) {
            // !TODO: not quite correct when not in cell center
            this_metric.v3_Cntrv2Hat({ i_ + HALF, j_ + HALF }, e_cntr, e_hat);
            this_metric.v3_Cntrv2Hat({ i_ + HALF, j_ + HALF }, b_cntr, b_hat);
          } else {
            e_hat[0] = e_cntr[0];
            e_hat[1] = e_cntr[1];
            e_hat[2] = e_cntr[2];
            b_hat[0] = b_cntr[0];
            b_hat[1] = b_cntr[1];
            b_hat[2] = b_cntr[2];
          }
          this_bckp(i, j, em::ex1) = e_hat[0];
          this_bckp(i, j, em::ex2) = e_hat[1];
          this_bckp(i, j, em::ex3) = e_hat[2];
          this_bckp(i, j, em::bx1) = b_hat[0];
          this_bckp(i, j, em::bx2) = b_hat[1];
          this_bckp(i, j, em::bx3) = b_hat[2];
        });
    } else if constexpr (D == Dim3) {
      NTTHostError("Not implemented.");
    }
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
#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "particle_macros.h"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::ComputeMoments(const SimulationParams& params,
                                       const Content&          content,
                                       const int&              ind,
                                       const short&            smooth) {
    auto this_metric = this->metric;
    AssertEmptyContent({ this->buff_content[ind] });
    std::size_t ni1 { 0 }, ni2 { 0 }, ni3 { 0 };
    real_t      weight { ZERO };
    if constexpr (D == Dim1) {
      weight          = (1.0 / params.ppc0()) / (2.0 * smooth + 1.0);
      ni1             = this->Ni1();
      auto buff_slice = Kokkos::subview(this->buff, Kokkos::ALL(), (int)ind);
      Kokkos::deep_copy(buff_slice, ZERO);
    } else if constexpr (D == Dim2) {
      weight          = (1.0 / params.ppc0()) / SQR(2.0 * smooth + 1.0);
      ni1             = this->Ni1();
      ni2             = this->Ni2();
      auto buff_slice = Kokkos::subview(this->buff, Kokkos::ALL(), Kokkos::ALL(), (int)ind);
      Kokkos::deep_copy(buff_slice, ZERO);
    } else if constexpr (D == Dim3) {
      weight = (1.0 / params.ppc0()) / CUBE(2.0 * smooth + 1.0);
      ni1    = this->Ni1();
      ni2    = this->Ni2();
      ni3    = this->Ni3();
      auto buff_slice
        = Kokkos::subview(this->buff, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL(), (int)ind);
      Kokkos::deep_copy(buff_slice, ZERO);
    }

    auto computing_for_photons = ((content == Content::photon_number_density)
                                  || (content == Content::photon_energy_density));
    auto computing_for_massive
      = ((content == Content::mass_density) || (content == Content::energy_density)
         || (content == Content::charge_density) || (content == Content::number_density));
    NTTHostErrorIf(!(computing_for_photons || computing_for_massive),
                   "Invalid content for ComputeMoments.");

    for (auto& species : particles) {
      auto scatter_buff = Kokkos::Experimental::create_scatter_view(this->buff);
      auto is_massless  = (species.mass() == 0.0);
      auto mass         = species.mass();
      auto charge       = species.charge();
      if ((is_massless && computing_for_photons) || (!is_massless && computing_for_massive)) {
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
                if (content == Content::mass_density) {
                  contrib = mass;
                } else if (content == Content::charge_density) {
                  contrib = charge;
                } else if ((content == Content::number_density)
                           || (content == Content::photon_number_density)) {
                  contrib = ONE;
                } else if (content == Content::energy_density) {
                  contrib = mass * get_prtl_Gamma_SR(species, p);
                } else if (content == Content::photon_energy_density) {
                  contrib = math::sqrt(get_prtl_Usqr_SR(species, p));
                }
                for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                  buff_access(i1_, ind) += contrib * weight * this_metric.min_cell_volume()
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
                if (content == Content::mass_density) {
                  contrib = mass;
                } else if (content == Content::charge_density) {
                  contrib = charge;
                } else if ((content == Content::number_density)
                           || (content == Content::photon_number_density)) {
                  contrib = ONE;
                } else if (content == Content::energy_density) {
                  contrib = mass * get_prtl_Gamma_SR(species, p);
                } else if (content == Content::photon_energy_density) {
                  contrib = math::sqrt(get_prtl_Usqr_SR(species, p));
                }
                for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                  for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                    buff_access(i1_, i2_, ind) += contrib * weight
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
                if (content == Content::mass_density) {
                  contrib = mass;
                } else if (content == Content::charge_density) {
                  contrib = charge;
                } else if ((content == Content::number_density)
                           || (content == Content::photon_number_density)) {
                  contrib = ONE;
                } else if (content == Content::energy_density) {
                  contrib = mass * get_prtl_Gamma_SR(species, p);
                } else if (content == Content::photon_energy_density) {
                  contrib = math::sqrt(get_prtl_Usqr_SR(species, p));
                }
                for (int i3_ = i3_min; i3_ <= i3_max; ++i3_) {
                  for (int i2_ = i2_min; i2_ <= i2_max; ++i2_) {
                    for (int i1_ = i1_min; i1_ <= i1_max; ++i1_) {
                      buff_access(i1_, i2_, i3_, ind)
                        += contrib * weight * this_metric.min_cell_volume()
                           / this_metric.sqrt_det_h({ x1, x2, x3 });
                    }
                  }
                }
              }
            });
        }
      }
      Kokkos::Experimental::contribute(this->buff, scatter_buff);
    }
    this->buff_content[ind] = content;
  }
}    // namespace ntt

#ifdef PIC_ENGINE
template void ntt::Meshblock<ntt::Dim1, ntt::PICEngine>::ComputeMoments(
  const SimulationParams&, const Content&, const int&, const short&);
template void ntt::Meshblock<ntt::Dim2, ntt::PICEngine>::ComputeMoments(
  const SimulationParams&, const Content&, const int&, const short&);
template void ntt::Meshblock<ntt::Dim3, ntt::PICEngine>::ComputeMoments(
  const SimulationParams&, const Content&, const int&, const short&);
#elif defined(GRPIC_ENGINE)

#endif
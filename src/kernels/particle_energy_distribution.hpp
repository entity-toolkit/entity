/**
 * @file kernels/particle_moments.hpp
 * @brief Kernels for computing particle distribution functions
 * @implements
 *   - kernel::EnergyBinning
 *   - kernel::ParticleDistributionBase_kernel<>
 *   - kernel::ParticleDistribution_kernel<>
 *   - kernel::ParticleDistributionSpatial_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_PARTICLE_ENERGY_DISTRIBUTION_HPP
#define KERNELS_PARTICLE_ENERGY_DISTRIBUTION_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"

namespace kernel {
  using namespace ntt;

  struct EnergyBinning {
    const real_t e_min, e_max;
    const bool   log_bins;
    const size_t n_bins;

    EnergyBinning(real_t e_min, real_t e_max, bool log_bins, size_t n_bins)
      : e_min { e_min }
      , e_max { e_max }
      , log_bins { log_bins }
      , n_bins { n_bins } {}
  };

  template <SimEngine::type S, MetricClass M>
  struct ParticleDistributionBase_kernel {
    const ParticleArrays particles;
    const bool           is_massive;
    const M              metric;

    const EnergyBinning energy_binning;

    ParticleDistributionBase_kernel(const Particles<M::Dim, M::CoordType>& particles,
                                    const EnergyBinning& energy_binning,
                                    const M&             metric)
      : particles { static_cast<const ParticleArrays&>(particles) }
      , is_massive { (particles.mass() != 0.0f) }
      , energy_binning { energy_binning }
      , metric { metric } {}

    Inline auto EnergyBinIndex(prtlidx_t p) const -> size_t {
      real_t en;
      if constexpr (S == SimEngine::SRPIC) {
        if (is_massive) {
          en = U2GAMMA(particles.ux1(p), particles.ux2(p), particles.ux3(p)) - ONE;
        } else {
          en = NORM(particles.ux1(p), particles.ux2(p), particles.ux3(p));
        }
      } else if constexpr (S == SimEngine::GRPIC) {
        coord_t<M::Dim> x_Code { ZERO };
        x_Code[0] = static_cast<real_t>(particles.i1(p)) +
                    static_cast<real_t>(particles.dx1(p));
        x_Code[1] = static_cast<real_t>(particles.i2(p)) +
                    static_cast<real_t>(particles.dx2(p));

        // raise full covariant 4-vector to get correct contravariant u^0
        // u^i != h^{ij} u_j
        const real_t    u_0_cov { metric.u_0(
          x_Code,
          { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          (is_massive) ? ONE : ZERO) };
        vec_t<Dim::_4D> u_cntrv_4d { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(
          x_Code,
          { u_0_cov, particles.ux1(p), particles.ux2(p), particles.ux3(p) },
          u_cntrv_4d);
        // in GR: u^0 = Gamma/alpha
        const real_t Gamma { metric.alpha(x_Code) * u_cntrv_4d[0] };
        en = is_massive ? (Gamma - ONE) : Gamma;
      }
      if (energy_binning.log_bins) {
        en = math::log10(en);
      }
      if (en <= energy_binning.e_min) {
        return 0u;
      } else if (en >= energy_binning.e_max) {
        return energy_binning.n_bins;
      } else {
        return static_cast<size_t>(static_cast<real_t>(energy_binning.n_bins) *
                                   (en - energy_binning.e_min) /
                                   (energy_binning.e_max - energy_binning.e_min));
      }
    }
  };

  template <SimEngine::type S, MetricClass M>
  class ParticleDistribution_kernel
    : public ParticleDistributionBase_kernel<S, M> {
    scatter_array_t<real_t*> dn_scatter;
    using ParticleDistributionBase_kernel<S, M>::particles;
    using ParticleDistributionBase_kernel<S, M>::EnergyBinIndex;

  public:
    ParticleDistribution_kernel(const Particles<M::Dim, M::CoordType>& particles,
                                const scatter_array_t<real_t*>& dn_scatter,
                                const EnergyBinning&            energy_binning,
                                const M&                        metric)
      : ParticleDistributionBase_kernel<S, M>(particles, energy_binning, metric)
      , dn_scatter { dn_scatter } {}

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        return;
      }
      const auto e_ind   = EnergyBinIndex(p);
      auto       dn_acc  = dn_scatter.access();
      dn_acc(e_ind)     += particles.weight(p);
    }
  };

  template <SimEngine::type S, MetricClass M>
  class ParticleDistributionSpatial_kernel
    : public ParticleDistributionBase_kernel<S, M> {
    static constexpr auto D     = M::Dim;
    static constexpr auto Dim   = static_cast<uint8_t>(D);
    static constexpr auto DimP1 = Dim + 1u;
    using ParticleDistributionBase_kernel<S, M>::particles;
    using ParticleDistributionBase_kernel<S, M>::EnergyBinIndex;

    scatter_nddata_t<DimP1, real_t> dn_scatter;
    const array_t<real_t[Dim]>      nmin_i;
    const array_t<real_t[Dim]>      dncells_i;
    const array_t<size_t[Dim]>      nbins_i;

  public:
    ParticleDistributionSpatial_kernel(
      const Particles<M::Dim, M::CoordType>& particles,
      const scatter_nddata_t<DimP1, real_t>& dn_scatter,
      const array_t<real_t[Dim]>             nmin_i,
      const array_t<real_t[Dim]>             dncells_i,
      const array_t<size_t[Dim]>             nbins_i,
      const EnergyBinning&                   energy_binning,
      const M&                               metric)
      : ParticleDistributionBase_kernel<S, M>(particles, energy_binning, metric)
      , dn_scatter { dn_scatter }
      , nmin_i { nmin_i }
      , dncells_i { dncells_i }
      , nbins_i { nbins_i } {}

    Inline auto SpatialBinIndex(real_t i, real_t nmin, real_t dncells, size_t nbins) const
      -> size_t {
      const auto ni = i + nmin;
      if (ni <= ZERO) {
        return 0u;
      } else if (ni >= dncells * static_cast<real_t>(nbins)) {
        return nbins - 1u;
      } else {
        return static_cast<size_t>(static_cast<real_t>(nbins) * ni / dncells);
      }
    }

    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        return;
      }
      const auto e_ind = EnergyBinIndex(p);
      if constexpr (D == Dim::_1D) {
        const auto i1_ind = SpatialBinIndex(static_cast<real_t>(particles.i1(p)) +
                                              static_cast<real_t>(particles.dx1(p)),
                                            nmin_i(0),
                                            dncells_i(0),
                                            nbins_i(0));
        auto dn_acc            = dn_scatter.access();
        dn_acc(i1_ind, e_ind) += particles.weight(p);
      } else if constexpr (D == Dim::_2D) {
        const auto i1_ind = SpatialBinIndex(static_cast<real_t>(particles.i1(p)) +
                                              static_cast<real_t>(particles.dx1(p)),
                                            nmin_i(0),
                                            dncells_i(0),
                                            nbins_i(0));
        const auto i2_ind = SpatialBinIndex(static_cast<real_t>(particles.i2(p)) +
                                              static_cast<real_t>(particles.dx2(p)),
                                            nmin_i(1),
                                            dncells_i(1),
                                            nbins_i(1));
        auto dn_acc                    = dn_scatter.access();
        dn_acc(i1_ind, i2_ind, e_ind) += particles.weight(p);
      } else if constexpr (D == Dim::_3D) {
        const auto i1_ind = SpatialBinIndex(static_cast<real_t>(particles.i1(p)) +
                                              static_cast<real_t>(particles.dx1(p)),
                                            nmin_i(0),
                                            dncells_i(0),
                                            nbins_i(0));
        const auto i2_ind = SpatialBinIndex(static_cast<real_t>(particles.i2(p)) +
                                              static_cast<real_t>(particles.dx2(p)),
                                            nmin_i(1),
                                            dncells_i(1),
                                            nbins_i(1));
        const auto i3_ind = SpatialBinIndex(static_cast<real_t>(particles.i3(p)) +
                                              static_cast<real_t>(particles.dx3(p)),
                                            nmin_i(2),
                                            dncells_i(2),
                                            nbins_i(2));
        auto dn_acc                            = dn_scatter.access();
        dn_acc(i1_ind, i2_ind, i3_ind, e_ind) += particles.weight(p);
      } else {
        raise::KernelError(HERE, "invalid dimension");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_PARTICLE_ENERGY_DISTRIBUTION_HPP

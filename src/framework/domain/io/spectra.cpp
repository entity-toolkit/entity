#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"

#include "framework/containers/particles.h"
#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"
#include "kernels/particle_energy_distribution.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void GenerateEnergyDistribution(const Particles<M::Dim, M::CoordType>& species,
                                  array_t<real_t*>&            dn,
                                  const kernel::EnergyBinning& energy_binning,
                                  const M&                     metric) {
    auto dn_scatter = Kokkos::Experimental::create_scatter_view(dn);
    Kokkos::parallel_for("ComputeSpectraSpatial",
                         species.rangeActiveParticles(),
                         kernel::ParticleDistribution_kernel<S, M> { species,
                                                                     dn_scatter,
                                                                     energy_binning,
                                                                     metric });
    Kokkos::Experimental::contribute(dn, dn_scatter);
  }

  template <SimEngine::type S, MetricClass M>
  void GenerateSpatialEnergyDistribution(
    const Particles<M::Dim, M::CoordType>&              species,
    nddata_t<static_cast<uint8_t>(M::Dim) + 1, real_t>& dn,
    const array_t<real_t[M::Dim]>&                      nmin_i,
    const array_t<real_t[M::Dim]>&                      dncells_i,
    const array_t<size_t[M::Dim]>&                      nbins_i,
    const kernel::EnergyBinning&                        energy_binning,
    const M&                                            metric) {
    auto dn_scatter = Kokkos::Experimental::create_scatter_view(dn);
    Kokkos::parallel_for(
      "ComputeSpectra",
      species.rangeActiveParticles(),
      kernel::ParticleDistributionSpatial_kernel<S, M> { species,
                                                         dn_scatter,
                                                         nmin_i,
                                                         dncells_i,
                                                         nbins_i,
                                                         energy_binning,
                                                         metric });
    Kokkos::Experimental::contribute(dn, dn_scatter);
  }

  template <idx_t O, MetricClass M>
  void GenerateSpatialBins(array_t<real_t*>& bins,
                           const M&          metric,
                           real_t            dncells,
                           size_t            nbins) {
    Kokkos::parallel_for(
      "GenerateSpatialBins",
      nbins + 1,
      Lambda(uint32_t b) {
        bins(b) = metric.template convert<O, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(b) * dncells);
      });
  }

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::WriteSpectra(const SimulationParams& params,
                                      Domain<S, M>*           local_domain,
                                      timestep_t              current_step,
                                      simtime_t               current_time) {
    constexpr auto dim = static_cast<size_t>(M::Dim);
    g_writer.beginWriting(WriteMode::Spectra, current_step, current_time);
    const auto log_bins = params.template get<bool>("output.spectra.log_bins");

    const auto num_energy_bins = params.template get<size_t>(
      "output.spectra.num_energy_bins");
    auto e_min = params.template get<real_t>("output.spectra.e_min");
    auto e_max = params.template get<real_t>("output.spectra.e_max");
    if (log_bins) {
      e_min = math::log10(e_min);
      e_max = math::log10(e_max);
    }

    const auto num_spatial_bins = params.template get<std::vector<size_t>>(
      "output.spectra.num_spatial_bins");
    const auto spatial_binning_enabled = std::any_of(num_spatial_bins.begin(),
                                                     num_spatial_bins.end(),
                                                     [](const auto& n) {
                                                       return n != 1u;
                                                     });

    // fractional number of cells per each direction in each bin
    const array_t<real_t[dim]> dncells_i { "dncells_i" };
    // left edge of the local domain in each direction in number of cells
    const array_t<real_t[dim]> nmin_i { "nmin_i" };
    const array_t<size_t[dim]> nbins_i { "nbins_i" };
    if (spatial_binning_enabled) {
      auto dncells_i_h = Kokkos::create_mirror_view(dncells_i);
      auto nmin_i_h    = Kokkos::create_mirror_view(nmin_i);
      auto nbins_i_h   = Kokkos::create_mirror_view(nbins_i);
      for (auto d = 0u; d < dim; ++d) {
        dncells_i_h(d) = static_cast<real_t>(mesh().n_active(static_cast<in>(d))) /
                         static_cast<real_t>(num_spatial_bins[d]);
        nmin_i_h(d)  = static_cast<real_t>(local_domain->offset_ncells()[d]);
        nbins_i_h(d) = num_spatial_bins[d];
      }
      Kokkos::deep_copy(dncells_i, dncells_i_h);
      Kokkos::deep_copy(nmin_i, nmin_i_h);
      Kokkos::deep_copy(nbins_i, nbins_i_h);
    }
    for (const auto& spec : g_writer.spectraWriters()) {
      auto& species = local_domain->species[spec.species() - 1];
      if (not spatial_binning_enabled) {
        array_t<real_t*> dn { "dn", num_energy_bins };
        GenerateEnergyDistribution<S, M>(species,
                                         dn,
                                         { e_min, e_max, log_bins, num_energy_bins },
                                         local_domain->mesh.metric);
        g_writer.writeSpectrum(dn, spec.name());
      } else {
        nddata_t<dim + 1u, real_t> dn;
        if constexpr (M::Dim == Dim::_1D) {
          dn = { "dn", num_spatial_bins[0], num_energy_bins };
        } else if constexpr (M::Dim == Dim::_2D) {
          dn = { "dn", num_spatial_bins[0], num_spatial_bins[1], num_energy_bins };
        } else if constexpr (M::Dim == Dim::_3D) {
          dn = { "dn",
                 num_spatial_bins[0],
                 num_spatial_bins[1],
                 num_spatial_bins[2],
                 num_energy_bins };
        } else {
          raise::Error("invalid dimension", HERE);
        }
        GenerateSpatialEnergyDistribution<S, M>(
          species,
          dn,
          nmin_i,
          dncells_i,
          nbins_i,
          { e_min, e_max, log_bins, num_energy_bins },
          local_domain->mesh.metric);
        g_writer.writeSpectrumSpatial<static_cast<uint8_t>(dim + 1u)>(dn,
                                                                      spec.name());
      }
    }

    {
      // energy bins
      const array_t<real_t*> energy { "energy", num_energy_bins + 1 };
      Kokkos::parallel_for(
        "GenerateEnergyBins",
        num_energy_bins + 1,
        Lambda(uint32_t e) {
          if (log_bins) {
            energy(e) = math::pow(static_cast<real_t>(10),
                                  e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                            static_cast<real_t>(num_energy_bins));
          } else {
            energy(e) = e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                  static_cast<real_t>(num_energy_bins);
          }
        });

      g_writer.writeSpectrumBins(energy, "sEbn");
    }
    if (spatial_binning_enabled) {
      const auto metric   = mesh().metric;
      const auto n_active = mesh().n_active();
      for (auto d = 0u; d < dim; ++d) {
        array_t<real_t*> xi { "xi", num_spatial_bins[d] + 1 };
        // # of cells per spatial bin in direction `d` (fractional)
        const auto       dncells = static_cast<real_t>(n_active[d]) /
                             static_cast<real_t>(num_spatial_bins[d]);
        if (d == 0) {
          GenerateSpatialBins<1>(xi, metric, dncells, num_spatial_bins[d]);
        } else if (d == 1) {
          if constexpr (dim > 1) {
            GenerateSpatialBins<2>(xi, metric, dncells, num_spatial_bins[d]);
          }
        } else if (d == 2) {
          if constexpr (dim > 2) {
            GenerateSpatialBins<3>(xi, metric, dncells, num_spatial_bins[d]);
          }
        } else {
          raise::Error("invalid dimension", HERE);
        }
        g_writer.writeSpectrumBins(xi, "sX" + std::to_string(d + 1) + "bn");
      }
    }
    g_writer.endWriting(WriteMode::Spectra);
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_OUTPUT(S, M, D)                                             \
  template void Metadomain<S, M<D>>::WriteSpectra(const SimulationParams&,     \
                                                  Domain<S, M<D>>*,            \
                                                  timestep_t,                  \
                                                  simtime_t);

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_OUTPUT)

#undef METADOMAIN_OUTPUT
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt

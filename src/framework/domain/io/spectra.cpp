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
#include "kernels/particle_moments.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::WriteSpectra(const SimulationParams& params,
                                      Domain<S, M>*           local_domain,
                                      timestep_t              current_step,
                                      simtime_t               current_time) {

    g_writer.beginWriting(WriteMode::Spectra, current_step, current_time);
    const auto log_bins = params.template get<bool>("output.spectra.log_bins");
    const auto n_bins   = params.template get<size_t>("output.spectra.n_bins");
    auto       e_min    = params.template get<real_t>("output.spectra.e_min");
    auto       e_max    = params.template get<real_t>("output.spectra.e_max");
    if (log_bins) {
      e_min = math::log10(e_min);
      e_max = math::log10(e_max);
    }
    const array_t<real_t*> energy { "energy", n_bins + 1 };
    Kokkos::parallel_for(
      "GenerateEnergyBins",
      n_bins + 1,
      Lambda(uint32_t e) {
        if (log_bins) {
          energy(e) = math::pow(static_cast<real_t>(10),
                                e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                          static_cast<real_t>(n_bins));
        } else {
          energy(e) = e_min + (e_max - e_min) * static_cast<real_t>(e) /
                                static_cast<real_t>(n_bins);
        }
      });
    for (const auto& spec : g_writer.spectraWriters()) {
      auto&            species = local_domain->species[spec.species() - 1];
      array_t<real_t*> dn { "dn", n_bins };
      auto dn_scatter = Kokkos::Experimental::create_scatter_view(dn);
      Kokkos::parallel_for(
        "ComputeSpectra",
        species.rangeActiveParticles(),
        kernel::ParticleDistribution_kernel<S, M> { species,
                                                    dn_scatter,
                                                    e_min,
                                                    e_max,
                                                    log_bins,
                                                    n_bins,
                                                    local_domain->mesh.metric });
      Kokkos::Experimental::contribute(dn, dn_scatter);
      g_writer.writeSpectrum(dn, spec.name());
    }
    g_writer.writeSpectrumBins(energy, "sEbn");
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

// if (write_spectra3D) {
//   g_writer.beginWriting(WriteMode::Spectra3D, current_step, current_time);
//   const auto log_bins = params.template get<bool>(
//     "output.spectra3D.log_bins");
//   const auto n_bins = params.template get<std::size_t>(
//     "output.spectra3D.n_bins");
//   const auto& metric   = local_domain->mesh.metric;
//   // extract the number of bins globally in each direction
//   const auto  nx1_bins = params.template get<std::size_t>(
//     "output.spectra3D.nx1");
//   const auto nx2_bins = params.template get<std::size_t>(
//     "output.spectra3D.nx2");
//   const auto nx3_bins = params.template get<std::size_t>(
//     "output.spectra3D.nx3");
//
//   // select the min and max energy for the spectra
//   auto e_min = params.template get<real_t>("output.spectra3D.e_min");
//   auto e_max = params.template get<real_t>("output.spectra3D.e_max");
//
//   auto x1_min = mesh().extent(in::x1).first; // extent is in physical units, not code
//   decltype(x1_min) x2_min = 0;
//   decltype(x1_min) x3_min = 0;
//
//   auto x1_max = mesh().extent(in::x1).second; // pairs in c++ are addressed by first and secocnd
//   decltype(x1_max) x2_max = 0;
//   decltype(x1_max) x3_max = 0;
//
//   if constexpr (D == Dim::_2D or
//                 D == Dim::_3D) { // only pick x2 if simulation is in 2D
//
//     x2_min = mesh().extent(in::x2).first;
//     x2_max = mesh().extent(in::x2).second;
//   }
//
//   if constexpr (D == Dim::_3D) {
//
//     x3_min = mesh().extent(in::x3).first;
//     x3_max = mesh().extent(in::x3).second;
//   }
//
//   // auto dx_slice = mesh().dx1;
//
//   // auto dxslice = (x1_max - x1_min) / x_bins
//
//   if (log_bins) {
//     e_min = math::log10(e_min);
//     e_max = math::log10(e_max);
//   }
//   array_t<real_t*> energy { "energy", n_bins + 1 };
//   // generating the energy bins
//   Kokkos::parallel_for(
//     "GenerateEnergyBins",
//     n_bins + 1,
//     Lambda(uint32_t e) {
//       if (log_bins) {
//         energy(e) = math::pow(10.0, e_min + (e_max - e_min) * e / n_bins);
//       } else {
//         energy(e) = e_min + (e_max - e_min) * e / n_bins;
//       }
//     });
//
//   for (const auto& spec : g_writer.spectraWriters()) {
//     auto& species = local_domain->species[spec.species() - 1];
//     array_t<real_t****> dn3d { "dn3d", nx1_bins, nx2_bins, nx3_bins, n_bins };
//     auto dn3d_scatter = Kokkos::Experimental::create_scatter_view(dn3d);
//     auto ux1          = species.ux1;
//     auto ux2          = species.ux2;
//     auto ux3          = species.ux3;
//     auto i1           = species.i1;
//     auto dx1          = species.dx1;
//     // adeep_copy;
//     decltype(i1)  i2;
//     decltype(i1)  i3;
//     decltype(dx1) dx2;
//     decltype(dx1) dx3;
//     if constexpr (D == Dim::_2D or
//                   D == Dim::_3D) { // only pick x2 if simulation is in 2D
//       i2  = species.i2;
//       dx2 = species.dx2;
//     }
//     if constexpr (D == Dim::_3D) {
//       i3  = species.i3;
//       dx3 = species.dx3;
//     }
//     auto       weight     = species.weight;
//     auto       tag        = species.tag;
//     const auto is_massive = species.mass() > 0.0f;
//     Kokkos::parallel_for(
//       "ComputeSpectra",
//       species.rangeActiveParticles(),
//       Lambda(prtlidx_t p) {
//         if (tag(p) != ParticleTag::alive) {
//           return;
//         }
//
//         coord_t<D> x_Cd { ZERO };
//         if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
//           x_Cd[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
//         }
//         if constexpr (D == Dim::_2D or D == Dim::_3D) {
//           x_Cd[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
//         }
//         if constexpr (D == Dim::_3D) {
//           x_Cd[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
//         }
//         coord_t<D> x_Ph { ZERO };
//         metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
//
//         real_t en;
//         if (is_massive) {
//           en = U2GAMMA(ux1(p), ux2(p), ux3(p)) - ONE;
//         } else {
//           en = NORM(ux1(p), ux2(p), ux3(p));
//         }
//         if (log_bins) {
//           en = math::log10(en);
//         }
//         std::size_t e_ind = 0;
//         if (en <= e_min) {
//           e_ind = 0;
//         } else if (en >= e_max) {
//           e_ind = n_bins - 1;
//         } else {
//           e_ind = static_cast<std::size_t>(
//             static_cast<real_t>(n_bins) * (en - e_min) / (e_max - e_min));
//         }
//
//         std::size_t x1_ind = 0;
//         if (x_Ph[0] <= x1_min) {
//           x1_ind = 0;
//         } else if (x_Ph[0] >= x1_max) {
//           x1_ind = nx1_bins - 1;
//         } else {
//           x1_ind = static_cast<std::size_t>(static_cast<real_t>(nx1_bins) *
//                                             (x_Ph[0] - x1_min) /
//                                             (x1_max - x1_min));
//         }
//
//         std::size_t x2_ind = 0;
//         if constexpr (D == Dim::_2D or
//                       D == Dim::_3D) { // only pick x2 if simulation is in 2D
//           if (x_Ph[1] <= x2_min) {
//             x2_ind = 0;
//           } else if (x_Ph[1] >= x2_max) {
//             x2_ind = nx2_bins - 1;
//           } else {
//             x2_ind = static_cast<std::size_t>(static_cast<real_t>(nx2_bins) *
//                                               (x_Ph[1] - x2_min) /
//                                               (x2_max - x2_min));
//           }
//         }
//         std::size_t x3_ind = 0;
//         if constexpr (D == Dim::_3D) { // only pick x3 if simulation is in 3D
//
//           if (x_Ph[2] <= x3_min) {
//             x3_ind = 0;
//           } else if (x_Ph[2] >= x3_max) {
//             x3_ind = nx3_bins - 1;
//           } else {
//             x3_ind = static_cast<std::size_t>(static_cast<real_t>(nx3_bins) *
//                                               (x_Ph[2] - x3_min) /
//                                               (x3_max - x3_min));
//           }
//         }
//
//         // now I want to save the nx_bins contained in each rank
//         // can save array of x1_inds which are saved?
//         // maybe I can ask what local x_min and x_max are for this rank, pass that to writeSpectrum3D
//
//         auto dn3d_acc                            = dn3d_scatter.access();
//         dn3d_acc(x1_ind, x2_ind, x3_ind, e_ind) += weight(p);
//       });
//     Kokkos::Experimental::contribute(dn3d, dn3d_scatter);
//     g_writer.writeSpectrum3D(dn3d, spec.name());
//   }
//   g_writer.writeSpectrumBins(energy, "sEbn");
//   g_writer.endWriting(WriteMode::Spectra3D);
// }

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"
#include "utils/log.h"

#include "framework/domain/domain.h"
#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "framework/specialization_registry.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <string>
#include <vector>

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  auto Metadomain<S, M>::Write(const SimulationParams&      params,
                               timestep_t                   current_step,
                               timestep_t                   finished_step,
                               simtime_t                    current_time,
                               simtime_t                    finished_time,
                               const custom_field_output_t& CustomFieldOutput)
    -> bool {
    raise::ErrorIf(
      l_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    const auto write_fields = params.template get<bool>(
                                "output.fields.enable") and
                              g_writer.shouldWrite("fields",
                                                   finished_step,
                                                   finished_time);
    const auto write_particles = params.template get<bool>(
                                   "output.particles.enable") and
                                 g_writer.shouldWrite("particles",
                                                      finished_step,
                                                      finished_time);
    const auto write_spectra = params.template get<bool>(
                                 "output.spectra.enable") and
                               g_writer.shouldWrite("spectra",
                                                    finished_step,
                                                    finished_time);
    const auto extension = params.template get<std::string>("output.format");
    if (not(write_fields or write_particles or write_spectra) and
        extension != "disabled") {
      return false;
    }
    auto local_domain = subdomain_ptr(l_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing output", HERE);
    if (write_fields) {
      WriteFields(params,
                  local_domain,
                  current_step,
                  finished_step,
                  current_time,
                  finished_time,
                  CustomFieldOutput);
    } // end shouldWrite("fields", step, time)

    if (write_particles) {
      g_writer.beginWriting(WriteMode::Particles, current_step, current_time);
      const auto prtl_stride = params.template get<npart_t>(
        "output.particles.stride");
      for (const auto spec : g_writer.speciesIndices()) {
        local_domain->species[spec - 1].template OutputWrite<S, M>(
          g_writer.io(),
          g_writer.writer(),
          prtl_stride,
          ndomains(),
          local_domain->index(),
          local_domain->mesh.metric);
      }
      g_writer.endWriting(WriteMode::Particles);
    } // end shouldWrite("particles", step, time)

    if (write_spectra) {
      WriteSpectra(params, local_domain, current_step, current_time);
    } // end shouldWrite("spectra", step, time)

    return true;
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_OUTPUT(S, M, D)                                             \
  template auto Metadomain<S, M<D>>::Write(const SimulationParams&,            \
                                           timestep_t,                         \
                                           timestep_t,                         \
                                           simtime_t,                          \
                                           simtime_t,                          \
                                           const custom_field_output_t&) -> bool;

  NTT_FOREACH_SPECIALIZATION(METADOMAIN_OUTPUT)

#undef METADOMAIN_OUTPUT
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/error.h"

#include "framework/domain/mesh.h"
#include "framework/domain/metadomain.h"
#include "framework/specialization_registry.h"

namespace ntt {

  template <SimEngine::type S, MetricClass M>
  void Metadomain<S, M>::ShiftByCells(int n_cells, in dir)
    requires(CartesianMetricClass<M>)
  {
    raise::ErrorIf(
      n_cells == 0,
      "ShiftByCells called with n_cells = 0, no shift will be performed",
      HERE);

    {
      // update metadomain g_mesh
      auto       new_extent  = g_mesh.extent();
      const auto delta_shift = static_cast<real_t>(n_cells) *
                               g_mesh.metric.get_dx();
      new_extent[static_cast<int>(dir)].first  += delta_shift;
      new_extent[static_cast<int>(dir)].second += delta_shift;

      g_mesh.set_extent(new_extent);
    }

    // update individual subdomain meshes
    for (auto& subdomain : g_subdomains) {
      auto       new_extent  = subdomain.mesh.extent();
      const auto delta_shift = static_cast<real_t>(n_cells) *
                               subdomain.mesh.metric.get_dx();
      new_extent[static_cast<int>(dir)].first  += delta_shift;
      new_extent[static_cast<int>(dir)].second += delta_shift;

      subdomain.mesh.set_extent(new_extent);
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define METADOMAIN_SHIFT(S, M, D)                                              \
  template void Metadomain<S, M<D>>::ShiftByCells(int n_cells, in dir);

  NTT_FOREACH_CARTESIAN_SPECIALIZATION(METADOMAIN_SHIFT)
#undef METADOMAIN_SHIFT
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt
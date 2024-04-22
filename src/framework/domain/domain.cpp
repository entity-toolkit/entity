#include "framework/domain/domain.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/parameters.h"

#include <vector>

namespace ntt {

#if defined(OUTPUT_ENABLED)
  template <SimEngine::type S, class M>
  void Domain<S, M>::InitWriter(
    const SimulationParams&          params,
    const std::vector<std::size_t>&  glob_shape,
    const std::vector<unsigned int>& glob_ndomains_per_dim) {
    raise::ErrorIf(is_placeholder(), "Writer called on a placeholder domain", HERE);
    raise::ErrorIf(glob_shape.size() != M::Dim, "Invalid global shape size", HERE);
    raise::ErrorIf(glob_ndomains_per_dim.size() != M::Dim,
                   "Invalid global ndomains size",
                   HERE);

    const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");

    auto glob_shape_with_ghosts = glob_shape;
    auto off_ncells_with_ghosts = offset_ncells();
    auto off_ndomains           = offset_ndomains();
    auto loc_shape_with_ghosts  = mesh.n_active();
    if (incl_ghosts) {
      for (auto d { 0 }; d <= M::Dim; ++d) {
        glob_shape_with_ghosts[d] += 2 * N_GHOSTS * glob_ndomains_per_dim[d];
        off_ncells_with_ghosts[d] += 2 * N_GHOSTS * off_ndomains[d];
        loc_shape_with_ghosts[d]  += 2 * N_GHOSTS;
      }
    }

    m_writer.defineFieldLayout(glob_shape,
                               off_ncells_with_ghosts,
                               loc_shape_with_ghosts,
                               incl_ghosts);
    m_writer.defineFieldOutputs(
      S,
      params.template get<std::vector<std::string>>("output.fields"));
  }
#endif

  template struct Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Domain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Domain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Domain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Domain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Domain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Domain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt
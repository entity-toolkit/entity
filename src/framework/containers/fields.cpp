#include "framework/containers/fields.h"

#include "enums.h"
#include "global.h"

#include "traits/engine.h"

#include "framework/specialization_registry.h"

#include <vector>

namespace ntt {

  template <Dimension D, SimEngine::type S>
  Fields<D, S>::Fields(const std::vector<ncells_t>& res) {
    ncells_t nx1, nx2, nx3;
    nx1 = res[0] + 2 * N_GHOSTS;
    if constexpr ((D == Dim::_3D) || (D == Dim::_2D)) {
      nx2 = res[1] + 2 * N_GHOSTS;
    }
    if constexpr (D == Dim::_3D) {
      nx3 = res[2] + 2 * N_GHOSTS;
    }

    if constexpr (D == Dim::_1D) {
      em   = ndfield_t<Dim::_1D, 6> { "EM", nx1 };
      bckp = ndfield_t<Dim::_1D, 6> { "BCKP", nx1 };
      cur  = ndfield_t<Dim::_1D, 3> { "J", nx1 };
      buff = ndfield_t<Dim::_1D, 3> { "BUFF", nx1 };
      if constexpr (::traits::engine::DefinesAuxFields<S>) {
        aux = ndfield_t<Dim::_1D, 6> { "AUX", nx1 };
      }
      if constexpr (::traits::engine::DefinesEM0Fields<S>) {
        em0 = ndfield_t<Dim::_1D, 6> { "EM0", nx1 };
      }
      if constexpr (::traits::engine::DefinesCur0Fields<S>) {
        cur0 = ndfield_t<Dim::_1D, 3> { "CUR0", nx1 };
      }
    } else if constexpr (D == Dim::_2D) {
      em   = ndfield_t<Dim::_2D, 6> { "EM", nx1, nx2 };
      bckp = ndfield_t<Dim::_2D, 6> { "BCKP", nx1, nx2 };
      cur  = ndfield_t<Dim::_2D, 3> { "J", nx1, nx2 };
      buff = ndfield_t<Dim::_2D, 3> { "BUFF", nx1, nx2 };
      if constexpr (::traits::engine::DefinesAuxFields<S>) {
        aux = ndfield_t<Dim::_2D, 6> { "AUX", nx1, nx2 };
      }
      if constexpr (::traits::engine::DefinesEM0Fields<S>) {
        em0 = ndfield_t<Dim::_2D, 6> { "EM0", nx1, nx2 };
      }
      if constexpr (::traits::engine::DefinesCur0Fields<S>) {
        cur0 = ndfield_t<Dim::_2D, 3> { "CUR0", nx1, nx2 };
      }
    } else if constexpr (D == Dim::_3D) {
      em   = ndfield_t<Dim::_3D, 6> { "EM", nx1, nx2, nx3 };
      bckp = ndfield_t<Dim::_3D, 6> { "BCKP", nx1, nx2, nx3 };
      cur  = ndfield_t<Dim::_3D, 3> { "J", nx1, nx2, nx3 };
      buff = ndfield_t<Dim::_3D, 3> { "BUFF", nx1, nx2, nx3 };
      if constexpr (::traits::engine::DefinesAuxFields<S>) {
        aux = ndfield_t<Dim::_3D, 6> { "AUX", nx1, nx2, nx3 };
      }
      if constexpr (::traits::engine::DefinesEM0Fields<S>) {
        em0 = ndfield_t<Dim::_3D, 6> { "EM0", nx1, nx2, nx3 };
      }
      if constexpr (::traits::engine::DefinesCur0Fields<S>) {
        cur0 = ndfield_t<Dim::_3D, 3> { "CUR0", nx1, nx2, nx3 };
      }
    }
  }

  // NOLINTBEGIN(bugprone-macro-parentheses)
#define FIELDS_CONSTRUCTOR(D, S) template struct Fields<D, S>;

  NTT_FOREACH_SPECIALIZATION_FIELDS(FIELDS_CONSTRUCTOR)
#undef FIELDS_CONSTRUCTOR
  // NOLINTEND(bugprone-macro-parentheses)

} // namespace ntt

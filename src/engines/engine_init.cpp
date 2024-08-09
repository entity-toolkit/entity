#include "enums.h"
#include "global.h"

#include "arch/traits.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "archetypes/field_setter.h"
#include "engines/engine.hpp"

#include <Kokkos_Core.hpp>

namespace ntt {

  template <SimEngine::type S, class M>
  void Engine<S, M>::init() {
    if constexpr (pgen_is_ok) {
#if defined(OUTPUT_ENABLED)
      m_metadomain.InitWriter(m_params);
#endif
      logger::Checkpoint("Initializing Engine", HERE);
      if constexpr (
        traits::has_member<traits::pgen::init_flds_t, user::PGen<S, M>>::value) {
        logger::Checkpoint("Initializing fields from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          Kokkos::parallel_for(
            "InitFields",
            loc_dom.mesh.rangeActiveCells(),
            arch::SetEMFields_kernel<decltype(m_pgen.init_flds), S, M> {
              loc_dom.fields.em,
              m_pgen.init_flds,
              loc_dom.mesh.metric });
        });
      }
      if constexpr (
        traits::has_member<traits::pgen::init_prtls_t, user::PGen<S, M>>::value) {
        logger::Checkpoint("Initializing particles from problem generator", HERE);
        m_metadomain.runOnLocalDomains([&](auto& loc_dom) {
          m_pgen.InitPrtls(loc_dom);
        });
      }
    }
  }

  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template class Engine<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
} // namespace ntt
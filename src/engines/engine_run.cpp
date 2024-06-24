#include "enums.h"

#include "arch/traits.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/engine.hpp"

namespace ntt {

  template <SimEngine::type S, class M>
  void Engine<S, M>::run() {
    if constexpr (pgen_is_ok) {
      init();

      auto timers = timer::Timers {
        { "FieldSolver",
         "CurrentFiltering", "CurrentDeposit",
         "ParticlePusher", "FieldBoundaries",
         "ParticleBoundaries", "Communications",
         "Injector", "Sorting",
         "Custom", "Output" },
        []() {
          Kokkos::fence();
         },
        m_params.get<bool>("diagnostics.blocking_timers")
      };
      const auto diag_interval = m_params.get<std::size_t>(
        "diagnostics.interval");

      auto       time_history  = pbar::DurationHistory { 1000 };
      const auto sort_interval = m_params.template get<std::size_t>(
        "particles.sort_interval");

      // main algorithm loop
      while (step < max_steps) {
        // run the engine-dependent algorithm step
        m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
          step_forward(timers, dom);
        });
        // poststep (if defined)
        if constexpr (
          traits::has_method<traits::pgen::custom_poststep_t, decltype(m_pgen)>::value) {
          timers.start("Custom");
          m_metadomain.runOnLocalDomains([&timers, this](auto& dom) {
            m_pgen.CustomPostStep(step, time, dom);
          });
          timers.stop("Custom");
        }
        auto print_sorting = (sort_interval > 0 and step % sort_interval == 0);

        // advance time & timestep
        ++step;
        time += dt;

        auto print_output = false;
#if defined(OUTPUT_ENABLED)
        timers.start("Output");
        print_output = m_metadomain.Write(m_params, step, time);
        timers.stop("Output");
#endif

        // advance time_history
        time_history.tick();
        // print final timestep report
        if (diag_interval > 0 and step % diag_interval == 0) {
          print_step_report(timers, time_history, print_output, print_sorting);
        }
        timers.resetAll();
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

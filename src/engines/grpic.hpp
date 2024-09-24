/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"

#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "engines/engine.hpp"

#include "kernels/ampere_gr.hpp"
#include "kernels/aux_fields_gr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/faraday_gr.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_pusher_gr.hpp"

#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>
#include <utility>

namespace ntt {

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using Engine<SimEngine::GRPIC, M>::m_params;
    using Engine<SimEngine::GRPIC, M>::m_metadomain;


  public:
    static constexpr auto S { SimEngine::SRPIC };

    GRPICEngine(SimulationParams& params)
      : Engine<SimEngine::GRPIC, M> { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers&, Domain<SimEngine::GRPIC, M>&) override {}
  };

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H

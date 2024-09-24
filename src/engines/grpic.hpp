/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - grpic.cpp
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
    using base_t   = Engine<SimEngine::GRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::GRPIC, M>;
    using domain_t = Domain<SimEngine::GRPIC, M>;
    // constexprs
    using base_t::pgen_is_ok;
    // contents
    using base_t::m_metadomain;
    using base_t::m_params;
    using base_t::m_pgen;
    // methods
    using base_t::init;
    // variables
    using base_t::dt;
    using base_t::max_steps;
    using base_t::runtime;
    using base_t::step;
    using base_t::time;

  public:
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine(SimulationParams& params) : base_t { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {}
  };

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H

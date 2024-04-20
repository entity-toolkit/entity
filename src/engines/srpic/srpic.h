/**
 * @file engines/srpic/srpic.h
 * @brief Simulation engien class which specialized on SRPIC
 * @implements
 *   - ntt::SRPICEngine<> : ntt::Engine<>
 * @depends:
 *   - enums.h
 *   - arch/traits.h
 *   - arch/kokkos_aliases.h
 *   - utils/log.h
 *   - utils/timer.h
 *   - utils/numeric.h
 *   - engines/engine.h
 *   - framework/domain/domain.h
 *   - framework/parameters.h
 *   - metrics/minkowski.h
 *   - metrics/qspherical.h
 *   - metrics/spherical.h
 *   - pgen.hpp
 *   - kernels/particle_pusher_sr.hpp
 *   - kernels/faraday_mink.hpp
 *   - kernels/faraday_sr.hpp
 *   - kernels/ampere_mink.hpp
 *   - kernels/ampere_sr.hpp
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_SRPIC_SRPIC_H
#define ENGINES_SRPIC_SRPIC_H

#include "enums.h"

#include "utils/timer.h"

#include "engines/engine.h"
#include "framework/domain/domain.h"

#include <utility>

namespace ntt {

  template <class M>
  class SRPICEngine : public Engine<SimEngine::SRPIC, M> {
    // constexprs
    using Engine<SimEngine::SRPIC, M>::pgen_is_ok;
    // contents
    using Engine<SimEngine::SRPIC, M>::m_params;
    using Engine<SimEngine::SRPIC, M>::m_pgen;
    // methods
    using Engine<SimEngine::SRPIC, M>::init;
    // variables
    using Engine<SimEngine::SRPIC, M>::runtime;
    using Engine<SimEngine::SRPIC, M>::dt;
    using Engine<SimEngine::SRPIC, M>::max_steps;
    using Engine<SimEngine::SRPIC, M>::time;
    using Engine<SimEngine::SRPIC, M>::step;
    using domain_t = Domain<SimEngine::SRPIC, M>;

  public:
    SRPICEngine(SimulationParams& params) :
      Engine<SimEngine::SRPIC, M> { params } {}

    ~SRPICEngine() = default;

    void step_forward(timer::Timers&, domain_t&) override;

    /* algorithm substeps --------------------------------------------------- */
    void Faraday(domain_t&, real_t = ONE);
    void Ampere(domain_t&, real_t = ONE);
    void ParticlePush(domain_t&);
    void CurrentsDeposit(domain_t&);
    void CurrentsAmpere(domain_t&);
  };

} // namespace ntt

#endif // ENGINES_SRPIC_SRPIC_H
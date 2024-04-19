/**
 * @file engines/grpic/grpic.h
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @depends:
 *   - enums.h
 *   - engines/engine.h
 *   - utils/timer.h
 *   - metrics/kerr_schild.h
 *   - metrics/kerr_schild_0.h
 *   - metrics/qkerr_schild.h
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"

#include "utils/timer.h"

#include "engines/engine.h"

namespace ntt {

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using Engine<SimEngine::GRPIC, M>::m_params;
    using Engine<SimEngine::GRPIC, M>::m_metadomain;

  public:
    GRPICEngine(SimulationParams& params) :
      Engine<SimEngine::GRPIC, M> { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers&) override;
  };

} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H
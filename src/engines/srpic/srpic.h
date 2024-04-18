/**
 * @file engines/srpic/srpic.h
 * @brief Simulation engien class which specialized on SRPIC
 * @implements
 *   - ntt::SRPICEngine<> : ntt::Engine<>
 * @depends:
 *   - enums.h
 *   - arch/traits.h
 *   - engines/engine.h
 *   - metrics/minkowski.h
 *   - metrics/qspherical.h
 *   - metrics/spherical.h
 *   - archetypes/field_setter.h
 *   - pgen.hpp
 * @cpp:
 *   - srpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_SRPIC_SRPIC_H
#define ENGINES_SRPIC_SRPIC_H

#include "enums.h"

#include "engines/engine.h"

#include <utility>

namespace ntt {

  template <class M>
  class SRPICEngine : public Engine<SimEngine::SRPIC, M> {
    using Engine<SimEngine::SRPIC, M>::m_params;
    using Engine<SimEngine::SRPIC, M>::m_metadomain;
    using Engine<SimEngine::SRPIC, M>::m_pgen;

  public:
    SRPICEngine(SimulationParams& params) :
      Engine<SimEngine::SRPIC, M> { params } {}

    ~SRPICEngine() = default;

    void init();
    void step_forward();

    void run();

    //  template <typename Func, typename... Args>
    // void run_on_local_domains(Func func, Args&&... args) {
    //  for (auto& ld : m_metadomain.local_domains()) {
    //     func(std::forward<Args>(args)...);
    //   }
    //  }
  };

} // namespace ntt

#endif // ENGINES_SRPIC_SRPIC_H
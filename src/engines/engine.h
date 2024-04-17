/**
 * @file engines/engine.h
 * @brief Base simulation class which just initializes the metadomain
 * @implements
 *   - ntt::Engine<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - pgen.hpp
 *   - arch/traits.h
 *   - arch/directions.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/formatting.h
 *   - framework/containers/fields.h
 *   - framework/containers/particles.h
 *   - framework/containers/species.h
 *   - framework/logistics/metadomain.h
 *   - framework/parameters.h
 *   - metrics/kerr_schild.h
 *   - metrics/kerr_schild_0.h
 *   - metrics/minkowski.h
 *   - metrics/qkerr_schild.h
 *   - metrics/qspherical.h
 *   - metrics/spherical.h
 * @cpp:
 *   - engine.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_ENGINE_H
#define ENGINES_ENGINE_H

#include "enums.h"
#include "global.h"

#include "arch/traits.h"
#include "utils/error.h"
#include "utils/log.h"

#include "framework/containers/fields.h"
#include "framework/containers/particles.h"
#include "framework/containers/species.h"
#include "framework/logistics/metadomain.h"
#include "framework/parameters.h"

#include "pgen.hpp"

#include <map>
#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  class Engine {
    static_assert(M::is_metric, "template arg for Engine class has to be a metric");
    static_assert(user::PGen<S, M>::is_pgen, "unrecognized problem generator");

  protected:
    SimulationParams& m_params;
    Metadomain<S, M>  m_metadomain;
    user::PGen<S, M>  m_pgen;

  public:
    static constexpr Dimension D { M::Dim };
    static constexpr bool      is_engine { true };

    Engine(SimulationParams& params) :
      m_params { params },
      m_metadomain {
        params.get<unsigned int>("simulation.domain.number"),
        params.get<std::vector<int>>("simulation.domain.decomposition"),
        params.get<std::vector<std::size_t>>("grid.resolution"),
        params.get<boundaries_t<real_t>>("grid.extent"),
        params.get<boundaries_t<FldsBC>>("grid.boundaries.fields"),
        params.get<boundaries_t<PrtlBC>>("grid.boundaries.particles"),
        params.get<std::map<std::string, real_t>>("grid.metric.params"),
        params.get<std::vector<ParticleSpecies>>("particles.species")
      },
      m_pgen { m_params } {
      raise::ErrorIf(
        not traits::check_compatibility<S>::value(m_pgen.engines),
        "Problem generator is not compatible with the picked engine",
        HERE);
      raise::ErrorIf(
        not traits::check_compatibility<M::MetricType>::value(m_pgen.metrics),
        "Problem generator is not compatible with the metric picked",
        HERE);
      raise::ErrorIf(
        not traits::check_compatibility<M::Dim>::value(m_pgen.dimensions),
        "Problem generator is not compatible with the dimension picked",
        HERE);
      print_report();
    }

    ~Engine() = default;

    void print_report() const;
  };

} // namespace ntt

#endif // ENGINES_ENGINE_H
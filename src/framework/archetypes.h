#ifndef FRAMEWORK_ARCHETYPES_H
#define FRAMEWORK_ARCHETYPES_H

#include "wrapper.h"
#include "sim_params.h"
#include "meshblock.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  template <Dimension D, SimulationType S>
  struct PGen {
    virtual inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void
    UserBCFields(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void
    UserDriveParticles(const real_t&, const SimulationParams&, Meshblock<D, S>&) {}

#ifdef NTTINY_ENABLED
    virtual inline void
    UserInitBuffers_nttiny(const SimulationParams&,
                           const Meshblock<D, S>&,
                           std::map<std::string, nttiny::ScrollingBuffer>&) {}
    virtual inline void
    UserSetBuffers_nttiny(const real_t&,
                          const SimulationParams&,
                          const Meshblock<D, S>&,
                          std::map<std::string, nttiny::ScrollingBuffer>&) {}
#endif
  };

  template <Dimension D, SimulationType S>
  struct EnergyDistribution {
    explicit EnergyDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params {params}, m_mblock {mblock} {}

  private:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationType S>
  struct ColdDist : public EnergyDistribution<D, S> {
    explicit ColdDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationType S>
  struct SpatialDistribution {
    explicit SpatialDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params {params}, m_mblock {mblock} {}

  private:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationType S>
  struct UniformDist : public SpatialDistribution<D, S> {
    explicit UniformDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(const coord_t<D>&) const { return ONE; }
  };

  template <Dimension D, SimulationType S>
  struct InjectionCriterion {
    explicit InjectionCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params {params}, m_mblock {mblock} {}

  private:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationType S>
  struct NoCriterion : public InjectionCriterion<D, S> {
    explicit NoCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>&) const { return true; }
  };

} // namespace ntt

#endif // FRAMEWORK_ARCHETYPES_H
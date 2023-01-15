#ifndef FRAMEWORK_ARCHETYPES_H
#define FRAMEWORK_ARCHETYPES_H

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  /* -------------------------------------------------------------------------- */
  /*                              Master pgen class                             */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct PGen {
    virtual inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) {}
    virtual inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) {}

    virtual inline void UserDriveFields(const real_t&,
                                        const SimulationParams&,
                                        Meshblock<D, S>&) {}
    virtual inline void UserDriveParticles(const real_t&,
                                           const SimulationParams&,
                                           Meshblock<D, S>&) {}

#ifdef NTTINY_ENABLED
    virtual inline void UserInitBuffers_nttiny(const SimulationParams&,
                                               const Meshblock<D, S>&,
                                               std::map<std::string, nttiny::ScrollingBuffer>&) {
    }
    virtual inline void UserSetBuffers_nttiny(const real_t&,
                                              const SimulationParams&,
                                              const Meshblock<D, S>&,
                                              std::map<std::string, nttiny::ScrollingBuffer>&) {
    }
#endif
  };

  /* -------------------------------------------------------------------------- */
  /*                             Target field class                             */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct TargetFields {
    TargetFields(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual real_t operator()(const em&, const coord_t<D>&) const {
      return ZERO;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  /* -------------------------------------------------------------------------- */
  /*                             Energy distribution                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct EnergyDistribution {
    EnergyDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual void operator()(const coord_t<D>&,
                                   vec_t<Dim3>& v,
                                   const int&   species = 0) const {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct ColdDist : public EnergyDistribution<D, S> {
    ColdDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species = 0) const override {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                            Spatial distribution                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct SpatialDistribution {
    SpatialDistribution(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

    Inline virtual auto operator()(const coord_t<D>&) const -> real_t {
      return ONE;
    }

  private:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct UniformDist : public SpatialDistribution<D, S> {
    UniformDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline auto operator()(const coord_t<D>&) const -> real_t override {
      return ONE;
    }
  };

  /* -------------------------------------------------------------------------- */
  /*                             Injection criterion                            */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, SimulationEngine S>
  struct InjectionCriterion {
    InjectionCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : m_params { params }, m_mblock { mblock } {}

  protected:
    SimulationParams m_params;
    Meshblock<D, S>  m_mblock;
  };

  template <Dimension D, SimulationEngine S>
  struct NoCriterion : public InjectionCriterion<D, S> {
    NoCriterion(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>&) const {
      return true;
    }
  };

}    // namespace ntt

#endif    // FRAMEWORK_ARCHETYPES_H
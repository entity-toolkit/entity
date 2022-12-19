#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"
#include "archetypes.hpp"
#include "injector.hpp"
#include "sim_params.h"
#include "meshblock.h"
#include "particle_macros.h"

#ifdef NTTINY_ENABLED
#  include "nttiny/api.h"
#endif

#include <map>

namespace ntt {

  template <Dimension D, SimulationType S>
  struct HotDist : public EnergyDistribution<D, S> {
    HotDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = ONE;
      v[1] = ONE;
      v[2] = ONE;
    }
  };

  template <Dimension D, SimulationType S>
  struct CoshDist : public SpatialDistribution<D, S> {
    explicit CoshDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(coord_t<D> x_ph) const { return 1.0 / math::cosh(x_ph[0] / 0.2); }
  };

  template <Dimension D, SimulationType S>
  struct EgtrBCrit : public InjectionCriterion<D, S> {
    explicit EgtrBCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>& xi) const;
  };

  template <>
  Inline bool EgtrBCrit<Dim2, TypePIC>::operator()(const coord_t<Dim2>& xi) const {
    this->m_mblock.em((int)xi[0], (int)xi[1], em::ex1);
    return true;
  }

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
  }; // struct ProblemGenerator

  template <>
  inline void
  ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
                                                     Meshblock<Dim2, TypePIC>& mblock) {
    auto nppc_per_spec = (real_t)(params.ppc0()) * HALF;
    InjectUniform<Dim2, TypePIC, HotDist>(params, mblock, {1, 2}, nppc_per_spec);

    InjectUniform<Dim2, TypePIC, HotDist>(
      params, mblock, {1, 2}, nppc_per_spec, {-0.5, 0.5, -0.1, 0.2});

    InjectInVolume<Dim2, TypePIC, HotDist, CoshDist>(params, mblock, {1, 2}, nppc_per_spec);

    InjectInVolume<Dim2, TypePIC, HotDist, UniformDist, EgtrBCrit>(
      params, mblock, {1, 2}, nppc_per_spec);
  }

} // namespace ntt

#endif

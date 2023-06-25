#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

#include <map>

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct MyDist : public EnergyDistribution<D, S> {
    MyDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = ONE;
      v[1] = ONE;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationEngine S>
  struct MyDist2 : public EnergyDistribution<D, S> {
    MyDist2(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = -ONE;
      v[1] = -ONE;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationEngine S>
  struct CoshDist : public SpatialDistribution<D, S> {
    explicit CoshDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(coord_t<D> x_ph) const {
      return 1.0 / math::cosh(x_ph[0] / 0.2);
    }
  };

  template <Dimension D, SimulationEngine S>
  struct ExpDist : public SpatialDistribution<D, S> {
    explicit ExpDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(const coord_t<D>&) const;
  };

  template <>
  Inline real_t ExpDist<Dim1, PICEngine>::operator()(const coord_t<Dim1>&) const {}

  template <>
  Inline real_t ExpDist<Dim2, PICEngine>::operator()(const coord_t<Dim2>& x_ph) const {
    return math::exp(-(SQR(x_ph[0]) + SQR(x_ph[1])) / SQR(0.05));
  }

  template <>
  Inline real_t ExpDist<Dim3, PICEngine>::operator()(const coord_t<Dim3>&) const {}

  template <Dimension D, SimulationEngine S>
  struct EgtrBCrit : public InjectionCriterion<D, S> {
    explicit EgtrBCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>& xi) const;
  };

  template <>
  Inline bool EgtrBCrit<Dim1, PICEngine>::operator()(const coord_t<Dim1>&) const {}

  template <>
  Inline bool EgtrBCrit<Dim2, PICEngine>::operator()(const coord_t<Dim2>& xi) const {
    this->m_mblock.em((int)xi[0], (int)xi[1], em::ex1);
    return true;
  }

  template <>
  Inline bool EgtrBCrit<Dim3, PICEngine>::operator()(const coord_t<Dim3>&) const {}

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&           time,
                                   const SimulationParams& params,
                                   Meshblock<D, S>&        mblock) override {
      auto nppc_per_spec = (real_t)(params.ppc0()) * HALF;
      InjectInVolume<D, S, MyDist, ExpDist>(params, mblock, { 1, 2 }, nppc_per_spec, {}, time);
    }
  };    // struct ProblemGenerator

}    // namespace ntt

#endif

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
  struct MyDist : public EnergyDistribution<D, S> {
    MyDist(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = ONE;
      v[1] = ONE;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationType S>
  struct MyDist2 : public EnergyDistribution<D, S> {
    MyDist2(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock) {}
    Inline void operator()(const coord_t<D>&, vec_t<Dim3>& v) const override {
      v[0] = -ONE;
      v[1] = -ONE;
      v[2] = ZERO;
    }
  };

  template <Dimension D, SimulationType S>
  struct CoshDist : public SpatialDistribution<D, S> {
    explicit CoshDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(coord_t<D> x_ph) const { return 1.0 / math::cosh(x_ph[0] / 0.2); }
  };

  template <Dimension D, SimulationType S>
  struct ExpDist : public SpatialDistribution<D, S> {
    explicit ExpDist(const SimulationParams& params, Meshblock<D, S>& mblock)
      : SpatialDistribution<D, S>(params, mblock) {}
    Inline real_t operator()(const coord_t<D>&) const;
  };

  template <>
  Inline real_t ExpDist<Dim1, TypePIC>::operator()(const coord_t<Dim1>&) const {}

  template <>
  Inline real_t ExpDist<Dim2, TypePIC>::operator()(const coord_t<Dim2>& x_ph) const {
    return math::exp(-(SQR(x_ph[0]) + SQR(x_ph[1])) / SQR(0.05));
  }

  template <>
  Inline real_t ExpDist<Dim3, TypePIC>::operator()(const coord_t<Dim3>&) const {}

  template <Dimension D, SimulationType S>
  struct EgtrBCrit : public InjectionCriterion<D, S> {
    explicit EgtrBCrit(const SimulationParams& params, Meshblock<D, S>& mblock)
      : InjectionCriterion<D, S>(params, mblock) {}
    Inline bool operator()(const coord_t<D>& xi) const;
  };

  template <>
  Inline bool EgtrBCrit<Dim1, TypePIC>::operator()(const coord_t<Dim1>&) const {}

  template <>
  Inline bool EgtrBCrit<Dim2, TypePIC>::operator()(const coord_t<Dim2>& xi) const {
    this->m_mblock.em((int)xi[0], (int)xi[1], em::ex1);
    return true;
  }

  template <>
  Inline bool EgtrBCrit<Dim3, TypePIC>::operator()(const coord_t<Dim3>&) const {}

  template <Dimension D, SimulationType S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}
    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override {}
    inline void UserDriveParticles(const real_t&           time,
                                   const SimulationParams& params,
                                   Meshblock<D, S>&        mblock) override {
      auto nppc_per_spec = (real_t)(params.ppc0()) * HALF;
      InjectInVolume<D, S, MyDist, ExpDist>(params, mblock, {1, 2}, nppc_per_spec, {}, time);
    }
  }; // struct ProblemGenerator

  // template <>
  // inline void
  // ProblemGenerator<Dim2, TypePIC>::UserInitParticles(const SimulationParams&   params,
  //                                                    Meshblock<Dim2, TypePIC>& mblock) {
  //   auto nppc_per_spec = (real_t)(params.ppc0()) * HALF;
  //   InjectUniform<Dim2, TypePIC, MyDist>(params, mblock, {1, 2}, nppc_per_spec);

  //   InjectUniform<Dim2, TypePIC, MyDist2>(
  //     params, mblock, {1, 2}, nppc_per_spec, {-0.5, 0.5, -0.1, 0.2});

  //   InjectInVolume<Dim2, TypePIC, ColdDist, CoshDist>(params, mblock, {1, 2},
  //   nppc_per_spec);

  //   // InjectInVolume<Dim2, TypePIC, HotDist, UniformDist, EgtrBCrit>(
  //   //   params, mblock, {1, 2}, nppc_per_spec);
  // }

} // namespace ntt

#endif

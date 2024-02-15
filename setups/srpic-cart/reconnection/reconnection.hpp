#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams& params) :
      cs_width { params.get<real_t>("problem", "cs_width") },
      cs_overdens { params.get<real_t>("problem", "cs_overdens") } {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
    }

    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override;

  private:
    const real_t cs_width, cs_overdens;
  }; // struct ProblemGenerator

  Inline void reconnectionField(const coord_t<Dim2>& x_ph,
                                vec_t<Dim3>&         e_out,
                                vec_t<Dim3>&         b_out,
                                real_t               csW,
                                real_t               c1,
                                real_t               c2) {
    b_out[1] = math::tanh((x_ph[0] - c1) / csW) -
               math::tanh((x_ph[0] - c2) / csW) - ONE;
  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    real_t Xmin = mblock.metric.x1_min;
    real_t Xmax = mblock.metric.x1_max;
    real_t sX   = Xmax - Xmin;
    real_t cX1  = Xmin + 0.25 * sX;
    real_t cX2  = Xmin + 0.75 * sX;
    Kokkos::parallel_for(
      "UserInitFields",
      mblock.rangeActiveCells(),
      ClassLambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, reconnectionField, cs_width, cX1, cX2);
      });
  }

  template <Dimension D, SimulationEngine S>
  struct CS_Dist : public EnergyDistribution<D, S> {
    CS_Dist(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      drift_beta { math::sqrt(params.sigma0()) * params.skindepth0() /
                   (params.get<real_t>("problem", "cs_overdens") *
                    params.get<real_t>("problem", "cs_width")) },
      drift_gamma { ONE / math::sqrt(ONE - SQR(drift_beta)) },
      temperature { HALF * params.sigma0() /
                    params.get<real_t>("problem", "cs_overdens") },
      xmid { (mblock.metric.x1_max - mblock.metric.x1_min) * HALF } {}

    Inline void operator()(const coord_t<D>& x_Ph,
                           vec_t<Dim3>&      v,
                           const int&        species) const override {
      maxwellian(v, temperature);
      const real_t sign = ((species % 2 == 0) ? ONE : -ONE) *
                          ((x_Ph[0] < xmid) ? ONE : -ONE);
      v[2] += sign * drift_beta * drift_gamma;
    }

  private:
    const ntt::Maxwellian<D, S> maxwellian;
    const real_t                temperature, drift_beta, drift_gamma, xmid;
  };

  template <Dimension D, SimulationEngine S>
  struct Cosh_Prof : public SpatialDistribution<D, S> {
    explicit Cosh_Prof(const SimulationParams& params, Meshblock<D, S>& mblock) :
      SpatialDistribution<D, S>(params, mblock),
      cs_width { params.get<real_t>("problem", "cs_width") },
      cX1 { mblock.metric.x1_min +
            (real_t)0.25 * (mblock.metric.x1_max - mblock.metric.x1_min) },
      cX2 { mblock.metric.x1_min +
            (real_t)0.75 * (mblock.metric.x1_max - mblock.metric.x1_min) } {}

    Inline real_t operator()(const coord_t<D>& x_Ph) const {
      return ONE / SQR(math::cosh((x_Ph[0] - cX1) / cs_width)) +
             ONE / SQR(math::cosh((x_Ph[0] - cX2) / cs_width));
    }

  private:
    const real_t cs_width, cX1, cX2;
  };

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitParticles(
    const SimulationParams& params,
    Meshblock<D, S>&        mblock) {
    InjectUniform<D, S>(params, mblock, { 1, 2 }, params.ppc0() * HALF);
    InjectInVolume<D, S, CS_Dist, Cosh_Prof>(params,
                                             mblock,
                                             { 1, 2 },
                                             params.ppc0() * HALF * cs_overdens);
  }
} // namespace ntt

#endif
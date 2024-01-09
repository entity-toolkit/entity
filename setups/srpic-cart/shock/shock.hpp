#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct WeibelInit : public EnergyDistribution<D, S> {
    WeibelInit(const SimulationParams& params, const Meshblock<D, S>& mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      drift_p { params.get<real_t>("problem", "drift_p", 10.0) },
      temp_p { params.get<real_t>("problem", "temperature_p", 0.0) } {}

    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      if (species == 1) {
        maxwellian(v, temp_p, drift_p, -dir::x);
      } else if (species == 2) {
        maxwellian(v, temp_p, drift_p, -dir::x);
      }
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           drift_p,temp_p;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    inline ProblemGenerator(const SimulationParams&) {}

    inline void UserInitParticles(const SimulationParams&, Meshblock<D, S>&) override;

    inline void UserDriveFields(const real_t&,
                                const SimulationParams&,
                                Meshblock<D, S>&) override {}

    inline void UserDriveParticles(const int&, 
                                   const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

  };

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveFields(
    const real_t&                      time,
    const SimulationParams&            params,
    Meshblock<Dim2, PICEngine>&        mblock) {
    // const auto i1_min = 0;
    // const auto i1_max = N_GHOSTS;
    // const auto ic     = N_GHOSTS + 100;

    // Kokkos::parallel_for(
    // "UserDriveFields_bc_left",
    // CreateRangePolicy<Dim2>({ i1_min, mblock.i2_min() },
    //                         { i1_max, mblock.i2_max() }),
    // ClassLambda(index_t i1, index_t i2) {
    //   mblock.em(ic - i1,     i2, em::ex1) =  mblock.em(ic + i1 + 1, i2, em::ex1);
    //   mblock.em(ic - i1 - 1, i2, em::ex2) = -mblock.em(ic + i1    , i2, em::ex2);
    //   mblock.em(ic - i1 - 1, i2, em::ex3) = -mblock.em(ic + i1    , i2, em::ex3);

    //   mblock.em(ic - i1 - 1, i2, em::bx1) = -mblock.em(ic + i1    , i2, em::bx1);
    //   mblock.em(ic - i1,     i2, em::bx2) =  mblock.em(ic + i1 + 1, i2, em::bx2);
    //   mblock.em(ic - i1,     i2, em::bx3) =  mblock.em(ic + i1 + 1, i2, em::bx3);
    // }
    // );

    // Kokkos::parallel_for(
    // "UserDriveFields_bc_right",
    // CreateRangePolicy<Dim2>({ mblock.i1_max()-N_GHOSTS-50, mblock.i2_min() },
    //                         { mblock.i1_max()         , mblock.i2_max() }),
    // ClassLambda(index_t i1, index_t i2) {
    //   mblock.em(i1, i2, em::ex1) = ZERO;
    //   mblock.em(i1, i2, em::ex2) = ZERO;
    //   mblock.em(i1, i2, em::ex3) = ZERO;

    //   mblock.em(i1, i2, em::bx1) = ZERO;
    //   mblock.em(i1, i2, em::bx2) = ZERO;
    //   mblock.em(i1, i2, em::bx3) = ZERO;

    //   mblock.cur(i1, i2, cur::jx1) = ZERO;
    //   mblock.cur(i1, i2, cur::jx2) = ZERO;
    //   mblock.cur(i1, i2, cur::jx3) = ZERO;
    // }
    // );

    // Kokkos::parallel_for(
    // "UserDriveFields_bc_right",
    // CreateRangePolicy<Dim2>({ mblock.i1_max()-N_GHOSTS-50, mblock.i2_min() },
    //                         { mblock.i1_max()         , mblock.i2_max() }),
    // ClassLambda(index_t i1, index_t i2) {
    //   mblock.em(i1, i2, em::ex1) = ZERO;
    //   mblock.em(i1, i2, em::ex2) = ZERO;
    //   mblock.em(i1, i2, em::ex3) = ZERO;

    //   mblock.em(i1, i2, em::bx1) = ZERO;
    //   mblock.em(i1, i2, em::bx2) = ZERO;
    //   mblock.em(i1, i2, em::bx3) = ZERO;

    //   mblock.cur(i1, i2, cur::jx1) = ZERO;
    //   mblock.cur(i1, i2, cur::jx2) = ZERO;
    //   mblock.cur(i1, i2, cur::jx3) = ZERO;
    // }
    // );

    // Kokkos::parallel_for(
    // "UserDriveFields_bc_right2",
    // CreateRangePolicy<Dim2>({ mblock.i1_min(), mblock.i2_min() },
    //                         { mblock.i1_max(), mblock.i2_max() }),
    // ClassLambda(index_t i1, index_t i2) {
    //   mblock.em(i1, i2, em::ex1) = ZERO;
    //   mblock.em(i1, i2, em::ex2) = ZERO;
    //   mblock.em(i1, i2, em::ex3) = ZERO;

    //   mblock.em(i1, i2, em::bx1) = ZERO;
    //   mblock.em(i1, i2, em::bx2) = ZERO;
    //   mblock.em(i1, i2, em::bx3) = ZERO;

    //   mblock.cur(i1, i2, cur::jx1) = ZERO;
    //   mblock.cur(i1, i2, cur::jx2) = ZERO;
    //   mblock.cur(i1, i2, cur::jx3) = ZERO;
    // }
    // );

  }

  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const int&                  step,
    const real_t&               time,
    const SimulationParams&     params,
    Meshblock<Dim2, PICEngine>& mblock) {
    const real_t drift_p  { params.get<real_t>("problem", "drift_p", 10.0) };
    const int    ninj     { params.get<int>(   "problem", "ninj",    100)  };
    const real_t x1_inj_0 { params.get<real_t>("problem", "x1_inj_0",20.0) };
    // params.dt() = (mblock.metric.x1_max - mblock.metric.x1_min) / (mblock.nx1 * vCC);
    for (auto& species : mblock.particles) {
      if (species.npart() == 0) {
          continue;
      }
      Kokkos::parallel_for(
        "reflective_bc",
        species.rangeActiveParticles(),
        Lambda( index_t p) {
          if (species.i1(p) < N_GHOSTS + 100) {
            species.i1(p)  = - species.i1(p) + 2 * (N_GHOSTS + 100) ;
            species.dx1(p) = ONE - species.dx1(p);
            species.ux1(p) = - species.ux1(p);
          }
        });
    }
    
    if ( step%ninj == 0 ) {
    // if ( (int)(time/vCC + 0.99)==0 ) {
      InjectInVolume<Dim2, PICEngine, WeibelInit>(
      params,
      mblock,
      { 1, 2 },
      params.ppc0(),
      // { mblock.metric.x1_max-100.*vCC*math::sqrt(ONE-ONE/SQR(drift_p)),mblock.metric.x1_max,mblock.metric.x2_min,mblock.metric.x2_max }
      { std::max(
        std::min(x1_inj_0 + mblock.timestep()*(real_t)(step),mblock.metric.x1_max) - mblock.timestep()*(real_t)ninj*math::sqrt(ONE-ONE/SQR(drift_p)),
        x1_inj_0
        ),
        std::min(x1_inj_0 + mblock.timestep()*(real_t)(step+ninj),mblock.metric.x1_max),
        mblock.metric.x2_min,
        mblock.metric.x2_max }
      );
    }

  }

  template <Dimension D, SimulationEngine S>
  inline void ProblemGenerator<D, S>::UserInitParticles(
    const SimulationParams& params,
    Meshblock<D, S>&        mblock) {
    // InjectUniform<D, S, WeibelInit>(params, mblock, { 1, 2 }, params.ppc0());
  }
} // namespace ntt

#endif

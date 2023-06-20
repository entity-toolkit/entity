#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

namespace ntt {

  /**
   * Define a structure which will initialize the particle energy distribution.
   * This is used below in the UserInitParticles function.
   */
  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : EnergyDistribution<D, S>(params, mblock),
        maxwellian { mblock },
        temperature { params.get<real_t>("problem", "temperature", 0.1) } {}
    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      /**
       * initializes the velocity distribution to a maxwellian
       * with a temperature read from the input (defaults to 0.1 me c^2 if not specified)
       * [TO BE MODIFIED]
       */
      maxwellian(v, temperature);

      /**
       * can also have different temperatures for different species:
       * ```
       *  if (species == 1) {
       *   maxwellian(v, temperature1);
       *  } else if (species == 2) {
       *   maxwellian(v, temperature2);
       *  }
       * ```
       *
       * can also have drift:
       * ```
       *  maxwellian(v, temperature, drift_four_velocity, dir::<direction>);
       * ```
       * for possible directions specified with dir::<...>, see mesh.h file
       */
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           temperature;
  };

  /**
   * Main problem generator class with all the required functions to define
   * the initial/boundary conditions and the source terms.
   */
  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
    // read additional parameters from the input file [default to 1 if not specified]
    inline ProblemGenerator(const SimulationParams& params)
      : nx1 { params.get<int>("problem", "nx1", 1) },
        nx2 { params.get<int>("problem", "nx2", 1) },
        nx3 { params.get<int>("problem", "nx3", 1) } {}
    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
      /**
       * this function we can define per each Dimension (Dim1, Dim2, Dim3) separately
       * so here we leave this empty as a default
       */
    }
    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {
      InjectUniform<D, S, ThermalBackground>(params, mblock, { 1, 2 }, params.ppc0() * 0.5);
      // ----------        ----------------                   ------   -------------------
      //      ^                     ^                           ^                   ^
      //      |                      \                          |                   |
      // (  function to        )    ( energy distribution )  ( species to inject ) /
      // ( inject particles    )                                                  /
      // (   uniformly         )                           ( average # of particles per cell )
      // ( in the whole domain )                           (        per each species         )
      //
      // Notes:
      //     - params.ppc0() is the same ppc0 number from the input file
      //     - 0.5 means we inject half of ppc0 per each species, so total ppc0 is preserved
      //     - species indices { 1, 2 } correspond to the indices defined in the input file
      //     - particles have to be initialized on top of each other, so we specify two species
      //       with opposing charges
    }

  private:
    // additional problem-specific parameters (i.e., wave numbers in x1, x2, x3 directions)
    const int nx1, nx2, nx3;
  };

  Inline void turbulent_fields_2d(const coord_t<Dim2>& x_ph,     // physical coordinate
                                  vec_t<Dim3>&         e_out,    // electric field [out]
                                  vec_t<Dim3>&         b_out,    // magnetic field [out]
                                  real_t               time,     // ... additional parameters
                                  real_t               sx1,      // ...
                                  real_t               sx2,      // ...
                                  int                  nx1,      // ...
                                  int                  nx2       // ...
  ) {
    // this is my silly understanding of how turbulent setup works (feel free to rewrite)
    const real_t kx1 = constant::TWO_PI * static_cast<real_t>(nx1) / sx1;
    const real_t kx2 = constant::TWO_PI * static_cast<real_t>(nx2) / sx2;
    const real_t ampl_x1
      = 0.1 * sx1 * static_cast<real_t>(nx2) / (sx2 * static_cast<real_t>(nx1));
    const real_t ampl_x2 = -0.1;

    // fields in physical units
    // (
    //    in physical units, when B = 1, and |q|/m = 1
    //    the larmor radius is equal to the fiducial value
    //    i.e., `m c^2 / |q| B == larmor0`
    // )
    e_out[0]             = 0.0;
    e_out[1]             = 0.0;
    e_out[2]             = 0.0;
    // some turbulent magnetic field goes here [TO BE MODIFIED]
    b_out[0]             = ampl_x1 * math::sin(kx1 * x_ph[0]) * math::cos(kx2 * x_ph[1]);
    b_out[1]             = ampl_x2 * math::cos(kx1 * x_ph[0]) * math::sin(kx2 * x_ph[1]);
    b_out[2]             = 1.0;
  }

  Inline void turbulent_fields_3d(const coord_t<Dim3>& x_ph,     // physical coordinate
                                  vec_t<Dim3>&         e_out,    // electric field [out]
                                  vec_t<Dim3>&         b_out,    // magnetic field [out]
                                  real_t               time,     // ... additional parameters
                                  int                  nx1,      // ...
                                  int                  nx2,      // ...
                                  int                  nx3       // ...
  ) {
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // some turbulent magnetic field goes here [TO BE MODIFIED]
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 1.0;
  }

  /**
   * Class that defines the force field applied to particles each time pusher is called.
   */
  template <Dimension D, SimulationEngine S>
  struct PgenForceField : public ForceField<D, S> {
    PgenForceField(const SimulationParams& params, const Meshblock<D, S>& mblock)
      : ForceField<D, S>(params, mblock),
        corr_time { params.get<real_t>("problem", "correlation_time") },
        sx1 { mblock.metric.x1_max - mblock.metric.x1_min } {}

    // force field components in physical units
    // arguments are:
    //    - time -- physical coordinate
    //    - x_ph -- 1D/2D/3D coordinate in physical units
    Inline auto x1(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
      // just as an example, implementing a weird sinusoidal force field in x1
      return math::sin(5.0 * constant::TWO_PI * x_ph[0] / sx1);
    }
    Inline auto x2(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
      return ZERO;
    }
    Inline auto x3(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
      return ZERO;
    }

  private:
    // additional parameters (i.e., correlation time)
    const real_t corr_time;
    const real_t sx1;
  };

  /**
   * Field initialization for 2D:
   */
  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time = this->time();
    const auto _nx1  = nx1;
    const auto _nx2  = nx2;
    const auto _sx1  = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _sx2  = mblock.metric.x2_max - mblock.metric.x2_min;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, turbulent_fields_2d, _time, _sx1, _sx2, _nx1, _nx2);
      });
    // the time of the simulation is in physical units (c = 1)
  }

  /**
   * Field initialization for 3D:
   */
  template <>
  inline void ProblemGenerator<Dim3, PICEngine>::UserInitFields(
    const SimulationParams&, Meshblock<Dim3, PICEngine>& mblock) {
    const auto _time = this->time();
    const auto _nx1  = nx1;
    const auto _nx2  = nx2;
    const auto _nx3  = nx3;
    Kokkos::parallel_for(
      "UserInitFields", mblock.rangeActiveCells(), Lambda(index_t i, index_t j, index_t k) {
        set_em_fields_3d(mblock, i, j, k, turbulent_fields_3d, _time, _nx1, _nx2, _nx3);
      });
  }
}    // namespace ntt

#endif
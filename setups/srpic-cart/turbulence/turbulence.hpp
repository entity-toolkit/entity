#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "wrapper.h"

#include "field_macros.h"
#include "sim_params.h"

#include "meshblock/meshblock.h"

#include "utils/archetypes.hpp"
#include "utils/injector.hpp"

#include <time.h>

#ifdef GUI_ENABLED
  #include "nttiny/api.h"
#endif

#define REAL (0)
#define IMAG (1)

namespace ntt {

  template <Dimension D, SimulationEngine S>
  struct ThermalBackground : public EnergyDistribution<D, S> {
    ThermalBackground(const SimulationParams& params,
                      const Meshblock<D, S>&  mblock) :
      EnergyDistribution<D, S>(params, mblock),
      maxwellian { mblock },
      temperature { params.get<real_t>("problem", "temperature", 0.1) } {}

    Inline void operator()(const coord_t<D>&,
                           vec_t<Dim3>& v,
                           const int&   species) const override {
      maxwellian(v, temperature);
    }

  private:
    const Maxwellian<D, S> maxwellian;
    const real_t           temperature;
  };

  template <Dimension D, SimulationEngine S>
  struct ProblemGenerator : public PGen<D, S> {
  public:
    real_t work_done { ZERO };

    // read additional parameters from the input file [default to 1 if not specified]
    inline ProblemGenerator(const SimulationParams& params) :
      nx1 { params.get<int>("problem", "nx1", 1) },
      nx2 { params.get<int>("problem", "nx2", 1) },
      nx3 { params.get<int>("problem", "nx3", 1) },
      sx1 { 2.0 },
      sx2 { 2.0 },
      sx3 { 2.0 },
      temperature { params.get<real_t>("problem", "temperature", 0.1) },
      machno { 0.1 },
      amplitudes { "DrivingModes", 6 } {}

    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}

    inline void UserInitFields(const SimulationParams&, Meshblock<D, S>&) override {
      /**
       * this function we can define per each Dimension (Dim1, Dim2, Dim3)
       * separately so here we leave this empty as a default
       */
    }

    inline void UserInitParticles(const SimulationParams& params,
                                  Meshblock<D, S>&        mblock) override {
      InjectUniform<D, S, ThermalBackground>(params,
                                             mblock,
                                             { 1, 2 },
                                             params.ppc0() * 0.5);
    }
#ifdef EXTERNAL_FORCE
    Inline auto ext_force_x1(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      auto   amplitudes_ = this->amplitudes;
      real_t k01         = ONE * constant::TWO_PI / sx1;
      real_t k02         = ZERO * constant::TWO_PI / sx2;
      real_t k03         = ZERO * constant::TWO_PI / sx3;
      real_t k04         = ONE;
      real_t k11         = ZERO * constant::TWO_PI / sx1;
      real_t k12         = ONE * constant::TWO_PI / sx2;
      real_t k13         = ZERO * constant::TWO_PI / sx3;
      real_t k14         = ONE;
      real_t k21         = ZERO * constant::TWO_PI / sx1;
      real_t k22         = ZERO * constant::TWO_PI / sx2;
      real_t k23         = ONE * constant::TWO_PI / sx3;
      real_t k24         = ONE;

      auto f_m1 = k14 * amplitudes_(0, REAL) *
                    cos(k11 * x_ph[0] + k12 * x_ph[1] + k13 * x_ph[2]) +
                  k14 * amplitudes_(0, IMAG) *
                    sin(k11 * x_ph[0] + k12 * x_ph[1] + k13 * x_ph[2]);
      auto f_m2 = k24 * amplitudes_(1, REAL) *
                    cos(k21 * x_ph[0] + k22 * x_ph[1] + k23 * x_ph[2]) +
                  k24 * amplitudes_(1, IMAG) *
                    sin(k21 * x_ph[0] + k22 * x_ph[1] + k23 * x_ph[2]);

      return f_m1 + f_m2;
    }

    Inline auto ext_force_x2(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {

      auto   amplitudes_ = this->amplitudes;
      real_t k01         = ONE * constant::TWO_PI / sx1;
      real_t k02         = ZERO * constant::TWO_PI / sx2;
      real_t k03         = ZERO * constant::TWO_PI / sx3;
      real_t k04         = ONE;
      real_t k11         = ZERO * constant::TWO_PI / sx1;
      real_t k12         = ONE * constant::TWO_PI / sx2;
      real_t k13         = ZERO * constant::TWO_PI / sx3;
      real_t k14         = ONE;
      real_t k21         = ZERO * constant::TWO_PI / sx1;
      real_t k22         = ZERO * constant::TWO_PI / sx2;
      real_t k23         = ONE * constant::TWO_PI / sx3;
      real_t k24         = ONE;

      auto f_m3 = k04 * amplitudes_(2, REAL) *
                    cos(k01 * x_ph[0] + k02 * x_ph[1] + k03 * x_ph[2]) +
                  k04 * amplitudes_(2, IMAG) *
                    sin(k01 * x_ph[0] + k02 * x_ph[1] + k03 * x_ph[2]);
      auto f_m4 = k24 * amplitudes_(3, REAL) *
                    cos(k21 * x_ph[0] + k22 * x_ph[1] + k23 * x_ph[2]) +
                  k24 * amplitudes_(3, IMAG) *
                    sin(k21 * x_ph[0] + k22 * x_ph[1] + k23 * x_ph[2]);

      return f_m3 + f_m4;

      // return ZERO;
    }

    Inline auto ext_force_x3(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {

      auto   amplitudes_ = this->amplitudes;
      real_t k01         = 1.0 * constant::TWO_PI / sx1;
      real_t k02         = 0.0 * constant::TWO_PI / sx2;
      real_t k03         = 0.0 * constant::TWO_PI / sx3;
      real_t k04         = 1.0;
      real_t k11         = 0.0 * constant::TWO_PI / sx1;
      real_t k12         = 1.0 * constant::TWO_PI / sx2;
      real_t k13         = 0.0 * constant::TWO_PI / sx3;
      real_t k14         = 1.0;
      real_t k21         = 0.0 * constant::TWO_PI / sx1;
      real_t k22         = 0.0 * constant::TWO_PI / sx2;
      real_t k23         = 1.0 * constant::TWO_PI / sx3;
      real_t k24         = 1.0;

      auto f_m5 = k04 * amplitudes_(4, REAL) *
                    cos(k01 * x_ph[0] + k02 * x_ph[1] + k03 * x_ph[2]) +
                  k04 * amplitudes_(4, IMAG) *
                    sin(k01 * x_ph[0] + k02 * x_ph[1] + k03 * x_ph[2]);
      auto f_m6 = k14 * amplitudes_(5, REAL) *
                    cos(k11 * x_ph[0] + k12 * x_ph[1] + k13 * x_ph[2]) +
                  k14 * amplitudes_(5, IMAG) *
                    sin(k11 * x_ph[0] + k12 * x_ph[1] + k13 * x_ph[2]);

      return f_m5 + f_m6;
    }
#endif

  private:
    // additional problem-specific parameters (i.e., wave numbers in x1, x2, x3 directions)
    const int            nx1, nx2, nx3;
    const real_t         sx1, sx2, sx3, temperature, machno;
    array_t<real_t* [2]> amplitudes;
  };

  Inline void turbulent_fields_2d(const coord_t<Dim2>& x_ph,
                                  vec_t<Dim3>&         e_out,
                                  vec_t<Dim3>&         b_out,
                                  real_t               time,
                                  real_t               sx1,
                                  real_t               sx2,
                                  int                  nx1,
                                  int                  nx2) {
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // ...
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 0.0;
  }

  Inline void turbulent_fields_3d(const coord_t<Dim3>& x_ph,
                                  vec_t<Dim3>&         e_out,
                                  vec_t<Dim3>&         b_out,
                                  real_t               time,
                                  int                  nx1,
                                  int                  nx2,
                                  int                  nx3) {
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // some turbulent magnetic field goes here [TO BE MODIFIED]
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 0.0;
  }

  /**
   * Field initialization for 2D:
   */
  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&,
    Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time = this->time();
    const auto _nx1  = nx1;
    const auto _nx2  = nx2;
    const auto _sx1  = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _sx2  = mblock.metric.x2_max - mblock.metric.x2_min;

    Kokkos::parallel_for(
      "UserInitFields",
      mblock.rangeActiveCells(),
      Lambda(index_t i, index_t j) {
        set_em_fields_2d(mblock, i, j, turbulent_fields_2d, _time, _sx1, _sx2, _nx1, _nx2);
      });
  }

  /**
   * Field initialization for 3D:
   */
  template <>
  inline void ProblemGenerator<Dim3, PICEngine>::UserInitFields(
    const SimulationParams&,
    Meshblock<Dim3, PICEngine>& mblock) {
    const auto _time        = this->time();
    const auto _nx1         = nx1;
    const auto _nx2         = nx2;
    const auto _nx3         = nx3;
    const auto _sx1         = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _sx2         = mblock.metric.x2_max - mblock.metric.x2_min;
    const auto _sx3         = mblock.metric.x3_max - mblock.metric.x3_min;
    auto       amplitudes_  = this->amplitudes;
    const auto _temperature = temperature;
    const auto _machno      = machno;

    auto amp0 = _machno * _temperature * mblock.particles[1].mass() / 6.0;
    auto phi0 = ((real_t)rand() / RAND_MAX) * constant::TWO_PI;

    Kokkos::parallel_for(
      "RandomAmplitudes",
      amplitudes_.extent(0),
      Lambda(index_t i) {
        amplitudes_(i, REAL) = amp0 * cos(phi0);
        amplitudes_(i, IMAG) = amp0 * sin(phi0);
      });

    Kokkos::parallel_for(
      "UserInitFields",
      mblock.rangeActiveCells(),
      Lambda(index_t i, index_t j, index_t k) {
        set_em_fields_3d(mblock, i, j, k, turbulent_fields_3d, _time, _nx1, _nx2, _nx3);
      });
  }

  template <>
  inline void ProblemGenerator<Dim3, PICEngine>::UserDriveParticles(
    const real_t&,
    const SimulationParams&     params,
    Meshblock<Dim3, PICEngine>& mblock) {
    const auto _time        = this->time();
    auto       amplitudes_  = this->amplitudes;
    auto       dt_          = mblock.timestep();
    const auto _sx1         = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _temperature = temperature;
    const auto _machno      = machno;

    auto amp0   = machno * temperature * mblock.particles[1].mass() / 6.0;
    auto omega0 = 0.6 * sqrt(temperature * machno * constant::TWO_PI / sx1);
    auto gamma0 = 0.5 * sqrt(temperature * machno * constant::TWO_PI / sx1);
    auto sigma0 = amp0 * sqrt(6.0 * gamma0);

    auto pool = *(mblock.random_pool_ptr);

    Kokkos::parallel_for(
      "RandomAmplitudes",
      amplitudes_.extent(0),
      Lambda(index_t i) {
        auto rand_gen = pool.get_state();
        auto unr      = rand_gen.frand() - 0.5;
        auto uni      = rand_gen.frand() - 0.5;
        pool.free_state(rand_gen);

        auto ampr_prev = amplitudes_(i, REAL);
        auto ampi_prev = amplitudes_(i, IMAG);

        amplitudes_(i, REAL) = (ampr_prev * cos(omega0 * dt_) +
                                ampi_prev * sin(omega0 * dt_)) *
                                 exp(-gamma0 * dt_) +
                               unr * sigma0;
        amplitudes_(i, IMAG) = (-ampr_prev * sin(omega0 * dt_) +
                                ampi_prev * cos(omega0 * dt_)) *
                                 exp(-gamma0 * dt_) +
                               uni * sigma0;
      });
  }

} // namespace ntt

#endif
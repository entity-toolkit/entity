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
#  include "nttiny/api.h"
#endif

#define REAL (0)
#define IMAG (1)

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
  public:
    real_t work_done { ZERO };
    // read additional parameters from the input file [default to 1 if not specified]
    inline ProblemGenerator(const SimulationParams& params)
      : nx1 { params.get<int>("problem", "nx1", 1) },
        nx2 { params.get<int>("problem", "nx2", 1) },
        nx3 { params.get<int>("problem", "nx3", 1) },
        sx1 { params.extent()[1] - params.extent()[0] },
        sx2 { params.extent()[3] - params.extent()[2] },
        sx3 { 1.0 },
        temperature { params.get<real_t>("problem", "temperature", 0.1) },
        machno { 0.1 },
        amplitudes { "DrivingModes", 6 } {}
    inline void UserDriveParticles(const real_t&,
                                   const SimulationParams&,
                                   Meshblock<D, S>&) override {}
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
#ifdef EXTERNAL_FORCE
    Inline auto ext_force_x1(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      // just as an example, implementing a weird sinusoidal force field in x1
      // return 0.1*math::sin(constant::TWO_PI * x_ph[1] / sx2);
      // return ZERO;

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

      auto   f_m1
        = k14 * amplitudes_(0, REAL) * cos(k11 * x_ph[0] + k12 * x_ph[1] + k13 * 0.0)
          + k14 * amplitudes_(0, IMAG) * sin(k11 * x_ph[0] + k12 * x_ph[1] + k13 * 0.0);
      auto f_m2
        = k24 * amplitudes_(1, REAL) * cos(k21 * x_ph[0] + k22 * x_ph[1] + k23 * 0.0)
          + k24 * amplitudes_(1, IMAG) * sin(k21 * x_ph[0] + k22 * x_ph[1] + k23 * 0.0);

      return f_m1 + f_m2;
    }
    Inline auto ext_force_x2(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      // return ZERO;

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

      auto   f_m3
        = k04 * amplitudes_(2, REAL) * cos(k01 * x_ph[0] + k02 * x_ph[1] + k03 * 0.0)
          + k04 * amplitudes_(2, IMAG) * sin(k01 * x_ph[0] + k02 * x_ph[1] + k03 * 0.0);
      auto f_m4
        = k24 * amplitudes_(3, REAL) * cos(k21 * x_ph[0] + k22 * x_ph[1] + k23 * 0.0)
          + k24 * amplitudes_(3, IMAG) * sin(k21 * x_ph[0] + k22 * x_ph[1] + k23 * 0.0);

      return f_m3 + f_m4;
    }
    Inline auto ext_force_x3(const real_t& time, const coord_t<D>& x_ph) const
      -> real_t override {
      // return ZERO;

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

      auto   f_m5
        = k04 * amplitudes_(4, REAL) * cos(k01 * x_ph[0] + k02 * x_ph[1] + k03 * 0.0)
          + k04 * amplitudes_(4, IMAG) * sin(k01 * x_ph[0] + k02 * x_ph[1] + k03 * 0.0);
      auto f_m6
        = k14 * amplitudes_(5, REAL) * cos(k11 * x_ph[0] + k12 * x_ph[1] + k13 * 0.0)
          + k14 * amplitudes_(5, IMAG) * sin(k11 * x_ph[0] + k12 * x_ph[1] + k13 * 0.0);

      return f_m5 + f_m6;
    }
#endif

#ifdef GUI_ENABLED
    inline void UserInitBuffers_nttiny(
      const SimulationParams&,
      const Meshblock<D, S>&,
      std::map<std::string, nttiny::ScrollingBuffer>& buffers) override {
      nttiny::ScrollingBuffer EM_energy;
      nttiny::ScrollingBuffer PRTL_energy;
      nttiny::ScrollingBuffer FORCE_energy;
      nttiny::ScrollingBuffer TOTAL_energy;
      buffers.insert({ "EM", std::move(EM_energy) });
      buffers.insert({ "PRTL", std::move(PRTL_energy) });
      buffers.insert({ "FORCE", std::move(FORCE_energy) });
      buffers.insert({ "TOTAL", std::move(TOTAL_energy) });
    }

    inline void UserSetBuffers_nttiny(
      const real_t&                                   time,
      const SimulationParams&                         params,
      Meshblock<D, S>&                                mblock,
      std::map<std::string, nttiny::ScrollingBuffer>& buffers) override {
      if constexpr (D == Dim2) {
        real_t em_sum { ZERO }, prtl_sum { ZERO };
        Kokkos::parallel_reduce(
          "EMEnergy",
          mblock.rangeActiveCells(),
          Lambda(index_t i, index_t j, real_t & sum) {
            const real_t i_ = static_cast<real_t>(i);
            const real_t j_ = static_cast<real_t>(j);
            vec_t<Dim3>  E { ZERO }, B { ZERO };
            mblock.metric.v3_Cntrv2Hat(
              { i_ + HALF, j_ + HALF },
              { mblock.em(i, j, em::ex1), mblock.em(i, j, em::ex2), mblock.em(i, j, em::ex3) },
              E);
            mblock.metric.v3_Cntrv2Hat(
              { i_ + HALF, j_ + HALF },
              { mblock.em(i, j, em::bx1), mblock.em(i, j, em::bx2), mblock.em(i, j, em::bx3) },
              B);
            sum += (SQR(E[0]) + SQR(E[1]) + SQR(E[2]) + SQR(B[0]) + SQR(B[1]) + SQR(B[2]))
                   * HALF;
          },
          em_sum);

        em_sum /= SQR(params.larmor0());
        em_sum *= mblock.metric.min_cell_volume();

        for (auto& species : mblock.particles) {
          real_t global_a = ZERO;
          Kokkos::parallel_reduce(
            "ParticleEnergy",
            species.npart(),
            Lambda(index_t p, real_t & sum) {
              sum += math::sqrt(ONE + SQR(species.ux1(p)) + SQR(species.ux2(p))
                                + SQR(species.ux3(p)));
            },
            global_a);
          prtl_sum += global_a;
        }
        buffers["EM"].AddPoint(time, em_sum);
        buffers["PRTL"].AddPoint(time, prtl_sum);
        buffers["FORCE"].AddPoint(time, work_done);
        buffers["TOTAL"].AddPoint(time, em_sum + prtl_sum - work_done);
      }
    }
#endif    // GUI_ENABLED
  private:
    // additional problem-specific parameters (i.e., wave numbers in x1, x2, x3 directions)
    const int            nx1, nx2, nx3;
    const real_t         sx1, sx2, sx3, temperature, machno;
    array_t<real_t* [2]> amplitudes;
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
    // const real_t kx1 = constant::TWO_PI * static_cast<real_t>(nx1) / sx1;
    // const real_t kx2 = constant::TWO_PI * static_cast<real_t>(nx2) / sx2;
    // const real_t ampl_x1
    //   = 0.1 * sx1 * static_cast<real_t>(nx2) / (sx2 * static_cast<real_t>(nx1));
    // const real_t ampl_x2 = -0.1;

    // fields in physical units
    // (
    //    in physical units, when B = 1, and |q|/m = 1
    //    the larmor radius is equal to the fiducial value
    //    i.e., `m c^2 / |q| B == larmor0`
    // )
    e_out[0] = 0.0;
    e_out[1] = 0.0;
    e_out[2] = 0.0;
    // some turbulent magnetic field goes here [TO BE MODIFIED]
    b_out[0] = 0.0;
    b_out[1] = 0.0;
    b_out[2] = 0.0;
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
    b_out[2] = 0.0;
  }

  // /**
  //  * Class that defines the force field applied to particles each time pusher is called.
  //  */
  // template <Dimension D, SimulationEngine S>
  // struct PgenForceField : public ForceField<D, S> {
  //   PgenForceField(const SimulationParams& params, const Meshblock<D, S>& mblock)
  //     : ForceField<D, S>(params, mblock),
  //       corr_time { params.get<real_t>("problem", "correlation_time") },
  //       sx2 { mblock.metric.x2_max - mblock.metric.x2_min } {}

  //   // force field components in physical units
  //   // arguments are:
  //   //    - time -- physical coordinate
  //   //    - x_ph -- 1D/2D/3D coordinate in physical units
  //   Inline auto x1(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
  //     // just as an example, implementing a weird sinusoidal force field in x1
  //     return math::sin(constant::TWO_PI * x_ph[1] / sx2);
  //     // return ZERO;
  //   }
  //   Inline auto x2(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
  //     return ZERO;
  //   }
  //   Inline auto x3(const real_t& time, const coord_t<D>& x_ph) const -> real_t override {
  //     return ZERO;
  //   }

  // private:
  //   // additional parameters (i.e., correlation time)
  //   const real_t corr_time;
  //   const real_t sx2;
  // };

  /**
   * Field initialization for 2D:
   */
  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserInitFields(
    const SimulationParams&, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time        = this->time();
    const auto _nx1         = nx1;
    const auto _nx2         = nx2;
    const auto _sx1         = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _sx2         = mblock.metric.x2_max - mblock.metric.x2_min;
    auto       amplitudes_  = this->amplitudes;
    const auto _temperature = temperature;
    const auto _machno      = machno;

    auto       amp0         = _machno * _temperature * mblock.particles[1].mass() / 6.0;

    // Initialize the mode driving with random values
    // todo: change number of modes to be driven
    // auto       pool        = *(mblock.random_pool_ptr);
    // auto rand_gen     = pool.get_state();
    auto       phi0 = ((real_t)rand() / RAND_MAX) * constant::TWO_PI;    // rand_gen.frand() *
                                                                         // constant::TWO_PI;
    // pool.free_state(rand_gen);

    Kokkos::parallel_for(
      "RandomAmplitudes", amplitudes_.extent(0), Lambda(index_t i) {
        amplitudes_(i, REAL) = amp0 * cos(phi0);
        amplitudes_(i, IMAG) = amp0 * sin(phi0);
      });

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

  /**
   *
   */
  template <>
  inline void ProblemGenerator<Dim2, PICEngine>::UserDriveParticles(
    const real_t&, const SimulationParams& params, Meshblock<Dim2, PICEngine>& mblock) {
    const auto _time        = this->time();
    auto       amplitudes_  = this->amplitudes;
    auto       dt_          = mblock.timestep();
    const auto _sx1         = mblock.metric.x1_max - mblock.metric.x1_min;
    const auto _temperature = temperature;
    const auto _machno      = machno;

    auto       amp0         = machno * temperature * mblock.particles[1].mass() / 6.0;
    auto       omega0       = 0.6 * sqrt(temperature * machno * constant::TWO_PI / sx1);
    auto       gamma0       = 0.5 * sqrt(temperature * machno * constant::TWO_PI / sx1);
    auto       sigma0       = amp0 * sqrt(6.0 * gamma0);

    // todo: change number of modes to be driven
    auto       pool         = *(mblock.random_pool_ptr);

    Kokkos::parallel_for(
      "RandomAmplitudes", amplitudes_.extent(0), Lambda(index_t i) {
        auto rand_gen = pool.get_state();
        auto unr      = rand_gen.frand() - 0.5;
        auto uni      = rand_gen.frand() - 0.5;
        pool.free_state(rand_gen);

        auto ampr_prev       = amplitudes_(i, REAL);
        auto ampi_prev       = amplitudes_(i, IMAG);

        amplitudes_(i, REAL) = (ampr_prev * cos(omega0 * dt_) + ampi_prev * sin(omega0 * dt_))
                                 * exp(-gamma0 * dt_)
                               + unr * sigma0;
        amplitudes_(i, IMAG) = (-ampr_prev * sin(omega0 * dt_) + ampi_prev * cos(omega0 * dt_))
                                 * exp(-gamma0 * dt_)
                               + uni * sigma0;
      });

    auto testout = Kokkos::create_mirror_view(amplitudes_);
    Kokkos::deep_copy(testout, amplitudes_);
    printf("amplitudes_real: %f %f %f %f %f %f\n",
           testout(0, 0),
           testout(1, 0),
           testout(2, 0),
           testout(3, 0),
           testout(4, 0),
           testout(5, 0));
    printf("amplitudes_imag: %f %f %f %f %f %f\n",
           testout(0, 1),
           testout(1, 1),
           testout(2, 1),
           testout(3, 1),
           testout(4, 1),
           testout(5, 1));

    // real_t global_sum = ZERO;
    // Kokkos::parallel_reduce(
    //   "EMEnergy",
    //   mblock.rangeActiveCells(),
    //   ClassLambda(index_t i, index_t j, real_t & sum) {
    //     sum += (SQR(mblock.em(i, j, em::ex1)) + SQR(mblock.em(i, j, em::ex2))
    //             + SQR(mblock.em(i, j, em::ex3)) + SQR(mblock.em(i, j, em::bx1))
    //             + SQR(mblock.em(i, j, em::bx2)) + SQR(mblock.em(i, j, em::bx3)))
    //            * HALF;
    //   },
    //   global_sum);

    // global_sum /= SQR(params.larmor0());
    // global_sum *= mblock.metric.min_cell_volume();
    // printf("EM energy: %e\n", global_sum);
    // global_sum = ZERO;

    // for (auto& species : mblock.particles) {
    //   real_t global_a = ZERO;
    //   Kokkos::parallel_reduce(
    //     "ParticleEnergy",
    //     species.npart(),
    //     ClassLambda(index_t p, real_t & sum) {
    //       sum += math::sqrt(ONE + SQR(species.ux1(p)) + SQR(species.ux2(p))
    //                         + SQR(species.ux3(p)));
    //     },
    //     global_a);
    //   global_sum += global_a;
    // }
    // printf("Particle energy: %e\n", global_sum);
  }

}    // namespace ntt

#endif
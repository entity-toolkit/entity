#ifndef PROBLEM_GENERATOR_H
#define PROBLEM_GENERATOR_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/numeric.h"

#include "archetypes/problem_generator.h"
#include "framework/domain/metadomain.h"

#include <vector>

#define REAL (0)
#define IMAG (1)

namespace user {
  using namespace ntt;

  template <Dimension D>
  struct InitFields {
    InitFields(array_t<real_t* [2]> amplitudes, real_t amp0, real_t phi0)
      : amps { amplitudes }
      , amp { amp0 }
      , phi { phi0 } {}

    Inline auto ex1(const coord_t<D>& x_Ph) const -> real_t {

    // TODO: THIS MIGHT BETTER GO SOMEWHERE ELSE? 
    Kokkos::parallel_for(
      "RandomAmplitudes",
      amps.extent(0),
      Lambda(index_t i) {
        amps(i, REAL) = amp * cos(phi);
        amps(i, IMAG) = amp * sin(phi);
      });

      return 0.0;
    }

    Inline auto ex2(const coord_t<D>& x_Ph) const -> real_t {
      return 0.0;
    }

    Inline auto ex3(const coord_t<D>& x_Ph) const -> real_t {
      return 0.0;
    }

    Inline auto bx1(const coord_t<D>& x_Ph) const -> real_t {
      return 0.0;
    }

    Inline auto bx2(const coord_t<D>& x_Ph) const -> real_t {
      return 0.0;
    }

    Inline auto bx3(const coord_t<D>& x_Ph) const -> real_t {
      return 0.0;
    }
    
  private:
    array_t<real_t* [2]> amps;
    const real_t         amp, phi;
  };

  template <Dimension D>
  struct ExtForce {
    ExtForce(real_t SX1, real_t SX2, real_t SX3) 
    : sx1 {SX1}
    , sx2 {SX2}
    , sx3 {SX3} {}
    const std::vector<unsigned short> species { 1, 2 };

    ExtForce() = default;

    Inline auto fx1(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;

      // auto   amplitudes_ = this->amplitudes;
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

      // auto f_m1 = k14 * amplitudes_(0, REAL) *
      //               cos(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2]) +
      //             k14 * amplitudes_(0, IMAG) *
      //               sin(k11 * x_Ph[0] + k12 * x_Ph[1] + k13 * x_Ph[2]);
      // auto f_m2 = k24 * amplitudes_(1, REAL) *
      //               cos(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]) +
      //             k24 * amplitudes_(1, IMAG) *
      //               sin(k21 * x_Ph[0] + k22 * x_Ph[1] + k23 * x_Ph[2]);

      // return f_m1 + f_m2;
      return ZERO;

    }

    Inline auto fx2(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;
      return ZERO;
    }

    Inline auto fx3(const unsigned short& sp,
                    const real_t&         time,
                    const coord_t<D>&     x_Ph) const -> real_t {
      (void)sp;
      (void)time;
      (void)x_Ph;
      return ZERO;
    }

  private:
    const real_t sx1, sx2, sx3;

  };

  template <SimEngine::type S, class M>
  struct PGen : public arch::ProblemGenerator<S, M> {
    // compatibility traits for the problem generator
    static constexpr auto engines = traits::compatible_with<SimEngine::SRPIC>::value;
    static constexpr auto metrics = traits::compatible_with<Metric::Minkowski>::value;
    static constexpr auto dimensions = traits::compatible_with<Dim::_2D, Dim::_3D>::value;

    // for easy access to variables in the child class
    using arch::ProblemGenerator<S, M>::D;
    using arch::ProblemGenerator<S, M>::C;
    using arch::ProblemGenerator<S, M>::params;

    const real_t         SX1, SX2, SX3;
    const real_t         temperature, machno;
    const real_t         amp0, phi0;
    array_t<real_t* [2]> amplitudes;
    ExtForce<M::PrtlDim> ext_force;
    InitFields<D> init_flds;

    inline PGen(const SimulationParams& p, const Metadomain<S, M>& global_domain)
      : arch::ProblemGenerator<S, M> { p }
      , SX1 { 2.0 }
      , SX2 { 2.0 }
      , SX3 { 2.0 }
      , temperature { params.template get<real_t>("problem.temperature", 0.1) }
      , machno { params.template get<real_t>("problem.machno", 0.1) }
      , amp0 { machno * temperature * 1.0 / 6.0 } // TODO: HERE NEED TO INCLUDE PARTICLE MASS
      , phi0 { ((real_t)rand() / RAND_MAX) * constant::TWO_PI }
      , amplitudes { "DrivingModes", 6 }
      , ext_force {SX1, SX2, SX3}
      , init_flds {amplitudes, amp0, phi0} {
      }

    void CustomPostStep(std::size_t, long double time, Domain<S, M>& domain) {

    auto amp0   = machno * temperature * 1.0 / 6.0;
    auto omega0 = 0.6 * sqrt(temperature * machno * constant::TWO_PI / SX1);
    auto gamma0 = 0.5 * sqrt(temperature * machno * constant::TWO_PI / SX2);
    auto sigma0 = amp0 * sqrt(6.0 * gamma0);

    // auto pool = *(mblock.random_pool_ptr);

    // Kokkos::parallel_for(
    //   "RandomAmplitudes",
    //   amplitudes_.extent(0),
    //   Lambda(index_t i) {
    //     auto rand_gen = pool.get_state();
    //     auto unr      = rand_gen.frand() - 0.5;
    //     auto uni      = rand_gen.frand() - 0.5;
    //     pool.free_state(rand_gen);

    //     auto ampr_prev = amplitudes_(i, REAL);
    //     auto ampi_prev = amplitudes_(i, IMAG);

    //     amplitudes_(i, REAL) = (ampr_prev * cos(omega0 * dt_) +
    //                             ampi_prev * sin(omega0 * dt_)) *
    //                              exp(-gamma0 * dt_) +
    //                            unr * sigma0;
    //     amplitudes_(i, IMAG) = (-ampr_prev * sin(omega0 * dt_) +
    //                             ampi_prev * cos(omega0 * dt_)) *
    //                              exp(-gamma0 * dt_) +
    //                            uni * sigma0;
    //   });

    }

  };

} // namespace user

#endif

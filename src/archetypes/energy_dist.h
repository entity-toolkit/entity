/**
 * @file archetypes/energy_dist.hpp
 * @brief Defines an archetype for energy distributions
 * @implements
 *   - arch::EnergyDistribution<>
 *   - arch::ColdDist<> : arch::EnergyDistribution<>
 *   - arch::Maxwellian<> : arch::EnergyDistribution<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - utils/comparators.h
 *   - utils/error.h
 *   - utils/numeric.h
 *   - arch/kokkos_aliases.h
 * @namespaces:
 *   - arch::
 * @note
 * The class returns a random velocity according to a coded distribution
 * For Cartesian: the returned velocity is in the global Cartesian basis
 * For non-Cartesian SR: the returned velocity is in the tetrad basis
 * For GR: the returned velocity is in the covariant basis
 */

#ifndef ARCHETYPES_ENERGY_DIST_HPP
#define ARCHETYPES_ENERGY_DIST_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace arch {
  using namespace ntt;

  template <SimEngine::type S, class M>
  struct EnergyDistribution {
    static constexpr auto D = M::Dim;
    static constexpr bool is_energy_dist { true };
    static_assert(M::is_metric, "M must be a metric class");

    EnergyDistribution(const M& metric) : metric { metric } {}

    // Takes the physical coordinate of the particle and returns
    //   the velocity in tetrad basis
    // last argument -- is the species index (1, ..., nspec)
    Inline virtual void operator()(const coord_t<D>&,
                                   vec_t<Dim::_3D>& v,
                                   unsigned short = 0) const {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }

  protected:
    const M metric;
  };

  template <SimEngine::type S, class M>
  struct ColdDist : public EnergyDistribution<S, M> {
    ColdDist(const M& metric) : EnergyDistribution<S, M> { metric } {}

    Inline void operator()(const coord_t<M::Dim>&,
                           vec_t<Dim::_3D>& v,
                           unsigned short = 0) const override {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <SimEngine::type S, class M>
  struct Maxwellian : public EnergyDistribution<S, M> {
    using EnergyDistribution<S, M>::metric;

    Maxwellian(const M&              metric,
               random_number_pool_t& pool,
               real_t                temperature,
               real_t                boost_vel       = ZERO,
               short                 boost_direction = 0)
      : EnergyDistribution<S, M> { metric }
      , pool { pool }
      , temperature { temperature }
      , boost_velocity { boost_vel }
      , boost_direction { boost_direction } {
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative",
                     HERE);
      raise::ErrorIf(
        (boost_direction != 0) && (M::CoordType != Coord::Cart),
        "Maxwellian: Boosting is only supported in Cartesian coordinates",
        HERE);
    }

    // Juttner-Synge distribution
    Inline void JS(vec_t<Dim::_3D>& v, const real_t& temp) const {
      auto   rand_gen = pool.get_state();
      real_t randu { Random<real_t>(rand_gen) },
        randeta { Random<real_t>(rand_gen) };
      real_t randX1 { Random<real_t>(rand_gen) },
        randX2 { Random<real_t>(rand_gen) };
      if (temp < static_cast<real_t>(0.5)) {
        // Juttner-Synge distribution using the Box-Muller method - non-relativistic

        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[0]   = randX1 * math::cos(randX2) * math::sqrt(temp);

        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

        randX1 = Random<real_t>(rand_gen);
        while (cmp::AlmostZero(randX1)) {
          randX1 = Random<real_t>(rand_gen);
        }
        randX1 = math::sqrt(-TWO * math::log(randX1));
        randX2 = constant::TWO_PI * Random<real_t>(rand_gen);
        v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      } else {
        // Juttner-Synge distribution using the Sobol method - relativistic
        randu = ONE;
        while (SQR(randeta) <= SQR(randu) + ONE) {
          randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                   Random<real_t>(rand_gen);
          while (cmp::AlmostZero(randX1)) {
            randX1 = Random<real_t>(rand_gen) * Random<real_t>(rand_gen) *
                     Random<real_t>(rand_gen);
          }
          randu  = -temp * math::log(randX1);
          randX2 = Random<real_t>(rand_gen);
          while (cmp::AlmostZero(randX2)) {
            randX2 = Random<real_t>(rand_gen);
          }
          randeta = -temp * math::log(randX1 * randX2);
        }
        randX1 = Random<real_t>(rand_gen);
        randX2 = Random<real_t>(rand_gen);
        v[0]   = randu * (TWO * randX1 - ONE);
        v[2]   = TWO * randu * math::sqrt(randX1 * (ONE - randX1));
        v[1]   = v[2] * math::cos(constant::TWO_PI * randX2);
        v[2]   = v[2] * math::sin(constant::TWO_PI * randX2);
      }
      pool.free_state(rand_gen);
    }

    // Boost a symmetric distribution to a relativistic speed using flipping
    // method https://arxiv.org/pdf/1504.03910.pdf
    Inline void boost(vec_t<Dim::_3D>& v,
                      const real_t&    boost_vel,
                      const short&     boost_direction) const {
      auto   rand_gen = pool.get_state();
      real_t boost_beta { boost_vel / math::sqrt(ONE + SQR(boost_vel)) };
      real_t boost_gamma { boost_vel / boost_beta };
      real_t ut { math::sqrt(ONE + SQR(v[0]) + SQR(v[1]) + SQR(v[2])) };
      if (-boost_beta * v[boost_direction] > ut * Random<real_t>(rand_gen)) {
        v[boost_direction] = -v[boost_direction];
      }
      pool.free_state(rand_gen);
      v[boost_direction] = boost_gamma * (v[boost_direction] + boost_beta * ut);
    }

    Inline void operator()(const coord_t<M::Dim>& x_Code,
                           vec_t<Dim::_3D>&       v,
                           unsigned short = 0) const override {
      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        JS(v, temperature);
      }
      if constexpr (S == SimEngine::GRPIC) {
        // convert from the tetrad basis to covariant
        vec_t<Dim::_3D> v_Hat;
        v_Hat[0] = v[0];
        v_Hat[1] = v[1];
        v_Hat[2] = v[2];
        metric.template transform<Idx::T, Idx::D>(x_Code, v_Hat, v);
      }
      if constexpr (M::CoordType == Coord::Cart) {
        // boost only when using cartesian coordinates
        if (not cmp::AlmostZero(boost_velocity)) {
          if (boost_direction < 0) {
            boost(v, -boost_velocity, -boost_direction - 1);
          } else if (boost_direction > 0) {
            boost(v, boost_velocity, boost_direction - 1);
          }
          // no boost when boost_direction == 0
        }
      }
    }

  private:
    random_number_pool_t pool;

    const real_t temperature;
    const real_t boost_velocity;
    const short  boost_direction;
  };

} // namespace arch

#endif // ARCHETYPES_ENERGY_DIST_HPP
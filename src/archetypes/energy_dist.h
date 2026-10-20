/**
 * @file archetypes/energy_dist.h
 * @brief Defines an archetype for energy distributions
 * @implements
 *   - arch::energy_dist::Cold<>
 *   - arch::energy_dist::Powerlaw<>
 *   - arch::energy_dist::Maxwellian<>
 * @namespaces:
 *   - arch::energy_dist::
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

namespace arch::energy_dist {
  using namespace ntt;

  template <Dimension D>
  struct Cold {
    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {

      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    }
  };

  template <Dimension D>
  struct Powerlaw {

    Powerlaw(random_number_pool_t& pool, real_t g_min, real_t g_max, real_t pl_ind)
      : g_min { g_min }
      , g_max { g_max }
      , pl_ind { pl_ind }
      , pool { pool } {}

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      auto rand_gen = pool.get_state();
      auto rand_X1  = Random<real_t>(rand_gen);
      auto rand_gam = ONE;

      // Power-law distribution from uniform (see https://mathworld.wolfram.com/RandomNumber.html)
      if (pl_ind != -ONE) {
        rand_gam += math::pow(
          math::pow(g_min, ONE + pl_ind) +
            (-math::pow(g_min, ONE + pl_ind) + math::pow(g_max, ONE + pl_ind)) *
              rand_X1,
          ONE / (ONE + pl_ind));
      } else {
        rand_gam += math::pow(g_min, ONE - rand_X1) * math::pow(g_max, rand_X1);
      }
      auto rand_u  = math::sqrt(SQR(rand_gam) - ONE);
      auto rand_X2 = Random<real_t>(rand_gen);
      auto rand_X3 = Random<real_t>(rand_gen);
      v[0]         = rand_u * (TWO * rand_X2 - ONE);
      v[2]         = TWO * rand_u * math::sqrt(rand_X2 * (ONE - rand_X2));
      v[1] = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * rand_X3);
      v[2] = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * rand_X3);

      pool.free_state(rand_gen);
    }

  private:
    const real_t         g_min, g_max, pl_ind;
    random_number_pool_t pool;
  };

  Inline void NonRelMaxwellian(vec_t<Dim::_3D>&            v,
                               real_t                      temp,
                               const random_number_pool_t& pool) {
    auto   rand_gen = pool.get_state();
    // Juttner-Synge distribution using the Box-Muller method - non-relativistic
    real_t randX1   = Random<real_t>(rand_gen);
    while (cmp::AlmostZero(randX1)) {
      randX1 = Random<real_t>(rand_gen);
    }
    randX1        = math::sqrt(-TWO * math::log(randX1));
    real_t randX2 = static_cast<real_t>(constant::TWO_PI) *
                    Random<real_t>(rand_gen);
    v[0] = randX1 * math::cos(randX2) * math::sqrt(temp);

    randX1 = Random<real_t>(rand_gen);
    while (cmp::AlmostZero(randX1)) {
      randX1 = Random<real_t>(rand_gen);
    }
    randX1 = math::sqrt(-TWO * math::log(randX1));
    randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
    v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

    randX1 = Random<real_t>(rand_gen);
    while (cmp::AlmostZero(randX1)) {
      randX1 = Random<real_t>(rand_gen);
    }
    randX1 = math::sqrt(-TWO * math::log(randX1));
    randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
    v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);
    pool.free_state(rand_gen);
  }

  Inline void JuttnerSinge(vec_t<Dim::_3D>&            v,
                           real_t                      temp,
                           const random_number_pool_t& pool) {
    auto   rand_gen = pool.get_state();
    real_t randX1, randX2;
    if (temp < static_cast<real_t>(0.5)) {
      // Juttner-Synge distribution using the Box-Muller method - non-relativistic
      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[0]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[1]   = randX1 * math::cos(randX2) * math::sqrt(temp);

      randX1 = Random<real_t>(rand_gen);
      while (cmp::AlmostZero(randX1)) {
        randX1 = Random<real_t>(rand_gen);
      }
      randX1 = math::sqrt(-TWO * math::log(randX1));
      randX2 = static_cast<real_t>(constant::TWO_PI) * Random<real_t>(rand_gen);
      v[2]   = randX1 * math::cos(randX2) * math::sqrt(temp);
    } else {
      // Juttner-Synge distribution using the Sobol method - relativistic
      auto randu   = ONE;
      auto randeta = Random<real_t>(rand_gen);
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
      v[1]   = v[2] * math::cos(static_cast<real_t>(constant::TWO_PI) * randX2);
      v[2]   = v[2] * math::sin(static_cast<real_t>(constant::TWO_PI) * randX2);
    }
    pool.free_state(rand_gen);
  }

  template <bool CanBoost>
  Inline void SampleFromMaxwellian(vec_t<Dim::_3D>&            v,
                                   const random_number_pool_t& pool,
                                   real_t                      temperature,
                                   real_t boost_velocity = static_cast<real_t>(0),
                                   in   boost_direction = in::x1,
                                   bool flip_velocity   = false) {
    if (cmp::AlmostZero(temperature)) {
      v[0] = ZERO;
      v[1] = ZERO;
      v[2] = ZERO;
    } else {
      JuttnerSinge(v, temperature, pool);
    }
    if constexpr (CanBoost) {
      // Boost a symmetric distribution to a relativistic speed using flipping
      // method https://arxiv.org/pdf/1504.03910.pdf
      // @note: boost only when using cartesian coordinates
      if (not cmp::AlmostZero(boost_velocity)) {
        const auto boost_dir = static_cast<dim_t>(boost_direction);
        const auto boost_beta { boost_velocity /
                                math::sqrt(ONE + SQR(boost_velocity)) };
        const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
        auto       rand_gen = pool.get_state();
        if (-boost_beta * v[boost_dir] > gamma * Random<real_t>(rand_gen)) {
          v[boost_dir] = -v[boost_dir];
        }
        pool.free_state(rand_gen);
        v[boost_dir] = math::sqrt(ONE + SQR(boost_velocity)) *
                       (v[boost_dir] + boost_beta * gamma);
        if (flip_velocity) {
          v[0] = -v[0];
          v[1] = -v[1];
          v[2] = -v[2];
        }
      }
    }
  }

  template <Dimension D, Coord::type C>
  struct Maxwellian {
    Maxwellian(random_number_pool_t&      pool,
               real_t                     temperature,
               const std::vector<real_t>& drift_four_vel = { ZERO, ZERO, ZERO })
      : pool { pool }
      , temperature { temperature } {
      raise::ErrorIf(drift_four_vel.size() != 3,
                     "Maxwellian: Drift velocity must be a 3D vector",
                     HERE);
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative",
                     HERE);
      if constexpr (C == Coord::Cartesian) {
        drift_4vel = NORM(drift_four_vel[0], drift_four_vel[1], drift_four_vel[2]);
        if (cmp::AlmostZero_host(drift_4vel)) {
          drift_dir = 0;
        } else {
          drift_3vel   = drift_4vel / math::sqrt(ONE + SQR(drift_4vel));
          drift_dir_x1 = drift_four_vel[0] / drift_4vel;
          drift_dir_x2 = drift_four_vel[1] / drift_4vel;
          drift_dir_x3 = drift_four_vel[2] / drift_4vel;

          // assume drift is in an arbitrary direction
          drift_dir = 4;
          // check whether drift is in one of principal directions
          for (auto d { 0u }; d < 3u; ++d) {
            const auto dprev = (d + 2) % 3;
            const auto dnext = (d + 1) % 3;
            if (cmp::AlmostZero_host(drift_four_vel[dprev]) and
                cmp::AlmostZero_host(drift_four_vel[dnext])) {
              drift_dir = SIGN(drift_four_vel[d]) * (static_cast<real_t>(d + 1));
              break;
            }
          }
        }
        raise::ErrorIf(drift_dir > 3 and drift_dir != 4,
                       "Maxwellian: Incorrect drift direction",
                       HERE);
        raise::ErrorIf(
          drift_dir != 0 and (C != Coord::Cartesian),
          "Maxwellian: Boosting is only supported in Cartesian coordinates",
          HERE);
      }
    }

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {
      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        JuttnerSinge(v, temperature, pool);
      }
      // @note: boost only when using cartesian coordinates
      if constexpr (C == Coord::Cartesian) {
        if (drift_dir != 0) {
          // Boost an isotropic Maxwellian with a drift velocity using
          // flipping method https://arxiv.org/pdf/1504.03910.pdf
          // 1. apply drift in X1 direction
          const auto gamma { U2GAMMA(v[0], v[1], v[2]) };
          auto       rand_gen = pool.get_state();
          if (-drift_3vel * v[0] > gamma * Random<real_t>(rand_gen)) {
            v[0] = -v[0];
          }
          pool.free_state(rand_gen);
          v[0] = math::sqrt(ONE + SQR(drift_4vel)) * (v[0] + drift_3vel * gamma);
          // 2. rotate to desired orientation
          if (drift_dir == -1) {
            v[0] = -v[0];
          } else if (drift_dir == 2 || drift_dir == -2) {
            const auto tmp = v[1];
            v[1]           = drift_dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (drift_dir == 3 || drift_dir == -3) {
            const auto tmp = v[2];
            v[2]           = drift_dir > 0 ? v[0] : -v[0];
            v[0]           = tmp;
          } else if (drift_dir == 4) {
            vec_t<Dim::_3D> v_old;
            v_old[0] = v[0];
            v_old[1] = v[1];
            v_old[2] = v[2];

            v[0] = v_old[0] * drift_dir_x1 - v_old[1] * drift_dir_x2 -
                   v_old[2] * drift_dir_x3;
            v[1] = (v_old[0] * drift_dir_x2 * (drift_dir_x1 + ONE) +
                    v_old[1] *
                      (SQR(drift_dir_x1) + drift_dir_x1 + SQR(drift_dir_x3)) -
                    v_old[2] * drift_dir_x2 * drift_dir_x3) /
                   (drift_dir_x1 + ONE);
            v[2] = (v_old[0] * drift_dir_x3 * (drift_dir_x1 + ONE) -
                    v_old[1] * drift_dir_x2 * drift_dir_x3 -
                    v_old[2] * (-drift_dir_x1 + SQR(drift_dir_x3) - ONE)) /
                   (drift_dir_x1 + ONE);
          }
        }
      }
    }

  private:
    random_number_pool_t pool;

    const real_t temperature;

    real_t drift_3vel { ZERO }, drift_4vel { ZERO };
    // components of the unit vector in the direction of the drift
    real_t drift_dir_x1 { ZERO }, drift_dir_x2 { ZERO }, drift_dir_x3 { ZERO };

    // values of boost_dir:
    // 4 -> arbitrary direction
    // 0 -> no drift
    // +/- 1 -> +/- x1
    // +/- 2 -> +/- x2
    // +/- 3 -> +/- x3
    short drift_dir { 0 };
  };

  template <Dimension D>
  struct MaxwellianNonRel {
    MaxwellianNonRel(random_number_pool_t& pool,
                     real_t                temperature,
                     const std::vector<real_t>& drift_vel = { ZERO, ZERO, ZERO })
      : pool { pool }
      , temperature { temperature } {
      raise::ErrorIf(drift_vel.size() != 3,
                     "Maxwellian: Drift velocity must be a 3D vector",
                     HERE);
      raise::ErrorIf(temperature < ZERO,
                     "Maxwellian: Temperature must be non-negative",
                     HERE);
      if (not cmp::AlmostZero_host(NORM(drift_vel[0], drift_vel[1], drift_vel[2]))) {
        drift_vel_x1 = drift_vel[0];
        drift_vel_x2 = drift_vel[1];
        drift_vel_x3 = drift_vel[2];
      }
    }

    Inline void operator()(const coord_t<D>&, vec_t<Dim::_3D>& v) const {

      if (cmp::AlmostZero(temperature)) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
      } else {
        NonRelMaxwellian(v, temperature, pool);
      }
      v[0] += drift_vel_x1;
      v[1] += drift_vel_x2;
      v[2] += drift_vel_x3;
    }

  private:
    random_number_pool_t pool;

    const real_t temperature;

    real_t drift_vel_x1 { ZERO }, drift_vel_x2 { ZERO }, drift_vel_x3 { ZERO };
  };

} // namespace arch::energy_dist

#endif // ARCHETYPES_ENERGY_DIST_HPP

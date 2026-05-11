#ifndef ARCHETYPES_QED_H
#define ARCHETYPES_QED_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/param_container.h"

#include "framework/containers/particles.h"

namespace arch {
  using namespace ntt;

  constexpr spidx_t MAXSP = 16u;

  template <Dimension D, Coord C>
  struct TwoBodyComptonScattering {
    ParticleArrays species1[MAXSP];
    ParticleArrays species2[MAXSP];

    const real_t         nominal_probability_density;
    const real_t         Thomson_limit;
    random_number_pool_t random_pool;

    TwoBodyComptonScattering(const prm::Parameters& params,
                             random_number_pool_t&  random_pool)
      : nominal_probability_density { params.template get<real_t>(
          "two_body.compton_scattering.nominal_probability_density") }
      , Thomson_limit { params.template get<real_t>(
          "two_body.compton_scattering.Thomson_limit") }
      , random_pool { random_pool } {
      if (nominal_probability_density <= ZERO) {
        raise::Error("nominal_probability must be in the range (0, 1]", HERE);
      }
      if (Thomson_limit <= ZERO or Thomson_limit > 2e-3) {
        raise::Error("Thomson_limit must be in the range (0, 2e-3]", HERE);
      }
    }

    Inline void operator()(spidx_t sp1,
                           npart_t p1,
                           spidx_t sp2,
                           npart_t p2,
                           real_t  tile_volume) const {
      // values with "_" are in the lepton rest-frame
      const auto lepton_ux1    = species1[sp1 - 1].ux1(p1);
      const auto lepton_ux2    = species1[sp1 - 1].ux2(p1);
      const auto lepton_ux3    = species1[sp1 - 1].ux3(p1);
      const auto lepton_gamma  = U2GAMMA(lepton_ux1, lepton_ux2, lepton_ux3);
      const auto lepton_weight = species1[sp1 - 1].weight(p1);

      const auto photon_px1    = species2[sp2 - 1].ux1(p2);
      const auto photon_px2    = species2[sp2 - 1].ux2(p2);
      const auto photon_px3    = species2[sp2 - 1].ux3(p2);
      const auto photon_energy = NORM(photon_px1, photon_px2, photon_px3);
      const auto photon_weight = species2[sp2 - 1].weight(p2);

      // boost photon momentum to lepton rest frame
      real_t photon_px1_ { ZERO }, photon_px2_ { ZERO }, photon_px3_ { ZERO },
        photon_energy_ { ZERO };
      {
        const real_t p_dot_k = DOT(lepton_ux1,
                                   lepton_ux2,
                                   lepton_ux3,
                                   photon_px1,
                                   photon_px2,
                                   photon_px3);

        photon_energy_ = lepton_gamma * photon_energy - p_dot_k;

        photon_px1_ = photon_px1 +
                      (p_dot_k / (ONE + lepton_gamma) - photon_energy) * lepton_ux1;
        photon_px2_ = photon_px2 +
                      (p_dot_k / (ONE + lepton_gamma) - photon_energy) * lepton_ux2;
        photon_px3_ = photon_px3 +
                      (p_dot_k / (ONE + lepton_gamma) - photon_energy) * lepton_ux3;
      }
      const bool KN_regime = photon_energy_ > Thomson_limit;
      real_t     f_KN { ONE };
      if (KN_regime) {
        if (photon_energy_ < static_cast<real_t>(2e-3)) {
          // correctly handle the eph_RF << 1 limit using 2nd order expansion of f_KN
          f_KN = ONE - TWO * photon_energy_ +
                 static_cast<real_t>(5.2) * SQR(photon_energy_);
        } else {
          f_KN = static_cast<real_t>(0.375) *
                 ((ONE - TWO / photon_energy_ - TWO / SQR(photon_energy_)) *
                    math::log(ONE + TWO * photon_energy_) +
                  HALF + FOUR / photon_energy_ -
                  HALF / SQR(ONE + TWO * photon_energy_)) /
                 photon_energy_;
        }
      }
      const auto scattering_probability = nominal_probability_density * f_KN *
                                          photon_energy_ * lepton_weight *
                                          photon_weight /
                                          (photon_energy * lepton_gamma *
                                           tile_volume);
      auto       gen = random_pool.get_state();
      const auto rnd = Random<real_t>(gen);
      random_pool.free_state(gen);

      if (rnd < scattering_probability) {
        // Define an orthonormal basis: {a, b, c} in the lepton frame
        const auto ax1_ { photon_px1_ / photon_energy_ };
        const auto ax2_ { photon_px2_ / photon_energy_ };
        const auto ax3_ { photon_px3_ / photon_energy_ };
        real_t     bx1_ { ONE }, bx2_ { ZERO }, bx3_ { ZERO };
        if (not cmp::AlmostZero(ax1_)) {
          bx1_  = -ax2_ / ax1_;
          bx2_  = ONE / math::sqrt(ONE + SQR(bx1_));
          bx1_ /= math::sqrt(ONE + SQR(bx1_));
        }
        const auto cx1_ { CROSS_x1(ax1_, ax2_, ax3_, bx1_, bx2_, bx3_) };
        const auto cx2_ { CROSS_x2(ax1_, ax2_, ax3_, bx1_, bx2_, bx3_) };
        const auto cx3_ { CROSS_x3(ax1_, ax2_, ax3_, bx1_, bx2_, bx3_) };

        auto       gen_ = random_pool.get_state();
        const auto rnd_ = Random<real_t>(gen_);
        random_pool.free_state(gen_);

        real_t costheta_ { ZERO };
        if (not KN_regime) {
          costheta_ = math::pow(
            FOUR * rnd_ - TWO +
              math::sqrt(FIVE + static_cast<real_t>(16) * rnd_ * (rnd_ - ONE)),
            THIRD);
          costheta_ -= ONE / costheta_;
        } else {
          // iter = 0
          // converged = .false.
          // c0 = 1.0d0 + 2.0d0 * eph_RF
          // c1 = eph_RF / c0
          // c2 = eph_RF**2 - 2.0d0 * eph_RF - 2.0d0
          // c3 = eph_RF - 1.0d0 - 0.5d0 * c1**2
          // c4 = 1.0d0 / (4.0d0 * eph_RF + 2.0d0 * eph_RF * (1.0d0 + eph_RF) * c1**2 + c2 * log(c0))
          // u = 2.0d0 * rnd - 1.0d0
          // do while (iter .lt. max_iter)
          //   iter = iter + 1
          //   du = du_KN_Newt(eph_RF, u, rnd, c0, c1, c2, c3, c4)
          //   u = u + du
          //   if (u .gt. 1.0d0) u = 1.0d0
          //   if (u .lt. -1.0d0) u = -1.0d0
          //   if (abs(du) .lt. thresh) then
          //     converged = .true.
          //     exit
          //   end if
          // end do
        }
        // costheta_RF = u
      } // not interacting
    }
  };

} // namespace arch

#endif // ARCHETYPES_QED_H

/**
 * @file archetypes/qed/compton.h
 * @brief Two-body collision policy of Compton scattering between leptons and photons
 * @implements
 *   - arch::qed::ComptonScattering<>
 * @namespaces:
 *   - arch::qed::
 */
#ifndef ARCHETYPES_QED_COMPTON_H
#define ARCHETYPES_QED_COMPTON_H

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"
#include "utils/param_container.h"

#include "framework/containers/particles.h"

#include <Kokkos_Pair.hpp>

namespace arch::qed {
  using namespace ntt;

  template <Dimension D>
  struct ComptonScattering {
    static constexpr spidx_t MAXSP    = 16u;
    static constexpr int     MAX_ITER = 10;

    ParticleArrays          species[MAXSP];
    static constexpr real_t low_energy_limit = static_cast<real_t>(2e-3);

    const real_t         nominal_probability_density;
    const real_t         Thomson_limit;
    random_number_pool_t random_pool;

    ComptonScattering(const prm::Parameters& params,
                      random_number_pool_t&  random_pool)
      : nominal_probability_density { params.template get<real_t>(
          "qed.compton_scattering.nominal_probability_density") }
      , Thomson_limit { params.template get<real_t>(
          "qed.compton_scattering.Thomson_limit") }
      , random_pool { random_pool } {
      if (nominal_probability_density <= ZERO) {
        raise::Error("nominal_probability must be in the range (0, 1]", HERE);
      }
      if (Thomson_limit <= ZERO or Thomson_limit > low_energy_limit) {
        raise::Error(
          "Thomson_limit must be in the range (0, small_energy_limit]",
          HERE);
      }
    }

    /*
     * Lorentz boost a 4-momentum p of the photon to the frame moving with u
     * @param u: 4-velocity of the boost frame
     * @param gamma: Lorentz factor of the boost frame
     * @param p: 4-momentum of the photon in the lab frame
     * @param e: energy of the photon in the lab frame
     * @return: 4-momentum of the photon in the boost frame
     * @return: energy of the photon in the boost frame
     */
    Inline void LorentzBoost(const vec_t<Dim::_3D>& u,
                             real_t                 gamma,
                             const vec_t<Dim::_3D>& p,
                             real_t                 e,
                             vec_t<Dim::_3D>&       p_,
                             real_t&                e_) const {
      const auto u_dot_p = DOT(u[0], u[1], u[2], p[0], p[1], p[2]);

      e_    = gamma * e - u_dot_p;
      p_[0] = p[0] + (u_dot_p / (ONE + gamma) - e) * u[0];
      p_[1] = p[1] + (u_dot_p / (ONE + gamma) - e) * u[1];
      p_[2] = p[2] + (u_dot_p / (ONE + gamma) - e) * u[2];
    }

    /*
     * Calculate the Klein-Nishina cross section for a photon with energy e_ in the lepton rest frame
     * @param e_: photon energy in the lepton rest frame
     * @return: pair of (is_KN_regime, f_KN) where
     *  - is_KN_regime: whether the photon energy is in the Klein-Nishina regime (e_ > Thomson_limit)
     *  - f_KN: the Klein-Nishina cross section normalized to the Thomson cross section
     * @note for e_ > low_energy_limit, full Klein-Nishina formula
     * @note for Thomson_limit < e_ <= low_energy_limit, 2nd order expansion of the Klein-Nishina formula
     * @note for e_ <= Thomson_limit, return 1 (Thomson limit)
     */
    Inline auto KNCrossSection(real_t e_) const -> Kokkos::pair<bool, real_t> {
      if (e_ > Thomson_limit) {
        if (e_ < low_energy_limit) {
          // correctly handle the e_ << 1 limit using 2nd order expansion of f_KN
          return { true, ONE - TWO * e_ + static_cast<real_t>(5.2) * SQR(e_) };
        } else {
          return { true,
                   static_cast<real_t>(0.375) *
                     ((ONE - TWO / e_ - TWO / SQR(e_)) * math::log(ONE + TWO * e_) +
                      HALF + FOUR / e_ - HALF / SQR(ONE + TWO * e_)) /
                     e_ };
        }
      } else {
        return { false, ONE };
      }
    }

    Inline auto RandomCosTheta_Th() const -> real_t {
      auto       gen_ = random_pool.get_state();
      const auto rnd_ = Random<real_t>(gen_);
      random_pool.free_state(gen_);
      const auto u = math::pow(
        FOUR * rnd_ - TWO +
          math::sqrt(FIVE + static_cast<real_t>(16) * rnd_ * (rnd_ - ONE)),
        THIRD);
      return u - ONE / u;
    }

    Inline auto RandomCosTheta_KN(double e_) const -> real_t {
      auto       gen_ = random_pool.get_state();
      const auto rnd_ = Random<double>(gen_);
      random_pool.free_state(gen_);

      auto u         = 2.0 * rnd_ - 1.0;
      bool converged = false;
      for (int iter = 0; iter < MAX_ITER; ++iter) {
        const auto CDF = (-((2.0 + e_ * (4.0 + e_ - 4.0 * (-1.0 + u) * u * e_ +
                                         2.0 * CUBE(-1.0 + u) * SQR(e_))) /
                            SQR(1.0 + e_ - u * e_)) +
                          (2.0 + e_ * (4.0 - e_ * (7.0 + 16.0 * e_))) /
                            SQR(1.0 + 2.0 * e_) +
                          2.0 * (-2.0 + (-2.0 + e_) * e_) *
                            math::log((1.0 + e_ - u * e_) / (1.0 + 2.0 * e_))) /
                         ((-4.0 * e_ * (2.0 + e_ * (1.0 + e_) * (8.0 + e_))) /
                            SQR(1.0 + 2.0 * e_) +
                          (4.0 - 2.0 * (-2.0 + e_) * e_) *
                            math::log(1.0 + 2.0 * e_));
        const auto dCDF_du = -((CUBE(e_) * SQR(1.0 + 2.0 * e_) *
                                (1.0 + SQR(u) - (-1.0 + u) * (1.0 + SQR(u)) * e_ +
                                 SQR(-1.0 + u) * SQR(e_))) /
                               (CUBE(-1.0 + (-1.0 + u) * e_) *
                                (2.0 * e_ * (2.0 + e_ * (1.0 + e_) * (8.0 + e_)) +
                                 SQR(1.0 + 2.0 * e_) * (-2.0 + (-2.0 + e_) * e_) *
                                   math::log(1.0 + 2.0 * e_))));

        const auto du = (rnd_ - CDF) / dCDF_du;

        u += du;
        if (u > 1.0) {
          u = 1.0;
        } else if (u < -1.0) {
          u = -1.0;
        }
        if (math::abs(du) < 1e-3) {
          converged = true;
          break;
        }
      } // iterative loop for u
      return static_cast<real_t>(u);
    }

    /*
     * Scatter a photon with initial momentum p_ and energy e_ in the lepton
     * rest frame to a new momentum pnew_ and energy enew_
     * @param KN_regime: whether the photon energy is in the Klein-Nishina regime
     * @param p_: initial photon momentum in the lepton rest frame
     * @param e_: initial photon energy in the lepton rest frame
     * @return pnew_: output photon momentum after scattering in the lepton rest frame
     * @return enew_: output photon energy after scattering in the lepton rest frame
     * @note the scattering angle is sampled from the Klein-Nishina differential
     * cross section if KN_regime is true, otherwise it is sampled from the Thomson limit
     */
    Inline void ScatterPhoton(bool                   KN_regime,
                              const vec_t<Dim::_3D>& p_,
                              real_t                 e_,
                              vec_t<Dim::_3D>&       pnew_,
                              real_t&                enew_) const {
      auto rand_costheta_ { ZERO };
      if (not KN_regime) {
        rand_costheta_ = RandomCosTheta_Th();
      } else {
        rand_costheta_ = RandomCosTheta_KN(e_);
      }
      const auto rand_sintheta_ = math::sqrt(ONE - SQR(rand_costheta_));

      auto       gen_      = random_pool.get_state();
      const auto rand_phi_ = static_cast<real_t>(constant::TWO_PI) *
                             Random<real_t>(gen_);
      random_pool.free_state(gen_);
      const auto rand_cosphi_ = math::cos(rand_phi_);
      const auto rand_sinphi_ = math::sin(rand_phi_);

      // Define an orthonormal basis: {a_, b_, c_} in the lepton frame
      const vec_t<Dim::_3D> a_ { p_[0] / e_, p_[1] / e_, p_[2] / e_ };
      vec_t<Dim::_3D>       b_ { ONE, ZERO, ZERO };
      if (not cmp::AlmostZero(a_[0])) {
        b_[0]  = -a_[1] / a_[0];
        b_[1]  = ONE / math::sqrt(ONE + SQR(b_[0]));
        b_[0] /= math::sqrt(ONE + SQR(b_[0]));
      }
      const vec_t<Dim::_3D> c_ {
        CROSS_x1(a_[0], a_[1], a_[2], b_[0], b_[1], b_[2]),
        CROSS_x2(a_[0], a_[1], a_[2], b_[0], b_[1], b_[2]),
        CROSS_x3(a_[0], a_[1], a_[2], b_[0], b_[1], b_[2])
      };

      enew_ = e_ / (ONE + e_ * (ONE - rand_costheta_));

      pnew_[0] = enew_ * (rand_costheta_ * a_[0] +
                          rand_sintheta_ * rand_cosphi_ * b_[0] +
                          rand_sintheta_ * rand_sinphi_ * c_[0]);
      pnew_[1] = enew_ * (rand_costheta_ * a_[1] +
                          rand_sintheta_ * rand_cosphi_ * b_[1] +
                          rand_sintheta_ * rand_sinphi_ * c_[1]);
      pnew_[2] = enew_ * (rand_costheta_ * a_[2] +
                          rand_sintheta_ * rand_cosphi_ * b_[2] +
                          rand_sintheta_ * rand_sinphi_ * c_[2]);
    }

    Inline void operator()(spidx_t sp1,
                           npart_t p1,
                           spidx_t sp2,
                           npart_t p2,
                           real_t  tile_volume) const {
      // @TODO coord/vec conversion
      // values with "_" are in the lepton rest-frame
      const vec_t<Dim::_3D> lepton_u { species[sp1 - 1].ux1(p1),
                                       species[sp1 - 1].ux2(p1),
                                       species[sp1 - 1].ux3(p1) };
      const auto lepton_gamma  = U2GAMMA(lepton_u[0], lepton_u[1], lepton_u[2]);
      const auto lepton_weight = species[sp1 - 1].weight(p1);

      const vec_t<Dim::_3D> photon_p { species[sp2 - 1].ux1(p2),
                                       species[sp2 - 1].ux2(p2),
                                       species[sp2 - 1].ux3(p2) };
      const auto photon_energy = NORM(photon_p[0], photon_p[1], photon_p[2]);
      const auto photon_weight = species[sp2 - 1].weight(p2);

      // boost photon momentum to lepton rest frame
      vec_t<Dim::_3D> photon_p_ { ZERO, ZERO, ZERO };
      real_t          photon_energy_ { ZERO };

      LorentzBoost(lepton_u, lepton_gamma, photon_p, photon_energy, photon_p_, photon_energy_);

      const auto [KN_regime, f_KN]      = KNCrossSection(photon_energy_);
      const auto scattering_probability = nominal_probability_density * f_KN *
                                          photon_energy_ * lepton_weight *
                                          photon_weight /
                                          (photon_energy * lepton_gamma *
                                           tile_volume);
      auto       gen = random_pool.get_state();
      const auto rnd = Random<real_t>(gen);
      random_pool.free_state(gen);

      if (rnd < scattering_probability) {
        vec_t<Dim::_3D> photon_pnew_ { ZERO, ZERO, ZERO },
          photon_pnew { ZERO, ZERO, ZERO };
        real_t photon_energy_new_ { ZERO }, photon_energy_new { ZERO };

        ScatterPhoton(KN_regime,
                      photon_p_,
                      photon_energy_,
                      photon_pnew_,
                      photon_energy_new_);
        LorentzBoost({ -lepton_u[0], -lepton_u[1], -lepton_u[2] },
                     lepton_gamma,
                     photon_pnew_,
                     photon_energy_new_,
                     photon_pnew,
                     photon_energy_new);
        species[sp1 - 1].ux1(p1) += (photon_p[0] - photon_pnew[0]) *
                                    photon_weight / lepton_weight;
        species[sp1 - 1].ux2(p1) += (photon_p[1] - photon_pnew[1]) *
                                    photon_weight / lepton_weight;
        species[sp1 - 1].ux3(p1) += (photon_p[2] - photon_pnew[2]) *
                                    photon_weight / lepton_weight;

        species[sp2 - 1].ux1(p2) = photon_pnew[0];
        species[sp2 - 1].ux2(p2) = photon_pnew[1];
        species[sp2 - 1].ux3(p2) = photon_pnew[2];
      } // not interacting
    }
  };

} // namespace arch::qed

#endif // ARCHETYPES_QED_COMPTON_H

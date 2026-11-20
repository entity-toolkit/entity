/**
 * @file archetypes/emission/bethe_heitler/bh_event_host.h
 * @brief Upper bound of the energy lost by the proton in one Bethe-Heitler
 *        interaction, and the largest weight of the pair that it allows
 * @implements
 *   - arch::bh::event::gamma_beta -> double
 *   - arch::bh::event::dmax_at -> double
 *   - arch::bh::event::eps_prime_hi -> double
 *   - arch::bh::event::dmax_unweighted -> double
 *   - arch::bh::event::a_max -> double
 * @namespaces:
 *   - arch::bh::event::
 *
 * @note
 * Host-only code that does not depend on Kokkos. Energies are in m_e c^2.
 *
 * @note
 * D = E- + E+ - eps is the energy that the proton loses in an interaction:
 * the energies of the two leptons minus that of the absorbed photon, in the
 * simulation frame. In the rest frame of the proton, where the photon has
 * the energy eps', the component of the total momentum of the pair along the
 * velocity of the proton lies in [-sqrt(eps'^2 - 4), sqrt(eps'^2 - 4)], and
 * eps' is at most eps'_hi = Gamma eps (1 + beta). Hence
 *   D <= D_max(eps, eps') = Gamma eps' - eps + Gamma beta sqrt(eps'^2 - 4),
 * which increases with eps' and with eps. Over the whole spectrum the bound
 * is reached at its largest photon energy eps_max and at
 * eps' = eps'_hi(eps_max):
 *   D_max(Gamma) = Gamma eps'_hi - eps_max + Gamma beta sqrt(eps'_hi^2 - 4).
 *
 * @note
 * A pair of weight a, in units of the weight of the proton, removes the
 * energy a D from the kinetic energy K = mu_r (Gamma - 1) of the proton, mu_r
 * being the proton-to-electron mass ratio. The largest weight for which no
 * outcome removes more than the fraction f_max of K is
 *   a_max(Gamma) = f_max K / D_max(Gamma).
 * For eps_max <= 1, a_max decreases monotonically with Gamma above the
 * threshold (K grows as Gamma - 1, D_max as Gamma^2). For a harder spectrum
 * it first rises: for eps_max = 1.9 (Gamma_th = 1.0013) from the threshold
 * to a maximum near Gamma = 1.33, and decreases beyond. The smallest a_max
 * over a range of Gamma is therefore to be found by scanning the range.
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_EVENT_HOST_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_EVENT_HOST_H

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace arch::bh::event {

  /**
   * @brief Gamma beta from Gamma (no cancellation for Gamma close to 1)
   */
  inline auto gamma_beta(double Gamma) -> double {
    return std::sqrt((Gamma - 1.0) * (Gamma + 1.0));
  }

  /**
   * @brief D_max(eps, eps') at a given (eps, eps'); 0 below the threshold of
   *        pair production, eps' < 2 (no interaction)
   */
  inline auto dmax_at(double Gamma, double eps, double eps_prime) -> double {
    if (!(eps_prime >= 2.0)) {
      return 0.0;
    }
    const double gb = gamma_beta(Gamma);
    return Gamma * eps_prime - eps +
           gb * std::sqrt((eps_prime - 2.0) * (eps_prime + 2.0));
  }

  /**
   * @brief The largest energy eps'_hi = Gamma eps (1 + beta), in the rest
   *        frame of the proton, of a photon of energy eps in the simulation
   *        frame
   */
  inline auto eps_prime_hi(double Gamma, double eps) -> double {
    return Gamma * eps + gamma_beta(Gamma) * eps;
  }

  /**
   * @brief D_max(Gamma): the bound over every outcome for a bath whose
   *        spectrum ends at eps_max (eps0 for a monochromatic bath); 0 when
   *        eps'_hi(eps_max) < 2 (below the threshold of the hardest photon:
   *        no interaction)
   */
  inline auto dmax_unweighted(double Gamma, double eps_max) -> double {
    return dmax_at(Gamma, eps_max, eps_prime_hi(Gamma, eps_max));
  }

  /**
   * @brief a_max(Gamma) = f_max K / D_max(Gamma) with K = mu_r (Gamma - 1);
   *        +inf when D_max = 0 (no interaction is possible: no bound can be
   *        exceeded)
   */
  inline auto a_max(double Gamma, double eps_max, double mu_r, double f_max)
    -> double {
    const double D = dmax_unweighted(Gamma, eps_max);
    if (!(D > 0.0)) {
      return std::numeric_limits<double>::infinity();
    }
    return f_max * mu_r * (Gamma - 1.0) / D;
  }

} // namespace arch::bh::event

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_EVENT_HOST_H

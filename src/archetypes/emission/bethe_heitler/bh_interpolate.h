/**
 * @file archetypes/emission/bethe_heitler/bh_interpolate.h
 * @brief Interpolation of the rate tables on the host, with the same law as
 *        the lookup on the device (bh_tables_device.h)
 * @implements
 *   - arch::bh::interpolate_loss_host -> double
 *   - arch::bh::interpolate_outcome_host -> void
 *   - arch::bh::interpolate_rate_host -> double
 * @namespaces:
 *   - arch::bh::
 *
 * @note
 * Host-only code that does not depend on Kokkos. The rates are interpolated
 * linearly in y = log10(Gamma - Gamma_th) of their logarithm; they are zero
 * below the lower edge of the table and equal to the value at the last node
 * above its upper edge.
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_INTERPOLATE_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_INTERPOLATE_H

#include "archetypes/emission/bethe_heitler/bh_physics.h"

#include <cmath>
#include <cstddef>

namespace arch::bh {

  /**
   * @brief Energy-loss rate I(Gamma): linear in y of ln I between the nodes
   *        of the host table
   */
  inline auto interpolate_loss_host(const HadronicTablesHost& T, double Gamma)
    -> double {
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      return 0.0;
    }
    if (Gamma >= T.gamma_max) {
      return T.I.back();
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      return 0.0;
    }
    if (yy >= T.y_max) {
      return T.I.back();
    }
    // bracket by bisection on the stored nodes
    std::size_t lo = 0, hi = T.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    return std::exp(
      std::log(T.I[lo]) + t * (std::log(T.I[hi]) - std::log(T.I[lo])));
  }

  /**
   * @brief Mean loss per interaction <D>(Gamma) and mean energy of the
   *        absorbed photons eps_mean(Gamma): linear in y of ln <D> and of
   *        eps_mean between the nodes of the host table; the value at the
   *        first node below the lower edge, the value at the last node at and
   *        above the upper edge
   */
  inline auto interpolate_outcome_host(const HadronicTablesHost& T,
                                       double                    Gamma,
                                       double&                   Dmean,
                                       double& eps_mean) -> void {
    const std::size_t last = T.size() - 1;
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      Dmean    = T.Dmean[0];
      eps_mean = T.eps_mean[0];
      return;
    }
    if (Gamma >= T.gamma_max) {
      Dmean    = T.Dmean[last];
      eps_mean = T.eps_mean[last];
      return;
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      Dmean    = T.Dmean[0];
      eps_mean = T.eps_mean[0];
      return;
    }
    if (yy >= T.y_max) {
      Dmean    = T.Dmean[last];
      eps_mean = T.eps_mean[last];
      return;
    }
    std::size_t lo = 0, hi = last;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    Dmean          = std::exp(std::log(T.Dmean[lo]) +
                     t * (std::log(T.Dmean[hi]) - std::log(T.Dmean[lo])));
    eps_mean       = T.eps_mean[lo] + t * (T.eps_mean[hi] - T.eps_mean[lo]);
  }

  /**
   * @brief Interaction rate J(Gamma): linear in y of ln J between the nodes
   *        of the host table, as `HadronicTables::J` on the device
   */
  inline auto interpolate_rate_host(const HadronicTablesHost& T, double Gamma)
    -> double {
    if (!(Gamma > T.gamma_th) || !(Gamma >= T.gamma_min)) {
      return 0.0;
    }
    if (Gamma >= T.gamma_max) {
      return T.J.back();
    }
    const double yy = std::log10(Gamma - T.gamma_th);
    if (yy < T.y_min) {
      return 0.0;
    }
    if (yy >= T.y_max) {
      return T.J.back();
    }
    std::size_t lo = 0, hi = T.size() - 1;
    while (hi - lo > 1) {
      const std::size_t mid = (lo + hi) / 2;
      if (T.y[mid] <= yy) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double t = (yy - T.y[lo]) / (T.y[hi] - T.y[lo]);
    return std::exp(
      std::log(T.J[lo]) + t * (std::log(T.J[hi]) - std::log(T.J[lo])));
  }

} // namespace arch::bh

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_INTERPOLATE_H

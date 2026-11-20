/**
 * @file archetypes/emission/bethe_heitler/bh_tables_device.h
 * @brief Device views of the Bethe-Heitler rate table and the lookup of the
 *        interaction rate J(Gamma) on the device
 * @implements
 *   - arch::bh::HadronicTables
 *   - arch::bh::mirror_tables_to_device -> HadronicTables
 * @namespaces:
 *   - arch::bh::
 *
 * @note
 * The host table (`arch::bh::HadronicTablesHost`, bh_physics.h) is built in
 * double precision on the host. `mirror_tables_to_device` copies its node
 * arrays into `Kokkos::View<double*>` and records the description of the
 * grid. The lookup interpolates ln J linearly in
 * y = log10(Gamma - Gamma_th); it returns 0 below the lower edge of the
 * table and the value at the last node above its upper edge. All the
 * arithmetic is in double precision.
 *
 * @note
 * Nodes: y_min + i / 256 for i = 0..256 (the first decade, `n_fine` nodes
 * with both ends), then y_min + 1 + j / 64 for j >= 1 while below
 * y_max - 1e-9, then y_max, where Gamma is exactly the upper edge. The cell
 * is located with the index computed from this layout, and the interpolation
 * uses the stored positions of the nodes, so that the shorter last interval
 * and a table that ends within the first decade are handled by the same
 * code.
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_TABLES_DEVICE_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_TABLES_DEVICE_H

#include "arch/kokkos_aliases.h"

#include "archetypes/emission/bethe_heitler/bh_interpolate.h"
#include "archetypes/emission/bethe_heitler/bh_physics.h"

#include <Kokkos_Core.hpp>

#include <cmath>
#include <stdexcept>
#include <string>

namespace arch::bh {

  /**
   * @brief Device views of the rate table and the description of its grid
   * @note `log_rate` holds ln J at the nodes (interpolated linearly in y),
   *       `rate` the values of J at the nodes
   */
  struct HadronicTables {
    // threshold Lorentz factor (the grid variable is y = log10(Gamma - gamma_th))
    double gamma_th { 0.0 };
    double gamma_min { 0.0 }; // lower edge: J = 0 below it
    double gamma_max { 0.0 }; // upper edge: the edge value is returned above it
    double y_min { 0.0 };
    double y_max { 0.0 };
    double dy_fine { 1.0 / 256.0 };
    double dy_coarse { 1.0 / 64.0 };
    int    n_fine { 0 }; // nodes in the first decade (including both ends)
    int    n { 0 };      // total node count

    array_t<double*> y;        // node positions in y
    array_t<double*> gamma;    // node Gamma
    array_t<double*> rate;     // J(Gamma)
    array_t<double*> log_rate; // ln J

    /**
     * @brief Index of the cell containing y (0 <= idx <= n - 2), for
     *        y_min <= y < y_max
     */
    Inline auto cell_index(double yy) const -> int {
      int idx;
      if (yy < y_min + 1.0 || n_fine < 257) {
        idx = static_cast<int>((yy - y_min) / dy_fine);
      } else {
        idx = (n_fine - 1) + static_cast<int>((yy - y_min - 1.0) / dy_coarse);
      }
      if (idx < 0) {
        idx = 0;
      }
      if (idx > n - 2) {
        idx = n - 2;
      }
      // guard against the rounding of the computed index: at most one step
      if (idx + 1 < n - 1 && yy >= y(idx + 1)) {
        ++idx;
      }
      if (idx > 0 && yy < y(idx)) {
        --idx;
      }
      return idx;
    }

    /**
     * @brief Linear interpolation in y of a log-valued node array
     * @param logv ln of the node values, linv the node values themselves
     * @param above set to true if Gamma lies above the upper edge (edge value
     *        returned)
     * @return the interpolated value, 0 below the lower edge
     */
    Inline auto interpolate_log(const array_t<double*>& logv,
                                const array_t<double*>& linv,
                                double                  Gamma,
                                bool&                   above) const -> double {
      above = false;
      if (!(Gamma > gamma_th) || !(Gamma >= gamma_min)) {
        return 0.0;
      }
      if (Gamma >= gamma_max) {
        above = Gamma > gamma_max;
        return linv(n - 1); // the edge value (the last node exactly)
      }
      const double yy = math::log10(Gamma - gamma_th);
      if (yy < y_min) {
        return 0.0;
      }
      if (yy >= y_max) {
        return linv(n - 1);
      }
      const int    i  = cell_index(yy);
      const double y0 = y(i), y1 = y(i + 1);
      const double t = (yy - y0) / (y1 - y0);
      return math::exp(logv(i) + t * (logv(i + 1) - logv(i)));
    }

    /**
     * @brief Interaction rate J(Gamma) per unit photon density (alpha r_e^2 c)
     */
    Inline auto J(double Gamma) const -> double {
      bool above;
      return interpolate_log(log_rate, rate, Gamma, above);
    }
  };

  /**
   * @brief Copy the host table to device views
   * @throws std::invalid_argument on an empty or inconsistent host table
   */
  inline auto mirror_tables_to_device(const HadronicTablesHost& T)
    -> HadronicTables {
    if (T.size() < 2) {
      throw std::invalid_argument(
        "mirror_tables_to_device: fewer than two nodes in the host table");
    }
    if (T.y.size() != T.size() || T.I.size() != T.size() || T.J.size() != T.size() ||
        T.Dmean.size() != T.size() || T.eps_mean.size() != T.size()) {
      throw std::invalid_argument(
        "mirror_tables_to_device: inconsistent node array sizes");
    }
    if (T.n_fine < 2 || T.n_fine > T.size()) {
      throw std::invalid_argument(
        "mirror_tables_to_device: n_fine = " + std::to_string(T.n_fine) +
        " inconsistent with " + std::to_string(T.size()) + " nodes");
    }
    HadronicTables D;
    D.gamma_th  = T.gamma_th;
    D.gamma_min = T.gamma_min;
    D.gamma_max = T.gamma_max;
    D.y_min     = T.y_min;
    D.y_max     = T.y_max;
    D.dy_fine   = 1.0 / static_cast<double>(T.nodes_per_decade_fine);
    D.dy_coarse = 1.0 / static_cast<double>(T.nodes_per_decade_coarse);
    D.n_fine    = static_cast<int>(T.n_fine);
    D.n         = static_cast<int>(T.size());

    const auto n = T.size();
    D.y          = array_t<double*> { "bh_y", n };
    D.gamma      = array_t<double*> { "bh_gamma", n };
    D.rate       = array_t<double*> { "bh_rate", n };
    D.log_rate   = array_t<double*> { "bh_log_rate", n };

    auto h_y  = Kokkos::create_mirror_view(D.y);
    auto h_g  = Kokkos::create_mirror_view(D.gamma);
    auto h_J  = Kokkos::create_mirror_view(D.rate);
    auto h_lJ = Kokkos::create_mirror_view(D.log_rate);
    for (std::size_t i = 0; i < n; ++i) {
      if (!(T.I[i] > 0.0) || !(T.J[i] > 0.0) || !std::isfinite(T.I[i]) ||
          !std::isfinite(T.J[i])) {
        throw std::invalid_argument(
          "mirror_tables_to_device: I or J not positive and finite at node " +
          std::to_string(i));
      }
      h_y(i)  = T.y[i];
      h_g(i)  = T.gamma[i];
      h_J(i)  = T.J[i];
      h_lJ(i) = std::log(T.J[i]);
      if (!(T.Dmean[i] > 0.0) || !std::isfinite(T.Dmean[i]) ||
          !(T.eps_mean[i] > 0.0) || !std::isfinite(T.eps_mean[i])) {
        throw std::invalid_argument("mirror_tables_to_device: <D> or eps_mean "
                                    "not positive and finite at node " +
                                    std::to_string(i));
      }
    }
    Kokkos::deep_copy(D.y, h_y);
    Kokkos::deep_copy(D.gamma, h_g);
    Kokkos::deep_copy(D.rate, h_J);
    Kokkos::deep_copy(D.log_rate, h_lJ);
    return D;
  }

} // namespace arch::bh

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_TABLES_DEVICE_H

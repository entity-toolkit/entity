/**
 * @file archetypes/emission/bethe_heitler/bh_born_qk.h
 * @brief Table of q_k(k), the mean momentum absorbed by the proton along the
 *        direction of the photon in its rest frame, from the Born cross
 *        section of Bethe-Heitler pair production
 * @implements
 *   - arch::bh::born::direct_quadrature -> rf::RestFrameQuadrature
 *   - arch::bh::born::direct -> double
 *   - arch::bh::born::CubicSpline
 *   - arch::bh::born::Table
 *   - arch::bh::born::table -> const Table&
 *   - arch::bh::born::hook_q_k_delta -> double
 * @namespaces:
 *   - arch::bh::born::
 *
 * @note
 * Host-only code that does not depend on Kokkos. Including this header
 * registers `hook_q_k_delta` in the hook of bh_physics.h.
 *
 * @note
 * With k the photon energy and E', p', theta' the energy, the momentum and
 * the angle to the photon of one lepton, all in the rest frame of the proton,
 *   q_k(k) = k - 2 <p' cos theta'> = 2 <(E' - p') + p' (1 - cos theta')>,
 * the average being taken with the angular law (Motz, Olsen & Koch 1969,
 * formula 3D-2000 = Blumenthal 1970, eq. 10; `rf::dsigma_dE_dcos_closed`)
 * normalised by its own integral. The quadrature is that of `rf::moments`:
 * panels in s for the energy, with E' = 1 + (k - 2) sin^2(pi s / 2), panels
 * in t = ln(1 - cos theta' + w) for the angle, Gauss-Legendre on each panel.
 * With the order of `direct_quadrature` the node values are accurate to
 * 1e-8.
 *
 * @note
 * The table:
 *   - nodes uniform in log10(k - 2), NODES_PER_DECADE per decade from
 *     k - 2 = DELTA_MIN to the upper end k_max of the range (the last node
 *     exactly at k_max), every node by direct quadrature, in parallel over
 *     the host threads;
 *   - between the nodes a not-a-knot cubic spline of ln q_k in
 *     t = ln(k - 2). The error of the spline is bounded by
 *     (5/384) h^4 max|f''''| with h = ln(10) / NODES_PER_DECADE and
 *     f = ln q_k(t), a smooth function with derivatives of order one. With
 *     64 nodes per decade the interpolant agrees with the direct quadrature
 *     to 5.2e-10 (relative) at the midpoints between the nodes;
 *   - below DELTA_MIN the linear continuation q_k = 2 - s (k - 2) with the
 *     slope s from the first node (q_k(2) = 2 exactly). Its error is of
 *     second order in k - 2: below 1e-10 (relative) at DELTA_MIN;
 *   - above k_max the value at k_max.
 * The table is built once per process, on first use, and the time this takes
 * is recorded (`Table::build_seconds`).
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_BORN_QK_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_BORN_QK_H

#include "archetypes/emission/bethe_heitler/bh_physics.h"
#include "archetypes/emission/bethe_heitler/bh_restframe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace arch::bh::born {

  /// grid: nodes per decade of k - 2, and k - 2 at the first node
  inline constexpr int    NODES_PER_DECADE = 64;
  inline constexpr double DELTA_MIN        = 1.0e-5;

  /**
   * @brief The quadrature of the direct evaluation of q_k: Gauss-Legendre of
   *        order 24 on 32 panels in lepton energy and on panels of width 0.25
   *        in t for the angle, for an accuracy of 1e-8
   */
  inline rf::RestFrameQuadrature direct_quadrature() {
    rf::RestFrameQuadrature q;
    q.gl_order      = 24;
    q.energy_panels = 32;
    q.h_t_angle     = 0.25;
    return q;
  }

  /**
   * @brief q_k(k) by direct quadrature of the angular law (no table)
   */
  inline double direct(double                         k,
                       const rf::RestFrameQuadrature& q = direct_quadrature()) {
    if (!(k > 2.0)) {
      return 2.0;
    }
    return rf::moments<double>(k, q).q_k;
  }

  /**
   * @brief Not-a-knot cubic spline on strictly increasing abscissae
   */
  class CubicSpline {
  public:
    CubicSpline() = default;

    CubicSpline(std::vector<double> x, std::vector<double> y)
      : m_x { std::move(x) }
      , m_y { std::move(y) } {
      const std::size_t n = m_x.size();
      if (n < 4 || m_y.size() != n) {
        throw std::invalid_argument("CubicSpline: need at least four nodes");
      }
      for (std::size_t i = 1; i < n; ++i) {
        if (!(m_x[i] > m_x[i - 1])) {
          throw std::invalid_argument("CubicSpline: abscissae must increase");
        }
      }
      // second derivatives M_i from the not-a-knot conditions (third derivative
      // continuous at x_1 and x_{n-2}); tridiagonal system solved by elimination
      std::vector<double> h(n - 1);
      for (std::size_t i = 0; i + 1 < n; ++i) {
        h[i] = m_x[i + 1] - m_x[i];
      }
      std::vector<double> a(n, 0.0), b(n, 0.0), c(n, 0.0), d(n, 0.0);
      // row 0: not-a-knot at x_1: h1 M0 - (h0 + h1) M1 + h0 M2 = 0
      // expressed through rows 0 and 1 to keep the system tridiagonal
      for (std::size_t i = 1; i + 1 < n; ++i) {
        a[i] = h[i - 1];
        b[i] = 2.0 * (h[i - 1] + h[i]);
        c[i] = h[i];
        d[i] = 6.0 *
               ((m_y[i + 1] - m_y[i]) / h[i] - (m_y[i] - m_y[i - 1]) / h[i - 1]);
      }
      // not-a-knot: M0 = ((h0 + h1) M1 - h0 M2) / h1 and
      // M_{n-1} = ((h_{n-2} + h_{n-3}) M_{n-2} - h_{n-2} M_{n-3}) / h_{n-3};
      // substitute into rows 1 and n-2
      {
        const double h0 = h[0], h1 = h[1];
        // row 1: a1 M0 + b1 M1 + c1 M2 = d1 with M0 = ((h0+h1) M1 - h0 M2)/h1
        b[1]            += a[1] * (h0 + h1) / h1;
        c[1]            -= a[1] * h0 / h1;
        a[1]             = 0.0;
        const double hm = h[n - 2], hp = h[n - 3];
        // row n-2: a M_{n-3} + b M_{n-2} + c M_{n-1} = d with
        // M_{n-1} = ((hm+hp) M_{n-2} - hm M_{n-3})/hp
        b[n - 2] += c[n - 2] * (hm + hp) / hp;
        a[n - 2] -= c[n - 2] * hm / hp;
        c[n - 2]  = 0.0;
      }
      // Thomas algorithm on rows 1..n-2
      std::vector<double> M(n, 0.0);
      {
        std::vector<double> cp(n, 0.0), dp(n, 0.0);
        cp[1] = c[1] / b[1];
        dp[1] = d[1] / b[1];
        for (std::size_t i = 2; i + 1 < n; ++i) {
          const double den = b[i] - a[i] * cp[i - 1];
          cp[i]            = c[i] / den;
          dp[i]            = (d[i] - a[i] * dp[i - 1]) / den;
        }
        M[n - 2] = dp[n - 2];
        for (std::size_t i = n - 3; i >= 1; --i) {
          M[i] = dp[i] - cp[i] * M[i + 1];
        }
        M[0]     = ((h[0] + h[1]) * M[1] - h[0] * M[2]) / h[1];
        M[n - 1] = ((h[n - 2] + h[n - 3]) * M[n - 2] - h[n - 2] * M[n - 3]) /
                   h[n - 3];
      }
      m_M = std::move(M);
    }

    double operator()(double x) const {
      const std::size_t n = m_x.size();
      if (x <= m_x.front()) {
        return m_y.front();
      }
      if (x >= m_x.back()) {
        return m_y.back();
      }
      std::size_t lo = 0, hi = n - 1;
      while (hi - lo > 1) {
        const std::size_t mid = (lo + hi) / 2;
        if (m_x[mid] <= x) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const double h = m_x[hi] - m_x[lo];
      const double A = (m_x[hi] - x) / h, B = (x - m_x[lo]) / h;
      return A * m_y[lo] + B * m_y[hi] +
             ((A * A * A - A) * m_M[lo] + (B * B * B - B) * m_M[hi]) * h * h / 6.0;
    }

    const std::vector<double>& x() const {
      return m_x;
    }

  private:
    std::vector<double> m_x, m_y, m_M;
  };

  /**
   * @brief The node table and its interpolant
   */
  class Table {
  public:
    Table() = default;

    /**
     * @brief Build the table: nodes delta_j = 10^(log10(DELTA_MIN) + j /
     * NODES_PER_DECADE) while 2 + delta_j < k_max, then k_max exactly; direct
     * quadrature at every node in parallel
     */
    explicit Table(double                         k_max,
                   const rf::RestFrameQuadrature& q = direct_quadrature(),
                   unsigned                       n_threads = 0) {
      if (!(k_max > 2.0 + DELTA_MIN)) {
        throw std::invalid_argument(
          "born::Table: k_max must exceed 2 + DELTA_MIN");
      }
      const auto   t0 = std::chrono::steady_clock::now();
      const double l0 = std::log10(DELTA_MIN);
      for (int j = 0;; ++j) {
        const double delta = std::pow(
          10.0,
          l0 + static_cast<double>(j) / NODES_PER_DECADE);
        if (2.0 + delta >= k_max * (1.0 - 1.0e-12)) {
          break;
        }
        m_k.push_back(2.0 + delta);
      }
      m_k.push_back(k_max);
      m_q.assign(m_k.size(), 0.0);
      std::vector<double>&       qv = m_q;
      const std::vector<double>& kv = m_k;
      rf::parallel_for_index(
        m_k.size(),
        [&qv, &kv, &q](std::size_t i) {
          qv[i] = direct(kv[i], q);
        },
        n_threads);
      std::vector<double> t(m_k.size()), lnq(m_k.size());
      for (std::size_t i = 0; i < m_k.size(); ++i) {
        if (!(m_q[i] > 0.0) || !std::isfinite(m_q[i])) {
          throw std::runtime_error(
            "born::Table: q_k not positive and finite at k = " +
            std::to_string(m_k[i]));
        }
        t[i]   = std::log(m_k[i] - 2.0);
        lnq[i] = std::log(m_q[i]);
      }
      m_spline        = CubicSpline(t, lnq);
      // continuation below the first node: q = 2 - s delta, s being the secant
      // from the exact value at threshold, q_k(2) = 2 (the pair at rest in the
      // frame of the proton), to the first node (delta = 1e-5):
      // s = 1/4 + O(delta), the slope at threshold being exactly 1/4
      m_slope         = (2.0 - m_q[0]) / (m_k[0] - 2.0);
      m_k_max         = k_max;
      m_build_seconds = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
    }

    /// q_k at k = 2 + delta
    double q_k_delta(double delta) const {
      if (!(delta > 0.0)) {
        return 2.0;
      }
      if (delta < m_k[0] - 2.0) {
        return 2.0 - m_slope * delta;
      }
      if (delta >= m_k_max - 2.0) {
        return m_q.back();
      }
      return std::exp(m_spline(std::log(delta)));
    }

    double q_k(double k) const {
      return q_k_delta(k - 2.0);
    }

    const std::vector<double>& nodes_k() const {
      return m_k;
    }

    const std::vector<double>& nodes_q() const {
      return m_q;
    }

    double threshold_slope() const {
      return m_slope;
    }

    double build_seconds() const {
      return m_build_seconds;
    }

    std::size_t size() const {
      return m_k.size();
    }

  private:
    std::vector<double> m_k, m_q;
    CubicSpline         m_spline;
    double              m_slope { 0.25 };
    double              m_k_max { constants::K_MAX };
    double              m_build_seconds { 0.0 };
  };

  /**
   * @brief The table of the process on [2, K_MAX], built on first use
   *        (thread-safe) and read by the cumulatives of bh_physics.h
   * @param n_threads host threads of the build (0: `rf::thread_limit()`);
   *        it matters only in the call that builds the table
   */
  inline const Table& table(unsigned n_threads = 0) {
    static const Table T(constants::K_MAX, direct_quadrature(), n_threads);
    return T;
  }

} // namespace arch::bh::born

namespace arch::bh::born {

  /// the function registered in the hook of bh_physics.h
  inline double hook_q_k_delta(double delta) {
    return table().q_k_delta(delta);
  }

  /// registration at static initialisation of every translation unit that
  /// includes this header
  inline const bool registered = (born_qk_hook() = &hook_q_k_delta, true);

} // namespace arch::bh::born

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_BORN_QK_H

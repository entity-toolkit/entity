/**
 * @file archetypes/emission/bethe_heitler/bh_sampler_host.h
 * @brief Host tables from which the outcome of a Bethe-Heitler interaction
 *        is sampled: the energy of the absorbed photon, and the energies and
 *        angles of the two leptons in the rest frame of the proton
 * @implements
 *   - arch::bh::rf::SamplerCumulatives
 *   - arch::bh::rf::RestFrameTables
 *   - arch::bh::rf::BathEpsTables
 * @namespaces:
 *   - arch::bh::rf::
 *
 * @note
 * Host-only code in double precision that does not depend on Kokkos. The
 * device views of these tables and the sampling itself are in
 * bh_sampler_device.h.
 *
 * @note
 * The outcome of an interaction of a proton of Lorentz factor Gamma is drawn
 * in this order (eps is the photon energy in the simulation frame, k = eps'
 * its energy in the rest frame of the proton, E' and theta' the energy of a
 * lepton and its angle to the photon in that frame):
 *   - eps, with a density proportional to n~(eps) / eps^2 Phi(k_hi), where
 *     k_hi = Gamma eps (1 + beta) and Phi(x) = int_2^x k sigma dk
 *     (`BathEpsTables`; a monochromatic bath has a single energy);
 *   - k in [2, k_hi], with a density proportional to k sigma(k)
 *     (`SamplerCumulatives`);
 *   - the energy of the electron from the energy law, then the angle of each
 *     lepton to the photon from the angular law at its energy
 *     (`RestFrameTables`); the two angles are drawn independently.
 *
 * @note
 * Grids:
 *   - nodes in k: uniform in log10(k - 2) from k - 2 = 1e-4 to the upper end
 *     k = 1000 of the tables, `nodes_per_decade` (16) per decade, the end
 *     point appended;
 *   - lepton energy on s in [0, 1] with E' = 1 + (k - 2) sin^2(pi s / 2),
 *     `n_s` (256) cells;
 *   - lepton angle on tau in [0, 1] with t = ln(1 - cos theta' + w) =
 *     ln w + tau (ln(2 + w) - ln w), where w = (E' - p') / p' is the scale of
 *     the forward peak, `n_tau` (256) cells;
 *   - the CDFs are stored at the cell edges. Sampling inverts the mixture of
 *     the CDFs of the bracketing nodes (weights linear in ln(k - 2) and in
 *     s) by bisection over the edges and linear interpolation inside a cell:
 *     the sampled density is piecewise constant in s and in tau;
 *   - ln Phi on a fine grid in t = ln(k - 2), of spacing 0.005 from
 *     k - 2 = 1e-8 to 998, interpolated linearly in t;
 *   - the CDF of eps of a broad bath at every Gamma node of the rate table,
 *     each on its own grid uniform in
 *     x = ln(eps / eps_lo) / ln(eps_max / eps_lo), with
 *     eps_lo = max(eps_min, 2 / (Gamma (1 + beta))), of at least 2048 cells
 *     at most 1e-3 wide in ln(eps). Between two nodes the CDFs are mixed at
 *     fixed x with weights linear in y = log10(Gamma - Gamma_th).
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_SAMPLER_HOST_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_SAMPLER_HOST_H

#include "archetypes/emission/bethe_heitler/bh_physics.h"
#include "archetypes/emission/bethe_heitler/bh_restframe.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace arch::bh::rf {

  // ------------------------------------------------------------------------ //
  // Phi(x) = int_2^x k sigma dk on a fine grid in t = ln(k - 2), from which
  // the rest-frame photon energy is drawn: ln Phi is stored at the nodes and
  // interpolated linearly in t
  // ------------------------------------------------------------------------ //
  class SamplerCumulatives {
  public:
    explicit SamplerCumulatives(double h      = 0.005,
                                double delta0 = 1.0e-8,
                                double kmax   = constants::K_MAX,
                                int    gl     = 16)
      : m_h { h }
      , m_delta0 { delta0 }
      , m_kmax { kmax } {
      if (!(h > 0.0) || !(delta0 > 0.0) || !(kmax > 4.0)) {
        throw std::invalid_argument("SamplerCumulatives: bad grid parameters");
      }
      const GaussLegendre G(gl);
      const double        t0 = std::log(delta0), t1 = std::log(kmax - 2.0);
      const auto n = static_cast<std::size_t>(std::ceil((t1 - t0) / h));
      m_t.resize(n + 1);
      for (std::size_t i = 0; i <= n; ++i) {
        m_t[i] = t0 + (t1 - t0) * static_cast<double>(i) / static_cast<double>(n);
      }
      // threshold law at delta0: k sigma -> (pi/6) d^3, Phi -> (pi/24) d^4
      double Phi = (constants::PI / 24.0) * std::pow(delta0, 4.0);
      m_lnPhi.resize(n + 1);
      m_lnPhi[0] = std::log(Phi);
      for (std::size_t i = 1; i <= n; ++i) {
        const double a = m_t[i - 1], b = m_t[i];
        const double mid = 0.5 * (a + b), half = 0.5 * (b - a);
        double       c1 = 0.0;
        for (int g = 0; g < G.order(); ++g) {
          const double t      = mid + half * G.x[g];
          const double delta  = std::exp(t);
          const double k      = 2.0 + delta;
          const double s      = sigma_tilde_delta(delta);
          const double wgt    = half * G.w[g] * delta; // dk = delta dt
          c1                 += wgt * k * s;
        }
        Phi        += c1;
        m_lnPhi[i]  = std::log(Phi);
      }
    }

    double h() const {
      return m_h;
    }

    double delta0() const {
      return m_delta0;
    }

    double kmax() const {
      return m_kmax;
    }

    std::size_t size() const {
      return m_t.size();
    }

    const std::vector<double>& t() const {
      return m_t;
    }

    const std::vector<double>& lnPhi() const {
      return m_lnPhi;
    }

    double Phi(double delta) const {
      return eval(m_lnPhi, delta, 4.0);
    }

  private:
    double eval(const std::vector<double>& ln, double delta, double power) const {
      if (!(delta > 0.0)) {
        return 0.0;
      }
      if (delta <= m_delta0) {
        return std::exp(ln.front()) * std::pow(delta / m_delta0, power);
      }
      const double dmax = m_kmax - 2.0;
      if (delta >= dmax) {
        return std::exp(ln.back());
      }
      const double t  = std::log(delta);
      const auto   it = std::upper_bound(m_t.begin(), m_t.end(), t);
      std::size_t  j  = static_cast<std::size_t>(it - m_t.begin());
      j = std::min(std::max<std::size_t>(j, 1), m_t.size() - 1) - 1;
      const double f = (t - m_t[j]) / (m_t[j + 1] - m_t[j]);
      return std::exp(ln[j] + f * (ln[j + 1] - ln[j]));
    }

    double              m_h, m_delta0, m_kmax;
    std::vector<double> m_t, m_lnPhi;
  };

  // ------------------------------------------------------------------------ //
  // The CDF tables of the energy law and of the angular law
  // ------------------------------------------------------------------------ //
  struct RestFrameTables {
    // grid
    std::vector<double> t_k; // ln(k - 2) at the nodes
    int                 n_s { 256 };
    int                 n_tau { 256 };
    int                 nodes_per_decade { 16 };
    // CDFs at the edges: cdf_E[ik][js], cdf_A[ik][js][jt]
    std::vector<double> cdf_E;
    std::vector<double> cdf_A;

    std::size_t n_k() const {
      return t_k.size();
    }

    // lepton energy at the position s in [0, 1]
    static double E_of_s(double k, double s) {
      const double sn = std::sin(0.5 * constants::PI * s);
      return 1.0 + (k - 2.0) * sn * sn;
    }

    /// the angular law used for the tables
    double law(double k, double E, double omc) const {
      return dsigma_dE_dcos_closed(k, E, omc);
    }

    /**
     * @brief Build the tables (threads over the nodes in k)
     * @param nodes_per_decade nodes per decade of k - 2
     * @param n_s cells in lepton energy
     * @param n_tau cells in lepton angle
     * @param gl_cell Gauss-Legendre order per cell
     * @param n_threads host threads (0: as many as `thread_limit()` allows)
     */
    static RestFrameTables build(int      nodes_per_decade = 16,
                                 int      n_s              = 256,
                                 int      n_tau            = 256,
                                 int      gl_cell          = 8,
                                 unsigned n_threads        = 0) {
      RestFrameTables T;
      T.n_s              = n_s;
      T.n_tau            = n_tau;
      T.nodes_per_decade = nodes_per_decade;
      for (int i = 0;; ++i) {
        const double delta = std::pow(
          10.0,
          -4.0 + static_cast<double>(i) / nodes_per_decade);
        if (2.0 + delta >= constants::K_MAX) {
          break;
        }
        T.t_k.push_back(std::log(delta));
      }
      T.t_k.push_back(std::log(constants::K_MAX - 2.0));
      const std::size_t nk = T.t_k.size();
      T.cdf_E.assign(nk * (n_s + 1), 0.0);
      T.cdf_A.assign(nk * (n_s + 1) * (n_tau + 1), 0.0);
      const GaussLegendre G(gl_cell);
      parallel_for_index(
        nk,
        [&](std::size_t ik) {
          const double k  = 2.0 + std::exp(T.t_k[ik]);
          double*      CE = T.cdf_E.data() + ik * (n_s + 1);
          // energy CDF on the s cells
          const double hs = 1.0 / n_s;
          CE[0]           = 0.0;
          for (int j = 0; j < n_s; ++j) {
            double cell = 0.0;
            for (int g = 0; g < G.order(); ++g) {
              const double s   = (j + 0.5 * (1.0 + G.x[g])) * hs;
              const double E   = E_of_s(k, s);
              const double jac = (k - 2.0) * 0.5 * constants::PI *
                                 std::sin(constants::PI * s);
              cell += 0.5 * hs * G.w[g] * jac * dsigma_dE(k, E);
            }
            CE[j + 1] = CE[j] + cell;
          }
          const double tot = CE[n_s];
          for (int j = 0; j <= n_s; ++j) {
            CE[j] = tot > 0.0 ? std::min(1.0, CE[j] / tot)
                              : static_cast<double>(j) / n_s;
          }
          CE[n_s]         = 1.0;
          // angular CDFs at the s nodes (end nodes evaluated half a cell inside)
          const double ht = 1.0 / n_tau;
          for (int j = 0; j <= n_s; ++j) {
            const double s_eval = std::min(
              1.0 - 0.5 * hs,
              std::max(0.5 * hs, static_cast<double>(j) * hs));
            const double E   = E_of_s(k, s_eval);
            const double wsc = angular_scale(E);
            const double ta = std::log(wsc), tb = std::log(2.0 + wsc);
            double* CA = T.cdf_A.data() + (ik * (n_s + 1) + j) * (n_tau + 1);
            CA[0]      = 0.0;
            for (int l = 0; l < n_tau; ++l) {
              double cell = 0.0;
              for (int g = 0; g < G.order(); ++g) {
                const double tau = (l + 0.5 * (1.0 + G.x[g])) * ht;
                const double t   = ta + tau * (tb - ta);
                const double om = std::min(2.0, std::max(0.0, std::exp(t) - wsc));
                cell += 0.5 * ht * G.w[g] * (tb - ta) * std::exp(t) *
                        T.law(k, E, om);
              }
              CA[l + 1] = CA[l] + std::max(0.0, cell);
            }
            const double totA = CA[n_tau];
            for (int l = 0; l <= n_tau; ++l) {
              CA[l] = totA > 0.0 ? std::min(1.0, CA[l] / totA)
                                 : static_cast<double>(l) / n_tau;
            }
            CA[n_tau] = 1.0;
          }
        },
        n_threads);
      return T;
    }
  };

  // ------------------------------------------------------------------------ //
  // The photon-energy tables of a broad bath at the Gamma nodes
  // ------------------------------------------------------------------------ //
  /**
   * @brief CDF of the energy eps of the photon absorbed by a proton of
   *        Lorentz factor Gamma in a broad bath, at a list of Gamma nodes
   *
   * @note
   * Each Gamma node has its own grid, adapted to the interval of photon
   * energies above the threshold at that Gamma, [eps_lo(Gamma), eps_max] with
   * eps_lo(Gamma) = max(eps_min of the bath, 2 / (Gamma (1 + beta))), uniform
   * in x = ln(eps / eps_lo) / ln(eps_max / eps_lo) in [0, 1] (`n_eps` cells).
   * A single grid in ln(eps) common to all the nodes would cover this
   * interval with a few cells near the threshold, the threshold falling
   * inside a cell, and bias <eps> by up to 3.5e-5. At a Gamma between two
   * nodes their CDFs are mixed at fixed x (weights linear in
   * y = log10(Gamma - Gamma_th)) and x is mapped to eps with eps_lo of that
   * Gamma, so that the threshold is respected at every Gamma and not only at
   * the nodes. Inside a cell the sampled density is uniform in x, that is in
   * ln(eps). The bias of <eps> that this leaves is of second order in the
   * width h of the cell in ln(eps) (2.9e-7 at h = 1.9e-3), so the number of
   * cells is max(`n_eps`, ceil(widest range of a node / `h_max`)), with
   * h_max = 1e-3 by default: the width of the cells, not their number, is
   * held fixed when the range of Gamma grows.
   *
   * @note
   * The mixture at fixed x assumes that the shapes at the two bracketing
   * nodes are related by the rescaling of [eps_lo(Gamma), eps_max], which
   * holds where eps_lo is smooth. eps_lo has a kink at Gamma_c, where
   * 2 / (Gamma (1 + beta)) = eps_min: below it the interval starts at the
   * threshold, where the density vanishes, above it at the lower edge of the
   * bath. In a cell in Gamma that contains Gamma_c the sampled CDF would be
   * wrong by up to 1.7e-3 (for a power law on [0.1, 1.5]), so `build_at`
   * adds Gamma_c as a node for every broad bath with eps_min > 0 when it
   * falls strictly between two of the nodes given. These tables therefore
   * have their own list of nodes: `gamma` and `y` are the list given plus
   * Gamma_c, and an index into them is not an index of the rate table.
   */
  struct BathEpsTables {
    bool   mono { true };
    double eps0 { 0.0 };
    double gamma_th { 0.0 };
    double eps_min_bath { 0.0 }; // the bath's lower edge (0 if none)
    double ln_eps_hi { 0.0 };    // ln eps_max
    std::vector<double> y;       // log10(Gamma - Gamma_th) at the nodes
    std::vector<double> gamma; // the nodes: the list given to `build_at` plus Gamma_c (see above)
    int n_eps { 2048 };
    std::vector<double> cdf_u; // [n_gamma][n_eps + 1] CDF of eps at x_j = j/n_eps

    /// ln eps_lo(Gamma) = ln max(eps_min, 2/(Gamma (1 + beta)))
    static double ln_eps_lo_of(double g, double eps_min_bath) {
      const double A = g + std::sqrt((g - 1.0) * (g + 1.0));
      return std::log(std::max(eps_min_bath, 2.0 / A));
    }

    double ln_eps_lo(double g) const {
      return ln_eps_lo_of(g, eps_min_bath);
    }

    /// the Gamma of the kink of eps_lo: 2/A = eps_min with A = Gamma (1 +
    /// beta), Gamma = (A + 1/A)/2; 0 without a lower edge
    static double gamma_kink(double eps_min_bath) {
      if (!(eps_min_bath > 0.0)) {
        return 0.0;
      }
      const double A = 2.0 / eps_min_bath;
      return 0.5 * (A + 1.0 / A);
    }

    /**
     * @brief Build the tables at the Gamma nodes of the rate table H
     * @param n_eps smallest number of cells in photon energy
     * @param gl_cell Gauss-Legendre order per cell
     * @param n_threads host threads (0: as many as `thread_limit()` allows)
     * @param h_max largest width of a cell in ln(eps) (0: no bound)
     */
    static BathEpsTables build(const PhotonBath&         bath,
                               const HadronicTablesHost& H,
                               const SamplerCumulatives& cum,
                               int                       n_eps     = 2048,
                               int                       gl_cell   = 8,
                               unsigned                  n_threads = 0,
                               double                    h_max     = 1.0e-3) {
      return build_at(bath, H.gamma, cum, n_eps, gl_cell, n_threads, h_max);
    }

    /// the same at an explicit, increasing list of Gamma nodes (> Gamma_th),
    /// plus Gamma_c when it falls strictly between two of them (a Gamma_c
    /// outside the range of the list, or on a node, leaves no cell that
    /// contains the kink)
    static BathEpsTables build_at(const PhotonBath&          bath,
                                  const std::vector<double>& nodes,
                                  const SamplerCumulatives&  cum,
                                  int                        n_eps     = 2048,
                                  int                        gl_cell   = 8,
                                  unsigned                   n_threads = 0,
                                  double                     h_max = 1.0e-3) {
      BathEpsTables B;
      B.gamma_th = bath.gamma_th();
      B.gamma    = nodes;
      B.mono     = bath.is_mono();
      if (B.mono) {
        B.eps0 = bath.eps0();
        return B;
      }
      if (nodes.size() < 2) {
        throw std::invalid_argument(
          "BathEpsTables: at least two Gamma nodes are needed");
      }
      const double g_c = gamma_kink(std::max(0.0, bath.eps_min()));
      const auto   up  = std::upper_bound(B.gamma.begin(), B.gamma.end(), g_c);
      if (g_c > 0.0 && up != B.gamma.begin() && up != B.gamma.end() &&
          g_c > *(up - 1) * (1.0 + 1.0e-12) && g_c < *up * (1.0 - 1.0e-12)) {
        B.gamma.insert(up, g_c);
      }
      const std::vector<double>& gammas = B.gamma;
      B.y.resize(gammas.size());
      for (std::size_t i = 0; i < gammas.size(); ++i) {
        if (!(gammas[i] > B.gamma_th) || (i > 0 && !(gammas[i] > gammas[i - 1]))) {
          throw std::invalid_argument(
            "BathEpsTables: Gamma nodes must be increasing and above Gamma_th");
        }
        B.y[i] = std::log10(gammas[i] - B.gamma_th);
      }
      B.eps_min_bath = std::max(0.0, bath.eps_min());
      B.ln_eps_hi    = std::log(bath.eps_max());
      double widest  = 0.0;
      for (const double g : gammas) {
        widest = std::max(widest, B.ln_eps_hi - B.ln_eps_lo(g));
      }
      if (h_max > 0.0) {
        n_eps = std::max(n_eps, static_cast<int>(std::ceil(widest / h_max)));
      }
      B.n_eps              = n_eps;
      const std::size_t ng = gammas.size();
      B.cdf_u.assign(ng * (n_eps + 1), 0.0);
      const GaussLegendre G(gl_cell);
      parallel_for_index(
        ng,
        [&](std::size_t ig) {
          const double g  = gammas[ig];
          const double gb = std::sqrt((g - 1.0) * (g + 1.0));
          const double A  = g + gb;
          const double lo = B.ln_eps_lo(g);
          if (!(B.ln_eps_hi > lo)) {
            throw std::invalid_argument(
              "BathEpsTables: empty photon-energy range at Gamma " +
              std::to_string(g));
          }
          const double h  = (B.ln_eps_hi - lo) / n_eps;
          double*      Cu = B.cdf_u.data() + ig * (n_eps + 1);
          Cu[0]           = 0.0;
          for (int j = 0; j < n_eps; ++j) {
            double       cu  = 0.0;
            const double a   = lo + h * j;
            const double mid = a + 0.5 * h, half = 0.5 * h;
            for (int q = 0; q < G.order(); ++q) {
              const double eps = std::exp(mid + half * G.x[q]);
              const double n   = bath.density(eps);
              const double dhi = eps * A - 2.0;
              if (!(n > 0.0) || !(dhi > 0.0)) {
                continue;
              }
              const double w = half * G.w[q] * n /
                               eps; // d eps / eps^2 = d(ln eps) / eps
              cu += w * cum.Phi(dhi);
            }
            Cu[j + 1] = Cu[j] + cu;
          }
          for (int j = 0; j <= n_eps; ++j) {
            Cu[j] = Cu[n_eps] > 0.0 ? std::min(1.0, Cu[j] / Cu[n_eps])
                                    : static_cast<double>(j) / n_eps;
          }
          Cu[n_eps] = 1.0;
        },
        n_threads);
      return B;
    }
  };

} // namespace arch::bh::rf

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_SAMPLER_HOST_H

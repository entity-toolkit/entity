/**
 * @file archetypes/emission/bethe_heitler/bh_restframe.h
 * @brief Differential cross sections of Bethe-Heitler pair production on a
 *        static point charge (Z = 1, unscreened, Born approximation) in the
 *        rest frame of the charge, and their moments
 * @implements
 *   - arch::bh::rf::momentum -> double
 *   - arch::bh::rf::energy_minus_momentum -> double
 *   - arch::bh::rf::dsigma_dE_T<> -> T
 *   - arch::bh::rf::dsigma_dE -> double
 *   - arch::bh::rf::dsigma_dE_dcos_T<> -> T
 *   - arch::bh::rf::dsigma_dE_dcos_closed -> double
 *   - arch::bh::rf::RestFrameQuadrature
 *   - arch::bh::rf::integrate_energy<> -> double
 *   - arch::bh::rf::integrate_angle<> -> double
 *   - arch::bh::rf::Moments
 *   - arch::bh::rf::moments<> -> Moments
 *   - arch::bh::rf::parallel_for_index<> -> void
 * @namespaces:
 *   - arch::bh::rf::
 *   - arch::bh::rf::stable::
 *
 * @note
 * Host-only code that does not depend on Kokkos. The cross sections are
 * templated on the floating type and used in double precision.
 *
 * @note
 * Units: energies in m_e c^2, momenta in m_e c, cross sections in
 * alpha r_e^2. All quantities are in the rest frame of the proton: k is the
 * photon energy, E+ and E- are the total energies of the positron and of the
 * electron (E+ + E- = k: the proton takes momentum but no energy),
 * p = sqrt(E^2 - 1), theta is the angle of a lepton to the direction of the
 * photon.
 *
 * @note
 * Cross sections:
 *   - the energy law dsigma/dE+: the cross section integrated over the
 *     directions of both leptons, Motz, Olsen & Koch 1969, Rev. Mod. Phys.
 *     41, 581, formula 3D-1000 (Heitler 1954, The Quantum Theory of
 *     Radiation, p. 258, eq. 8). Its integral over E+ is the total cross
 *     section, of which `sigma_tilde(k)` of bh_physics.h is a series
 *     representation;
 *   - the angular law d^2 sigma/(dE+ dcos theta+): the cross section
 *     integrated over the direction of the other lepton, Motz, Olsen & Koch
 *     1969, formula 3D-2000, which is eq. (10) of Blumenthal 1970, Phys. Rev.
 *     D 1, 1596 (with the symbols of his eq. 11). By the symmetry of the Born
 *     cross section under the exchange of the two leptons the same law holds
 *     for the electron with E+ -> E-.
 *
 * @note
 * `moments` gives the mean momentum absorbed by the proton along the photon
 * direction, q_k(k) = k - 2 <p cos theta>, by Gauss-Legendre quadrature of
 * the angular law over E+ and cos theta+.
 *
 * @note
 * Both closed forms are sums of terms of order 1/p^4 and 1/p^2 that cancel to
 * a result of order p^2 when a lepton is slow. They are evaluated with a
 * representation that is stable in double precision (namespace `stable`):
 * power series near the threshold, a Taylor series computed at run time when
 * one lepton is slow, and the closed forms written without the logarithm of a
 * number close to 1 elsewhere.
 */

#ifndef ARCHETYPES_EMISSION_BETHE_HEITLER_BH_RESTFRAME_H
#define ARCHETYPES_EMISSION_BETHE_HEITLER_BH_RESTFRAME_H

#include "archetypes/emission/bethe_heitler/bh_physics.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <exception>
#include <functional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(__linux__)
  #include <sched.h>
#endif

namespace arch::bh::rf {

  // ------------------------------------------------------------------------ //
  // Kinematic helpers
  // ------------------------------------------------------------------------ //

  /// p = sqrt(E^2 - 1) without cancellation
  inline double momentum(double E) {
    return E > 1.0 ? std::sqrt((E - 1.0) * (E + 1.0)) : 0.0;
  }

  /// E - p = 1 / (E + p)
  inline double energy_minus_momentum(double E, double p) {
    return 1.0 / (E + p);
  }

  // ------------------------------------------------------------------------ //
  // Stable evaluation of the two closed forms
  //
  // The energy law and the angular law are sums of terms of order 1/p^4 and
  // 1/p^2 that cancel to a result of order p^2 as the lepton momenta go to
  // zero: the relative cancellation is of order p^6 at threshold (both
  // momenta small) and of order p+^2 or worse at an end point of the energy
  // range (one momentum small). Three regions of (p+, p-) are distinguished:
  //
  //  - threshold region, both momenta <= P_SW = 0.15: power series with
  //    coefficients computed in exact rational arithmetic
  //    (bh_threshold_series.inc). The energy law is a series B(a, b) in
  //    a = p+^2 and b = p-^2 of total degree 16 (152 coefficients; its
  //    logarithm is 2 artanh(p+ p- / (1 + E+ E-)), odd in each momentum, so
  //    that B depends on the momenta through a and b only). The angular law
  //    is a series K(p+, p-^2, cos theta) of terms p+^i (p-^2)^j cos^l theta
  //    with i + 2 j <= 20 (503 coefficients). The terms that the cancellation
  //    removes vanish identically in rational arithmetic. The relative
  //    truncation error is at most about 1e-14 at p = P_SW;
  //  - slow region, the momentum that enters as 1/p^2 (p+ for the angular
  //    law, the smaller momentum for the energy law) below P_SLOW = 0.1 and
  //    the other above P_SW: a truncated Taylor series in that momentum,
  //    computed at run time (`Tps`, 20 terms). The coefficients of orders 0
  //    and 1, which the cancellation sets to zero, are dropped exactly; the
  //    others are computed from quantities of order one. The truncation error
  //    is at most P_SLOW^18;
  //  - general region: the closed forms with sub-expressions that involve no
  //    logarithm of a number close to 1: eps = 2 asinh(p),
  //    L = 2 artanh(p+ p- / (1 + E+ E-)), eps^Q = 2 artanh(p- / Q). The
  //    rounding error is amplified by at most about
  //    1 / (P_SLOW^2 P_SW^2) ~ 4e3: the relative error is below 2e-11.
  //
  // Every region is evaluated in the template type T.
  // ------------------------------------------------------------------------ //
  namespace stable {

    inline constexpr double P_SW = 0.15; // threshold region: both momenta <= P_SW
    inline constexpr double P_SLOW = 0.1; // slow region: the critical momentum < P_SLOW
    inline constexpr int TPS_N = 20; // terms of the run-time Taylor series

#include "archetypes/emission/bethe_heitler/bh_threshold_series.inc"

    enum class Region : int {
      Threshold = 0,
      Slow      = 1,
      General   = 2
    };

    /// truncated univariate power series sum_{n<N} c[n] u^n in the type T
    template <class T, int N>
    struct Tps {
      T c[N];

      Tps() {
        for (int n = 0; n < N; ++n) {
          c[n] = T(0);
        }
      }

      explicit Tps(T v) : Tps() {
        c[0] = v;
      }

      static Tps var() {
        Tps r;
        if (N > 1) {
          r.c[1] = T(1);
        }
        return r;
      }

      friend Tps operator+(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] + y.c[n];
        }
        return r;
      }

      friend Tps operator-(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] - y.c[n];
        }
        return r;
      }

      friend Tps operator-(const Tps& x) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = -x.c[n];
        }
        return r;
      }

      friend Tps operator+(const Tps& x, T v) {
        Tps r   = x;
        r.c[0] += v;
        return r;
      }

      friend Tps operator-(const Tps& x, T v) {
        Tps r   = x;
        r.c[0] -= v;
        return r;
      }

      friend Tps operator*(const Tps& x, T v) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          r.c[n] = x.c[n] * v;
        }
        return r;
      }

      friend Tps operator*(const Tps& x, const Tps& y) {
        Tps r;
        for (int n = 0; n < N; ++n) {
          T s = T(0);
          for (int m = 0; m <= n; ++m) {
            s += x.c[m] * y.c[n - m];
          }
          r.c[n] = s;
        }
        return r;
      }
    };

    template <class T, int N>
    inline Tps<T, N> recip(const Tps<T, N>& x) {
      Tps<T, N> r;
      const T   r0 = T(1) / x.c[0];
      r.c[0]       = r0;
      for (int n = 1; n < N; ++n) {
        T s = T(0);
        for (int m = 1; m <= n; ++m) {
          s += x.c[m] * r.c[n - m];
        }
        r.c[n] = -r0 * s;
      }
      return r;
    }

    template <class T, int N>
    inline Tps<T, N> sqrt(const Tps<T, N>& x) {
      Tps<T, N> r;
      r.c[0] = std::sqrt(x.c[0]);
      for (int n = 1; n < N; ++n) {
        T s = x.c[n];
        for (int m = 1; m < n; ++m) {
          s -= r.c[m] * r.c[n - m];
        }
        r.c[n] = s / (T(2) * r.c[0]);
      }
      return r;
    }

    /// artanh of a series with |x0| < 1: f' = x'/(1 - x^2)
    template <class T, int N>
    inline Tps<T, N> atanh(const Tps<T, N>& x) {
      Tps<T, N> d;
      for (int n = 0; n + 1 < N; ++n) {
        d.c[n] = T(n + 1) * x.c[n + 1];
      }
      const Tps<T, N> q = d * recip(Tps<T, N>(T(1)) - x * x);
      Tps<T, N>       r;
      r.c[0] = std::atanh(x.c[0]);
      for (int n = 1; n < N; ++n) {
        r.c[n] = q.c[n - 1] / T(n);
      }
      return r;
    }

    /// sum_n coef(n) x^n for a series x without constant term (Horner)
    template <class T, int N, class C>
    inline Tps<T, N> compose(const Tps<T, N>& x, C&& coef) {
      Tps<T, N> r(coef(N - 1));
      for (int n = N - 2; n >= 0; --n) {
        r = r * x + coef(n);
      }
      return r;
    }

    /// coefficient of x^n in asinh(sqrt(x))/sqrt(x): (-1)^n (2n-1)!!/((2n)!! (2n+1))
    template <class T>
    inline T asinh_over_coef(int n) {
      T c = T(1);
      for (int m = 1; m <= n; ++m) {
        c *= -T(2 * m - 1) / T(2 * m);
      }
      return c / T(2 * n + 1);
    }

    /// 2 asinh(p)/p for p >= 0 (p^2 = a), stable at p -> 0
    template <class T>
    inline T two_asinh_over(T p, T a) {
      if (p < T(0.05)) {
        T s = T(0);
        for (int n = 12; n >= 0; --n) {
          s = s * a + asinh_over_coef<T>(n);
        }
        return T(2) * s;
      }
      return T(2) * std::asinh(p) / p;
    }

    /// 2 artanh(t)/t for |t| < 1, stable at t -> 0
    template <class T>
    inline T two_atanh_over(T t) {
      const T t2 = t * t;
      if (t2 < T(0.0025)) {
        T s = T(0);
        for (int n = 14; n >= 0; --n) {
          s = s * t2 + T(1) / T(2 * n + 1);
        }
        return T(2) * s;
      }
      return T(2) * std::atanh(t) / t;
    }

    // ---------------------------------------------------------------------- //
    // the threshold series
    // ---------------------------------------------------------------------- //

    /// B(a, b) of the energy law (both momenta <= P_SW)
    template <class T>
    inline T series_B(T a, T b) {
      T s = T(0);
      for (int i = BS_DEG; i >= 0; --i) {
        T r = T(0);
        for (int j = BS_DEG - i; j >= 0; --j) {
          r = r * b + static_cast<T>(BS_C[i][j]);
        }
        s = s * a + r;
      }
      return s;
    }

    /// K(p+, b = p-^2, c = cos theta) of the angular law (both momenta <= P_SW)
    template <class T>
    inline T series_K(T p, T b, T c) {
      T pw_p[KS_MAXI + 1], pw_b[KS_MAXJ + 1], pw_c[KS_MAXL + 1];
      pw_p[0] = pw_b[0] = pw_c[0] = T(1);
      for (int n = 1; n <= KS_MAXI; ++n) {
        pw_p[n] = pw_p[n - 1] * p;
      }
      for (int n = 1; n <= KS_MAXJ; ++n) {
        pw_b[n] = pw_b[n - 1] * b;
      }
      for (int n = 1; n <= KS_MAXL; ++n) {
        pw_c[n] = pw_c[n - 1] * c;
      }
      // highest weights first (smallest terms first)
      T s = T(0);
      for (int n = KS_N - 1; n >= 0; --n) {
        s += static_cast<T>(KS_C[n]) * pw_p[KS_I[n]] * pw_b[KS_J[n]] *
             pw_c[KS_L[n]];
      }
      return s;
    }

    // ---------------------------------------------------------------------- //
    // the energy law: the bracket times a b in the general region, and the
    // series for a slow lepton
    // ---------------------------------------------------------------------- //

    /**
     * @brief Q = B a b of the energy law with stable sub-expressions:
     *        Q = -(4/3) a b - 2 E+E- (a + b) + E- R+ b + E+ R- a - R+ R- a b +
     * M N, R = eps/p = 2 asinh(p)/p, M = L/(p+ p-) = 2 artanh(t)/t /(1 + E+E-)
     *        with t = p+ p-/(1 + E+E-), N = k^2 (E+^2 E-^2 + a b - E+E-) -
     * (8/3) E+E- a b - (k/2) [(E+E- - a) R+ b + (E+E- - b) R- a]
     */
    template <class T>
    inline T energy_Q_general(T Ep, T Em, T pp, T pm, T a, T b) {
      const T k  = Ep + Em;
      const T EE = Ep * Em;
      const T Rp = two_asinh_over(pp, a);
      const T Rm = two_asinh_over(pm, b);
      const T i1 = T(1) / (T(1) + EE);
      const T M  = two_atanh_over(pp * pm * i1) * i1;
      const T ab = a * b;
      const T Nn = k * k * (EE * EE + ab - EE) - (T(8) / T(3)) * EE * ab -
                   T(0.5) * k * ((EE - a) * Rp * b + (EE - b) * Rm * a);
      return -(T(4) / T(3)) * ab - T(2) * EE * (a + b) + Em * Rp * b +
             Ep * Rm * a - Rp * Rm * ab + M * Nn;
    }

    /**
     * @brief B of the energy law for p+ < P_SLOW (p+ the smaller momentum): the
     *        same Q as a series in u = p+, orders 0 and 1 dropped (zero
     *        identically: B is finite at p+ = 0), B = sum_{n>=2} Q_n p+^(n-2) / b
     */
    template <class T>
    inline T energy_B_slow(T pp, T pm, T Em, T b) {
      using X    = Tps<T, TPS_N>;
      const X u  = X::var();
      const X a  = u * u;
      const X Ep = sqrt(a + T(1));
      const X Rp = compose(a, [](int n) {
        return T(2) * asinh_over_coef<T>(n);
      });
      const T Rm = two_asinh_over(pm, b);
      const X EE = Ep * Em;
      const X k  = Ep + Em;
      const X i1 = recip(EE + T(1));
      const X t2 = a * i1 * i1 * b;
      const X M  = compose(t2,
                          [](int n) {
                            return T(1) / T(2 * n + 1);
                          }) *
                  i1 * T(2);
      const X ab = a * b;
      const X Nn = k * k * (EE * EE + ab - EE) - EE * ab * (T(8) / T(3)) -
                   k * T(0.5) * ((EE - a) * Rp * b + (EE - b) * a * Rm);
      const X Q = ab * (-T(4) / T(3)) - EE * (a + b) * T(2) + Rp * (Em * b) +
                  Ep * a * Rm - Rp * ab * Rm + M * Nn;
      T s = T(0);
      for (int n = TPS_N - 1; n >= 2; --n) {
        s = s * pp + Q.c[n];
      }
      return s / b;
    }

    /// the region of the energy law for momenta pp <= pm
    template <class T>
    inline Region energy_region(T pp, T pm) {
      if (pm <= T(P_SW)) {
        return Region::Threshold;
      }
      if (pp < T(P_SLOW)) {
        return Region::Slow;
      }
      return Region::General;
    }

  } // namespace stable

  // ------------------------------------------------------------------------ //
  // The energy law: Motz, Olsen & Koch 1969, formula 3D-1000 (Heitler 1954,
  // p. 258, eq. 8)
  // ------------------------------------------------------------------------ //

  /**
   * @brief dsigma/dE+ in alpha r_e^2 per m_e c^2 (Motz, Olsen & Koch 1969,
   *        formula 3D-1000, Z = 1, unscreened; Heitler 1954, p. 258, eq. 8),
   *        evaluated in the type T with the stable representation of namespace
   *        `stable`. Zero outside 1 < E+ < k - 1. dsigma/dE+ = p+ p- B / k^3.
   */
  template <class T>
  inline T dsigma_dE_T(T k, T Ep) {
    T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1))) {
      return T(0);
    }
    // the law is symmetric under E+ <-> E-: evaluate with the operands in a
    // canonical order so that the values at E and at k - E are bitwise equal
    // whenever the two energies are exactly complementary
    if (Ep > Em) {
      std::swap(Ep, Em);
    }
    const T xp = Ep - T(1), xm = Em - T(1);
    const T a = xp * (xp + T(2)), b = xm * (xm + T(2));
    const T pp = std::sqrt(a), pm = std::sqrt(b);
    T       B;
    switch (stable::energy_region(pp, pm)) {
      case stable::Region::Threshold:
        B = stable::series_B(a, b);
        break;
      case stable::Region::Slow:
        B = stable::energy_B_slow(pp, pm, Em, b);
        break;
      default:
        B = stable::energy_Q_general(Ep, Em, pp, pm, a, b) / (a * b);
        break;
    }
    return pp * pm * B / (k * k * k);
  }

  /// the energy law evaluated in the type T, returned as a double (the
  /// quadratures are in double)
  template <class T>
  inline double dsigma_dE_as(double k, double Ep) {
    return static_cast<double>(
      dsigma_dE_T<T>(static_cast<T>(k), static_cast<T>(Ep)));
  }

  /// the energy law in double precision
  inline double dsigma_dE(double k, double Ep) {
    return dsigma_dE_as<double>(k, Ep);
  }

  // ------------------------------------------------------------------------ //
  // The angular law: the closed form of Sauter, Gluckstern and Hull (Motz,
  // Olsen & Koch 1969, formula 3D-2000 = Blumenthal 1970, eq. 10 with the
  // symbols of his eq. 11)
  // ------------------------------------------------------------------------ //

  namespace stable {

    /**
     * @brief The bracket K of the angular law in the general region,
     *        d^2 sigma/(dE+ dcos theta+) = -p+ p- K / (4 k^3), with stable
     *        sub-expressions: L = 2 artanh(p+ p- / (1 + E+ E-)),
     *        eps- = 2 asinh(p-), eps^Q = 2 artanh(p- / Q)
     * @param omc 1 - cos theta+
     * @param sL sign of the terms in L (-1 in the cross section)
     * @param se sign of the term in eps- (+1 in the cross section)
     * @param sQ sign of the terms in eps^Q (+1 in the cross section)
     */
    template <class T>
    inline T angle_K_general(T Ep, T Em, T pp, T pm, T omc, T sL, T se, T sQ) {
      const T k   = Ep + Em;
      const T a   = T(1) / (Ep + pp); // E+ - p+
      const T sp2 = omc * (T(2) - omc);
      const T D   = a + pp * omc; // Delta+ = E+ - p+ cos theta+
      const T D2 = D * D, D4 = D2 * D2;
      const T pp2 = pp * pp;
      const T k2  = k * k;
      const T Ep2 = Ep * Ep, Em2 = Em * Em;
      // Q^2 = |k z^ - p+|^2 = (E- + a)^2 + 2 k p+ omc
      const T Q2 = (Em + a) * (Em + a) + T(2) * k * pp * omc;
      const T Q  = std::sqrt(Q2);
      const T i1 = T(1) / (T(1) + Ep * Em);
      const T M  = two_atanh_over(pp * pm * i1) * i1; // L/(p+ p-)
      const T Rm = two_asinh_over(pm, pm * pm);       // eps-/p-
      const T eQ = two_atanh_over(pm / Q) / (Q * Q);  // eps^Q/(p- Q)
      const T t1 = T(8) * sp2 * (T(2) * Ep2 + T(1)) / (pp2 * D4);
      const T t2 = -T(2) * (T(5) * Ep2 - T(2) * Em * Ep + T(3)) / (pp2 * D2);
      const T t3 = -T(2) * (pp2 - k2) / (Q2 * D2);
      const T t4 = -T(4) * Em / (pp2 * D);
      const T t5 = M * (T(4) * Ep * sp2 * (T(3) * k + pp2 * Em) / (pp2 * D4) +
                        T(4) * Ep2 * (Ep2 + Em2) / (pp2 * D2) +
                        (T(2) - T(2) * (T(7) * Ep2 + T(3) * Em * Ep + Em2)) /
                          (pp2 * D2) +
                        T(2) * k * (Ep2 - Ep * Em - T(1)) / (pp2 * D));
      const T t6 = T(4) * Rm / D;
      const T t7 = eQ *
                   (T(4) / D2 - T(6) * k / D - T(2) * k * (pp2 - k2) / (Q2 * D));
      return t1 + t2 + t3 + t4 + sL * t5 + se * t6 + sQ * t7;
    }

    /**
     * @brief K of the angular law for p+ < P_SLOW (p- > P_SW): K p+^2 as a
     *        series in u = p+, orders 0 and 1 dropped (zero identically: K is
     *        finite at p+ = 0), K = sum_{n>=2} (K p+^2)_n p+^(n-2)
     */
    template <class T>
    inline T angle_K_slow(T pp, T pm, T Em, T omc) {
      using X       = Tps<T, TPS_N>;
      const X u     = X::var();
      const X p2    = u * u;
      const X Ep    = sqrt(p2 + T(1));
      const X k     = Ep + Em;
      const X Ep2   = Ep * Ep;
      const T Em2   = Em * Em;
      const X k2    = k * k;
      const T sp2   = omc * (T(2) - omc);
      const X aa    = Ep - u;       // E+ - p+
      const X Dl    = aa + u * omc; // E+ - p+ cos theta
      const X invD  = recip(Dl);
      const X invD2 = invD * invD;
      const X invD4 = invD2 * invD2;
      const X Q2    = (aa + Em) * (aa + Em) + k * u * (T(2) * omc);
      const X invQ2 = recip(Q2);
      const X invQ  = recip(sqrt(Q2));
      const X eQ    = atanh(invQ * pm) * invQ * (T(2) / pm); // eps^Q/(p- Q)
      const X EE    = Ep * Em;
      const X i1    = recip(EE + T(1));
      const T b     = pm * pm;
      const X M     = compose(p2 * i1 * i1 * b,
                          [](int n) {
                            return T(1) / T(2 * n + 1);
                          }) *
                  i1 * T(2);
      const T Rm = two_asinh_over(pm, b);
      const X T1 = (Ep2 * T(2) + T(1)) * invD4 * (T(8) * sp2);
      const X T2 = (Ep2 * T(5) - Ep * (T(2) * Em) + T(3)) * invD2 * T(-2);
      const X T3 = p2 * (p2 - k2) * invQ2 * invD2 * T(-2);
      const X T4 = invD * (T(-4) * Em);
      const X T5 = -(
        M * (Ep * (k * T(3) + p2 * Em) * invD4 * (T(4) * sp2) +
             Ep2 * (Ep2 + Em2) * invD2 * T(4) +
             ((Ep2 * T(7) + Ep * (T(3) * Em) + Em2) * T(-2) + T(2)) * invD2 +
             k * (Ep2 - Ep * Em - T(1)) * invD * T(2)));
      const X T6 = p2 * invD * (T(4) * Rm);
      const X T7 = eQ * p2 *
                   (invD2 * T(4) - k * invD * T(6) -
                    k * (p2 - k2) * invQ2 * invD * T(2));
      const X Kp2 = T1 + T2 + T3 + T4 + T5 + T6 + T7;
      T       s   = T(0);
      for (int n = TPS_N - 1; n >= 2; --n) {
        s = s * pp + Kp2.c[n];
      }
      return s;
    }

    /// the region of the angular law (p+ the momentum of the lepton whose angle
    /// is given)
    template <class T>
    inline Region angle_region(T pp, T pm) {
      if (pp <= T(P_SW) && pm <= T(P_SW)) {
        return Region::Threshold;
      }
      if (pp < T(P_SLOW)) {
        return Region::Slow;
      }
      return Region::General;
    }

  } // namespace stable

  /**
   * @brief d^2 sigma/(dE+ dcos theta+) in alpha r_e^2 per m_e c^2 (Motz, Olsen
   *        & Koch 1969, formula 3D-2000 = Blumenthal 1970, eq. 10), evaluated
   *        in the type T with the stable representation of namespace `stable`.
   *        Zero outside 1 < E+ < k - 1.
   *        d^2 sigma/(dE+ dcos theta+) = -p+ p- K / (4 k^3).
   * @param k photon energy
   * @param Ep energy of the lepton whose angle is given
   * @param omc 1 - cos theta+ (passed directly to resolve the forward peak)
   */
  template <class T>
  inline T dsigma_dE_dcos_T(T k, T Ep, T omc) {
    const T Em = k - Ep;
    if (!(Ep > T(1)) || !(Em > T(1)) || !(omc >= T(0)) || !(omc <= T(2))) {
      return T(0);
    }
    const T xp = Ep - T(1), xm = Em - T(1);
    const T pp = std::sqrt(xp * (xp + T(2))), pm = std::sqrt(xm * (xm + T(2)));
    T       K;
    switch (stable::angle_region(pp, pm)) {
      case stable::Region::Threshold:
        K = stable::series_K(pp, pm * pm, T(1) - omc);
        break;
      case stable::Region::Slow:
        K = stable::angle_K_slow(pp, pm, Em, omc);
        break;
      default:
        K = stable::angle_K_general(Ep, Em, pp, pm, omc, T(-1), T(1), T(1));
        break;
    }
    return -pp * pm * K / (T(4) * k * k * k);
  }

  /// the angular law evaluated in the type T, returned as a double
  template <class T>
  inline double dsigma_dE_dcos_as(double k, double Ep, double omc) {
    return static_cast<double>(dsigma_dE_dcos_T<T>(static_cast<T>(k),
                                                   static_cast<T>(Ep),
                                                   static_cast<T>(omc)));
  }

  /// the angular law in double precision
  inline double dsigma_dE_dcos_closed(double k, double Ep, double omc) {
    return dsigma_dE_dcos_as<double>(k, Ep, omc);
  }

  // ------------------------------------------------------------------------ //
  // Quadratures over E+ and over cos theta+
  // ------------------------------------------------------------------------ //

  /**
   * @brief Parameters of the quadratures over E+ and over cos theta+
   */
  struct RestFrameQuadrature {
    int    gl_order      = 16; // Gauss-Legendre order per panel
    int    energy_panels = 16; // panels in s, E+ = 1 + (k - 2) sin^2(pi s/2)
    double h_t_angle = 0.5; // panel width in t = ln(omc + w), w = (E+ - p+)/p+
  };

  /**
   * @brief int_1^{k-1} f(E+) dE+ with E+ = 1 + (k - 2) sin^2(pi s/2), which
   *        removes the sqrt end-point behaviour of p+ p-
   */
  template <class F>
  inline double integrate_energy(double k, F&& f, const RestFrameQuadrature& q) {
    if (!(k > 2.0)) {
      return 0.0;
    }
    const GaussLegendre gl(q.gl_order);
    const double        d   = k - 2.0;
    const int           np  = std::max(1, q.energy_panels);
    double              sum = 0.0;
    for (int p = 0; p < np; ++p) {
      const double s0 = static_cast<double>(p) / np,
                   s1 = static_cast<double>(p + 1) / np;
      const double sm = 0.5 * (s0 + s1), sh = 0.5 * (s1 - s0);
      for (int i = 0; i < gl.order(); ++i) {
        const double s  = sm + sh * gl.x[i];
        const double sn = std::sin(0.5 * constants::PI * s);
        const double E  = 1.0 + d * sn * sn;
        const double jac = d * 0.5 * constants::PI * std::sin(constants::PI * s);
        sum += sh * gl.w[i] * jac * f(E);
      }
    }
    return sum;
  }

  /// the scale of the forward peak of the angular law in omc = 1 - cos theta+:
  /// (E+ - p+)/p+
  inline double angular_scale(double Ep) {
    const double pp = momentum(Ep);
    return pp > 0.0 ? energy_minus_momentum(Ep, pp) / pp : 2.0;
  }

  /**
   * @brief int_0^2 f(omc) d(omc) (omc = 1 - cos theta+) with t = ln(omc + w),
   *        w = (E+ - p+)/p+, panels of width <= h_t_angle
   */
  template <class F>
  inline double integrate_angle(double Ep, F&& f, const RestFrameQuadrature& q) {
    const double        w  = angular_scale(Ep);
    const double        t0 = std::log(w), t1 = std::log(2.0 + w);
    const GaussLegendre gl(q.gl_order);
    const int           np  = std::max(1,
                            static_cast<int>(std::ceil((t1 - t0) / q.h_t_angle)));
    double              sum = 0.0;
    for (int p = 0; p < np; ++p) {
      const double a  = t0 + (t1 - t0) * static_cast<double>(p) / np;
      const double b  = t0 + (t1 - t0) * static_cast<double>(p + 1) / np;
      const double tm = 0.5 * (a + b), th = 0.5 * (b - a);
      for (int i = 0; i < gl.order(); ++i) {
        const double t    = tm + th * gl.x[i];
        const double et   = std::exp(t);
        const double omc  = std::min(2.0, std::max(0.0, et - w));
        sum              += th * gl.w[i] * et * f(omc);
      }
    }
    return sum;
  }

  /// int d^2 sigma/(dE+ dcos theta+) dcos theta+ at fixed E+: the energy law
  /// from the angular law evaluated in the type T
  template <class T = double>
  inline double dsigma_dE_from_angular_law(double                     k,
                                           double                     Ep,
                                           const RestFrameQuadrature& q) {
    return integrate_angle(
      Ep,
      [k, Ep](double omc) {
        return dsigma_dE_dcos_as<T>(k, Ep, omc);
      },
      q);
  }

  // ------------------------------------------------------------------------ //
  // Moments and the recoil
  // ------------------------------------------------------------------------ //

  struct Moments {
    double k { 0.0 };
    double sigma_g { 0.0 };       // int dsigma/dE+ dE+
    double sigma_maximon { 0.0 }; // sigma_tilde(k), Maximon's series
    double sigma_h { 0.0 };       // int int d^2 sigma dcos theta+ dE+
    double mean_E { 0.0 };        // <E+> (= <E-> = k/2 by symmetry)
    double mean_pcos { 0.0 };     // <p+ cos theta+>
    double q_k { 0.0 }; // k - 2 <p cos theta> = 2 <(E - p) + p (1 - cos theta)>
  };

  /**
   * @brief The moments at photon energy k of the energy law and of the
   *        angular law evaluated in the type T
   * @note q_k = k - 2 <p cos theta> is the mean momentum absorbed by the
   *       proton along the direction of the photon [m_e c]
   */
  template <class T = double>
  inline Moments moments(double k, const RestFrameQuadrature& q) {
    Moments m;
    m.k             = k;
    m.sigma_maximon = sigma_tilde(k);
    if (!(k > 2.0) || !(m.sigma_maximon > 0.0)) {
      m.q_k = 2.0;
      return m;
    }
    // the moments are normalised by the integrals of the laws themselves, not
    // by Maximon's series, whose truncation error (1.5e-4 at k = 4) would
    // otherwise enter them; sigma_g and sigma_h are returned alongside
    m.sigma_g = integrate_energy(
      k,
      [k](double E) {
        return dsigma_dE_as<T>(k, E);
      },
      q);
    m.mean_E = integrate_energy(
                 k,
                 [k](double E) {
                   return E * dsigma_dE_as<T>(k, E);
                 },
                 q) /
               m.sigma_g;
    // <p+ cos theta+> and the total cross section from the angular law
    m.sigma_h = integrate_energy(
      k,
      [&](double E) {
        return dsigma_dE_from_angular_law<T>(k, E, q);
      },
      q);
    m.mean_pcos = integrate_energy(
                    k,
                    [&](double E) {
                      const double pp = momentum(E);
                      return pp * integrate_angle(
                                    E,
                                    [&](double omc) {
                                      return dsigma_dE_dcos_as<T>(k, E, omc) *
                                             (1.0 - omc);
                                    },
                                    q);
                    },
                    q) /
                  m.sigma_h;
    // q_k = k - 2 <p cos theta> written without cancellation: with
    // k - 2 p cos = (k - 2E) + 2 (E - p) + 2 p (1 - cos) and <k - 2E> = 0
    // exactly, by the symmetry of the energy law under the exchange of the
    // two leptons, q_k = 2 <(E - p) + p (1 - cos theta)>, a mean of positive
    // terms; the direct difference loses about 1e-9 relative at k ~ 1000 in
    // double precision (q_k ~ 0.04 from 2 <p cos> ~ 1000)
    m.q_k = 2.0 *
            integrate_energy(
              k,
              [&](double E) {
                const double pp  = momentum(E);
                const double emp = energy_minus_momentum(E, pp);
                return integrate_angle(
                  E,
                  [&](double omc) {
                    return dsigma_dE_dcos_as<T>(k, E, omc) * (emp + pp * omc);
                  },
                  q);
              },
              q) /
            m.sigma_h;
    return m;
  }

  // ------------------------------------------------------------------------ //
  // A small pool of host threads for the work per table node
  // ------------------------------------------------------------------------ //

  /**
   * @brief Largest number of host threads the build of the tables may start
   * @note Initially the number of processors this process is allowed to run
   *       on: its affinity mask where the system has one (so that an MPI rank
   *       bound to its cores does not use those of its neighbours), the
   *       hardware concurrency elsewhere. The caller may lower it.
   */
  inline auto thread_limit() -> unsigned& {
    static unsigned limit = []() {
      unsigned n = std::thread::hardware_concurrency();
#if defined(__linux__)
      cpu_set_t mask;
      CPU_ZERO(&mask);
      if (sched_getaffinity(0, sizeof(mask), &mask) == 0) {
        const auto allowed = static_cast<unsigned>(CPU_COUNT(&mask));
        if (allowed > 0u and (n == 0u or allowed < n)) {
          n = allowed;
        }
      }
#endif
      return std::max(1u, n);
    }();
    return limit;
  }

  /**
   * @brief Call f(i) for every i in [0, n) on host threads
   * @param n_threads number of threads (0: as many as `thread_limit()`
   *        allows; never more than that)
   * @note An exception thrown by f is rethrown in the calling thread
   * @note Every index is computed by one call of f, so the result does not
   *       depend on the number of threads
   */
  template <class F>
  inline void parallel_for_index(std::size_t n, F&& f, unsigned n_threads = 0) {
    if (n == 0) {
      return;
    }
    const unsigned limit = thread_limit();
    n_threads = (n_threads == 0) ? limit : std::min(n_threads, limit);
    n_threads = static_cast<unsigned>(std::min<std::size_t>(n_threads, n));
    if (n_threads <= 1) {
      for (std::size_t i = 0; i < n; ++i) {
        f(i);
      }
      return;
    }
    std::atomic<std::size_t>        next { 0 };
    std::vector<std::thread>        pool;
    std::vector<std::exception_ptr> errors(n_threads);
    for (unsigned t = 0; t < n_threads; ++t) {
      pool.emplace_back([&, t]() {
        try {
          for (;;) {
            const std::size_t i = next.fetch_add(1);
            if (i >= n) {
              break;
            }
            f(i);
          }
        } catch (...) {
          errors[t] = std::current_exception();
        }
      });
    }
    for (auto& th : pool) {
      th.join();
    }
    for (auto& e : errors) {
      if (e) {
        std::rethrow_exception(e);
      }
    }
  }

} // namespace arch::bh::rf

#endif // ARCHETYPES_EMISSION_BETHE_HEITLER_BH_RESTFRAME_H

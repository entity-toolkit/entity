#ifndef FRAMEWORK_METRICS_QKERR_SCHILD_H
#define FRAMEWORK_METRICS_QKERR_SCHILD_H

#include "global.h"
#include "metric_base.h"

#include <cmath>
#include <cassert>

namespace ntt {
  /**
   * Kerr metric in Kerr-Schild coordinates
   * Units: c = rg = 1
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Metric : public MetricBase<D> {
  private:
    // Spin parameter, in [0,1[
    const real_t a;
    const real_t r0, h;
    const real_t chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_inv, deta_inv, dphi_inv;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;

  public:
    Metric(std::vector<unsigned int> resolution, std::vector<real_t> extent, const real_t* params)
      : MetricBase<D> {"qkerr_schild", resolution, extent},
        a(params[4]),
        r0(params[0]),
        h(params[1]),
        chi_min {math::log(this->x1_min - r0)},
        eta_min {ZERO},
        phi_min {ZERO},
        dchi {(math::log(this->x1_max - r0) - chi_min) / this->nx1},
        deta {constant::PI / this->nx2},
        dphi {constant::TWO_PI / this->nx3},
        dchi_inv {ONE / dchi},
        deta_inv {ONE / deta},
        dphi_inv {ONE / dphi},
        dchi_sqr {dchi * dchi},
        deta_sqr {deta * deta},
        dphi_sqr {dphi * dphi} {}
    ~Metric() = default;

    [[nodiscard]] Inline auto spin() const -> const real_t& { return a; }

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dimension::TWO_D) {
        real_t min_dx {-1.0};
        for (int i {0}; i < this->nx1; ++i) {
          for (int j {0}; j < this->nx2; ++j) {
            real_t i_ {(real_t)(i) + HALF};
            real_t j_ {(real_t)(j) + HALF};
            real_t inv_dx1_ {this->h_11_inv({i_, j_})};
            real_t inv_dx2_ {this->h_22_inv({i_, j_})};
            real_t dx = 1.0 / (this->alpha({i_, j_}) * math::sqrt(inv_dx1_ + inv_dx2_) + this->beta1u({i_, j_}));
            if ((min_dx >= dx) || (min_dx < 0.0)) { min_dx = dx; }
          }
        }
        return min_dx;
      } else {
        NTTError("min cell finding not implemented for 3D qspherical");
      }
      return ZERO;
    }

    /**
     * @brief Compute d(th) / d(eta) for a given eta.
     *
     */
    Inline auto dtheta_deta(const real_t& eta) const -> real_t {
      return (ONE + TWO * h
              + static_cast<real_t>(12.0) * h * (eta * constant::INV_PI) * ((eta * constant::INV_PI) - ONE));
    }

    /**
     * @brief Convert quasi-spherical eta to spherical theta.
     *
     */
    Inline auto eta2theta(const real_t& eta) const -> real_t {
      return eta + TWO * h * eta * (constant::PI - TWO * eta) * (constant::PI - eta) * constant::INV_PI_SQR;
    }
    /**
     * @brief Convert spherical theta to quasi-spherical eta.
     *
     */
    Inline auto theta2eta(const real_t& theta) const -> real_t {
      // R = (-9 h^2 (Pi - 2 y) + Sqrt[3] Sqrt[-(h^3 ((-4 + h) (Pi + 2 h Pi)^2 + 108 h Pi y - 108 h y^2))])^(1/3)
      double R {math::pow(-9.0 * SQR(h) * (constant::PI - 2.0 * theta)
                            + constant::SQRT3
                                * math::sqrt(-(CUBE(h)
                                               * ((h - 4.0) * SQR(constant::PI + h * constant::TWO_PI)
                                                  + 108.0 * h * constant::PI * theta - 108.0 * h * SQR(theta)))),
                          static_cast<real_t>(1.0 / 3.0))};
      // eta = Pi^(2/3)(6 Pi^(1/3) + 2 2^(1/3)(h-1)(3Pi)^(2/3)/R + 2^(2/3) 3^(1/3) R / h)/12
      constexpr double PI_TO_TWO_THIRD {2.14502939711102560008};
      constexpr double PI_TO_ONE_THIRD {1.46459188756152326302};
      constexpr double TWO_TO_TWO_THIRD {1.58740105196819947475};
      constexpr double THREE_TO_ONE_THIRD {1.442249570307408382321};
      constexpr double TWO_TO_ONE_THIRD {1.2599210498948731647672};
      constexpr double THREE_PI_TO_TWO_THIRD {4.46184094890142313715794};
      return static_cast<real_t>(PI_TO_TWO_THIRD
                                 * (6.0 * PI_TO_ONE_THIRD
                                    + 2.0 * TWO_TO_ONE_THIRD * (h - ONE) * THREE_PI_TO_TWO_THIRD / R
                                    + TWO_TO_TWO_THIRD * THREE_TO_ONE_THIRD * R / h)
                                 / 12.0);
    }

    /**
     * Compute metric component 11.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("h_11 not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t cth {math::cos(theta)};
        return dchi_sqr * math::exp(2.0 * chi) * (ONE + TWO * r / (r * r + a * a * cth * cth));
      }
    }

    /**
     * Compute metric component 22.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("h_22 not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t dtheta_deta_ {dtheta_deta(eta)};
        real_t cth {math::cos(theta)};
        return deta_sqr * SQR(dtheta_deta_) * (r * r + a * a * cth * cth);
      }
    }

    /**
     * Compute metric component 33.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("h_33 not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t cth {math::cos(theta)};
        real_t sth {math::sin(theta)};
        real_t delta {r * r - TWO * r + a * a};
        real_t As {(r * r + a * a) * (r * r + a * a) - a * a * delta * sth * sth};
        return As * sth * sth / (r * r + a * a * cth * cth);
      }
    }

    /**
     * Compute metric component 13.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns h_13 (covariant, lower index) metric component.
     */
    Inline auto h_13(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("h_13 not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t cth {math::cos(theta)};
        real_t sth {math::sin(theta)};
        return -dchi * math::exp(chi) * a * sth * sth * (ONE + TWO * r / (r * r + a * a * cth * cth));
      }
    }

    /**
     * Compute lapse function.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns alpha.
     */
    Inline auto alpha(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("alpha not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t cth {math::cos(theta)};
        real_t z {TWO * r / (r * r + a * a * cth * cth)};
        return ONE / math::sqrt(ONE + z);
      }
    }

    /**
     * Compute radial component of shift vector.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @returns beta^1 (contravariant).
     */
    Inline auto beta1u(const coord_t<D>& xi) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("beta1u not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t eta {xi[1] * deta + eta_min};
        real_t theta {eta2theta(eta)};
        real_t cth {math::cos(theta)};
        real_t z {TWO * r / (r * r + a * a * cth * cth)};
        return math::exp(-chi) * dchi_inv * (z / (ONE + z));
      }
    }

    /**
     * Compute the square root of the determinant of h-matrix divided by sin(theta).
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h))/sin(theta).
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t { return h_22(x) / alpha(x); }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("polar_area not implemented for 1D qspherical");
        return ZERO;
      } else {
        real_t chi {x[0] * dchi + chi_min};
        real_t r {r0 + math::exp(chi)};
        real_t del_eta {x[1] * deta + eta_min};
        real_t del_theta {eta2theta(del_eta)};
        return dchi * math::exp(chi) * (SQR(r) + SQR(a)) * math::sqrt(ONE + TWO * r / (SQR(r) + SQR(a)))
               * (ONE - math::cos(del_theta));
      }
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "non_diag_vector_transform.h"

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cartesian physical units (size of the array is D).
     */
    Inline void x_Code2Cart(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Cart not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        coord_t<D> x_sph;
        x_Code2Sph(xi, x_sph);
        x[0] = x_sph[0] * math::sin(x_sph[1]);
        x[1] = x_sph[0] * math::cos(x_sph[1]);
      } else if constexpr (D == Dimension::THREE_D) {
        coord_t<D> x_sph;
        x_Code2Sph(xi, x_sph);
        x[0] = x_sph[0] * math::sin(x_sph[1]) * math::cos(x_sph[2]);
        x[1] = x_sph[0] * math::sin(x_sph[1]) * math::sin(x_sph[2]);
        x[2] = x_sph[0] * math::cos(x_sph[1]);
      }
    }

    /**
     * Coordinate conversion from Cartesian physical units to code units.
     *
     * @param x coordinate array in Cartesian coordinates in
     * physical units (size of the array is D).
     * @param xi coordinate array in code units (size of the array is D).
     */
    Inline void x_Cart2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Cart2Code not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        coord_t<D> x_sph;
        x_sph[0] = math::sqrt(x[0] * x[0] + x[1] * x[1]);
        x_sph[1] = math::atan2(x[1], x[0]);
        x_Sph2Code(x_sph, xi);
      } else if constexpr (D == Dimension::THREE_D) {
        coord_t<D> x_sph;
        x_sph[0] = math::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        x_sph[1] = math::atan2(x[1], x[0]);
        x_sph[2] = math::acos(x[2] / x_sph[0]);
        x_Sph2Code(x_sph, xi);
      }
    }

    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Spherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Sph not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        real_t chi {xi[0] * dchi + chi_min};
        real_t eta {xi[1] * deta + eta_min};
        x[0] = r0 + math::exp(chi);
        x[1] = eta2theta(eta);
      } else if constexpr (D == Dimension::THREE_D) {
        real_t chi {xi[0] * dchi + chi_min};
        real_t eta {xi[1] * deta + eta_min};
        real_t phi {xi[2] * dphi + phi_min};
        x[0] = r0 + math::exp(chi);
        x[1] = eta2theta(eta);
        x[2] = phi;
      }
    }

    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param x coordinate array in Spherical coordinates in physical units (size of the array is D).
     * @param xi coordinate array in code units (size of the array is D).
     */
    Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("x_Code2Sph not implemented for 1D");
      } else if constexpr (D == Dimension::TWO_D) {
        real_t chi {math::log(x[0] - r0)};
        real_t eta {theta2eta(x[1])};
        xi[0] = (chi - chi_min) * dchi_inv;
        xi[1] = (eta - eta_min) * deta_inv;
      } else if constexpr (D == Dimension::THREE_D) {
        real_t chi {math::log(x[0] - r0)};
        real_t eta {theta2eta(x[1])};
        real_t phi {x[2]};
        xi[0] = (chi - chi_min) * dchi_inv;
        xi[1] = (eta - eta_min) * deta_inv;
        xi[2] = (phi - phi_min) * dphi_inv;
      }
    }

    /**
     * Vector conversion from contravariant to spherical contravariant.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_cntrv vector in contravariant basis (size of the array is 3).
     * @param vsph_cntrv vector in spherical contravariant basis (size of the array is 3).
     */
    Inline void v_Cntr2SphCntrv(const coord_t<D>&                xi,
                                const vec_t<Dimension::THREE_D>& vi_cntrv,
                                vec_t<Dimension::THREE_D>&       vsph_cntrv) const {
      if constexpr (D == Dimension::ONE_D) {
        NTTError("v_Cntr2SphCntrv not implemented for 1D");
      } else {
        real_t chi {xi[0] * dchi + chi_min};
        real_t eta {xi[1] * deta + eta_min};
        real_t deta_dtheta_ {ONE / dtheta_deta(eta)};
        vsph_cntrv[0] = vi_cntrv[0] * math::exp(-chi) * dchi_inv;
        vsph_cntrv[1] = vi_cntrv[1] * deta_dtheta_ * deta_inv;
        vsph_cntrv[2] = vi_cntrv[2] * dphi_inv;
      }
    }
  };

} // namespace ntt

#endif

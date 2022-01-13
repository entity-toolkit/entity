#ifndef FRAMEWORK_METRICS_QSPHERICAL_H
#define FRAMEWORK_METRICS_QSPHERICAL_H

#include "global.h"
#include "metric.h"

#include <cmath>
#include <stdexcept>

namespace ntt {
  /**
   * Flat metric in quasi-spherical system.
   * chi, eta, phi = log(r-r0), f(h, theta), phi
   *
   * @todo change `eta_min`, `phi_min`.
   * @tparam D dimension.
   */
  template <Dimension D>
  class QSpherical : virtual public Metric<D> {
  private:
    const real_t r0, h, chi_min, eta_min, phi_min;
    const real_t dchi, deta, dphi;
    const real_t dchi_sqr, deta_sqr, dphi_sqr;

  public:
    QSpherical(std::vector<std::size_t> resolution, std::vector<real_t> extent, const real_t& r0_, const real_t& h_)
      : Metric<D> {"qspherical", resolution, extent},
        r0 {r0_},
        h {h_},
        chi_min {std::log(this->x1_min - r0)},
        eta_min {ZERO},
        phi_min {ZERO},
        dchi((std::log(this->x1_max - r0) - chi_min) / this->nx1),
        deta(constant::PI / this->nx2),
        dphi(constant::TWO_PI / this->nx3),
        dchi_sqr(dchi * dchi),
        deta_sqr(deta * deta),
        dphi_sqr(dphi * dphi) {}
    ~QSpherical() = default;

    auto findSmallestCell() const -> real_t override {
      if constexpr (D == Dimension::TWO_D) {
        real_t min_dx {-1.0};
        for (int i {0}; i < this->nx1; ++i) {
          for (int j {0}; j < this->nx2; ++j) {
            real_t i_ {(real_t)(i) + HALF};
            real_t j_ {(real_t)(j) + HALF};
            real_t dx1_ {this->h_11({i_, j_})};
            real_t dx2_ {this->h_22({i_, j_})};
            real_t dx = 1.0 / std::sqrt(1.0 / dx1_ + 1.0 / dx2_);
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
      return (ONE + static_cast<real_t>(2.0) * h + static_cast<real_t>(12.0) * h * (eta * constant::INV_PI) * ((eta * constant::INV_PI) - ONE));
    }

    /**
     * @brief Convert quasi-spherical eta to spherical theta.
     *
     */
    Inline auto eta2theta(const real_t& eta) const -> real_t {
      return eta
             + static_cast<real_t>(2.0) * h * eta * (constant::PI - static_cast<real_t>(2.0) * eta) * (constant::PI - eta)
                 * constant::INV_PI_SQR;
    }

    Inline auto h_11(const coord_t<D>& x) const -> real_t override {
      auto chi {x[0] * dchi + chi_min};
      return dchi_sqr * std::exp(2.0 * chi);
    }
    Inline auto h_22(const coord_t<D>& x) const -> real_t override {
      auto chi {x[0] * dchi + chi_min};
      auto r {r0 + std::exp(chi)};
      auto eta {x[1] * deta + eta_min};
      auto dtheta_deta_ {dtheta_deta(eta)};
      return deta_sqr * r * r * dtheta_deta_ * dtheta_deta_;
    }
    Inline auto h_33(const coord_t<D>& x) const -> real_t override {
      auto chi {x[0] * dchi + chi_min};
      auto r {r0 + std::exp(chi)};
      auto eta {x[1] * deta + eta_min};
      auto theta {eta2theta(eta)};
      auto sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }

    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t override {
      auto chi {x[0] * dchi + chi_min};
      auto r {r0 + std::exp(chi)};
      auto eta {x[1] * deta + eta_min};
      auto theta {eta2theta(eta)};
      auto sin_theta {std::sin(theta)};
      auto dtheta_deta_ {dtheta_deta(eta)};
      return dchi * deta * std::exp(chi) * r * r * sin_theta * dtheta_deta_;
    }

    Inline auto polar_area(const coord_t<D>& x) const -> real_t override {
      auto chi {x[0] * dchi + chi_min};
      auto r {r0 + std::exp(chi)};
      auto eta {x[1] * deta + eta_min};
      auto theta {eta2theta(eta)};
      return deta * std::exp(chi) * r * r * (ONE - std::cos(theta));
    }

    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const override;

    Inline void v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const override;
    Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const override;

    // todo
    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const override {};

    // defaults
    Inline auto h_12(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_12(x); }
    Inline auto h_13(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_13(x); }
    Inline auto h_21(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_21(x); }
    Inline auto h_23(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_23(x); }
    Inline auto h_31(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_31(x); }
    Inline auto h_32(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_32(x); }
  };

  // * * * * * * * * * * * * * * *
  // vector transformations
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void QSpherical<D>::v_Hat2Cntrv(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi_hat,
                                        vec_t<Dimension::THREE_D>& vi) const {
    vi[0] = vi_hat[0] / std::sqrt(h_11(xi));
    vi[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi[2] = vi_hat[2] / std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void QSpherical<D>::v_Cntrv2Hat(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi,
                                        vec_t<Dimension::THREE_D>& vi_hat) const {
    vi_hat[0] = vi[0] * std::sqrt(h_11(xi));
    vi_hat[1] = vi[1] * std::sqrt(h_22(xi));
    vi_hat[2] = vi[2] * std::sqrt(h_33(xi));
  }

  // * * * * * * * * * * * * * * *
  // 1D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void QSpherical<Dimension::ONE_D>::x_Code2Sph(const coord_t<Dimension::ONE_D>&,
                                                      coord_t<Dimension::ONE_D>&) const { }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void QSpherical<Dimension::TWO_D>::x_Code2Sph(const coord_t<Dimension::TWO_D>& xi,
                                                      coord_t<Dimension::TWO_D>& x) const {
    real_t chi {xi[0] * dchi + chi_min};
    real_t eta {xi[1] * deta + eta_min};
    x[0] = r0 + std::exp(chi);
    x[1] = eta2theta(eta);
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void QSpherical<Dimension::THREE_D>::x_Code2Sph(const coord_t<Dimension::THREE_D>& xi,
                                                        coord_t<Dimension::THREE_D>& x) const {
    real_t chi {xi[0] * dchi + chi_min};
    real_t eta {xi[1] * deta + eta_min};
    real_t phi {xi[2] * dphi + phi_min};
    x[0] = r0 + std::exp(chi);
    x[1] = eta2theta(eta);
    x[2] = phi;
  }

  } // namespace ntt

#endif

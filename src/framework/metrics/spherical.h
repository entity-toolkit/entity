#ifndef FRAMEWORK_METRICS_SPHERICAL_H
#define FRAMEWORK_METRICS_SPHERICAL_H

#include "global.h"
#include "metric.h"

#include <cmath>
#include <cassert>

namespace ntt {
  /**
   * Flat metric in spherical system: diag(-1, 1, r^2, r^2 sin(th)^2).
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Spherical : virtual public Metric<D> {
  private:
    const real_t dr, dtheta, dphi;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;

  public:
    Spherical(std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : Metric<D> {"spherical", resolution, extent},
        dr((this->x1_max - this->x1_min) / this->nx1),
        dtheta(constant::PI / this->nx2),
        dphi(constant::TWO_PI / this->nx3),
        dr_sqr(dr * dr),
        dtheta_sqr(dtheta * dtheta),
        dphi_sqr(dphi * dphi) {}
    ~Spherical() = default;

    auto findSmallestCell() const -> real_t override {
      if constexpr (D == Dimension::TWO_D) {
        auto dx1 {dr};
        auto dx2 {this->x1_min * dtheta};
        return ONE / std::sqrt(ONE / (dx1 * dx1) + ONE / (dx2 * dx2));
      } else {
        NTTError("min cell finding not implemented for 3D spherical");
      }
      return ZERO;
    }

    Inline auto h_11(const coord_t<D>&) const -> real_t override { return dr_sqr; }
    Inline auto h_22(const coord_t<D>& x) const -> real_t override {
      real_t r {x[0] * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }
    Inline auto h_33(const coord_t<D>& x) const -> real_t override {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      real_t sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }

    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t override {
      real_t r {x[0] * dr + this->x1_min};
      real_t theta {x[1] * dtheta};
      return dr * dtheta * r * r * std::sin(theta);
    }

    Inline auto polar_area(const coord_t<D>& x) const -> real_t override {
      real_t r {x[0] * dr + this->x1_min};
      real_t del_theta {x[1] * dtheta};
      return dtheta * dphi * r * r * (ONE - std::cos(del_theta));
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
  Inline void Spherical<D>::v_Hat2Cntrv(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi_hat,
                                        vec_t<Dimension::THREE_D>& vi) const {
    vi[0] = vi_hat[0] / std::sqrt(h_11(xi));
    vi[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi[2] = vi_hat[2] / std::sqrt(h_33(xi));
  }
  template <Dimension D>
  Inline void Spherical<D>::v_Cntrv2Hat(const coord_t<D>& xi,
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
  Inline void Spherical<Dimension::ONE_D>::x_Code2Sph(const coord_t<Dimension::ONE_D>&,
                                                      coord_t<Dimension::ONE_D>&) const { }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Spherical<Dimension::TWO_D>::x_Code2Sph(const coord_t<Dimension::TWO_D>& xi,
                                                      coord_t<Dimension::TWO_D>& x) const {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta + this->x2_min;
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Spherical<Dimension::THREE_D>::x_Code2Sph(const coord_t<Dimension::THREE_D>& xi,
                                                        coord_t<Dimension::THREE_D>& x) const {
    x[0] = xi[0] * dr + this->x1_min;
    x[1] = xi[1] * dtheta + this->x2_min;
    x[2] = xi[2] * dphi + this->x3_min;
  }

  } // namespace ntt

#endif

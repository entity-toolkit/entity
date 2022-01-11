#ifndef FRAMEWORK_METRICS_SPHERICAL_H
#define FRAMEWORK_METRICS_SPHERICAL_H

#include "global.h"
#include "metric.h"

#include <cmath>

namespace ntt {
  /**
   * Flat metric in spherical system: diag(-1, 1, r^2, r^2 sin(th)^2).
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Spherical : public Metric<D> {
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

    /**
     * Compute minimum effective cell size for Minkowski metric: `dx / sqrt(D)` (in physical units).
     *
     * @returns Minimum cell size [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dimension::TWO_D) {
        auto dx1 {dr};
        auto dx2 {this->x1_min * dtheta};
        return ONE / std::sqrt(ONE / (dx1 * dx1) + ONE / (dx2 * dx2));
      } else {
        NTTError("min cell finding not implemented for 3D spherical");
      }
    }

    /**
     * Compute metric component 11.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_11 (covariant, lower index) metric component.
     */
    Inline auto h_11(const coord_t<D>&) const -> real_t { return dr_sqr; }
    /**
     * Compute metric component 22.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_22 (covariant, lower index) metric component.
     */
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      auto r {x[0] * dr + this->x1_min};
      return dtheta_sqr * r * r;
    }
    /**
     * Compute metric component 33.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns h_33 (covariant, lower index) metric component.
     */
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      auto r {x[0] * dr + this->x1_min};
      auto theta {x[1] * dtheta};
      auto sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }

    /**
     * Compute the square root of the determinant of h-matrix.
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns sqrt(det(h_ij)).
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      auto r {x[0] * dr + this->x1_min};
      auto theta {x[1] * dtheta};
      return dr * dtheta * r * r * std::sin(theta);
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     *
     * @param x coordinate array in code units (size of the array is D).
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      auto r {x[0] * dr + this->x1_min};
      auto del_theta {x[1] * dtheta};
      return dtheta * dphi * r * r * (ONE - std::cos(del_theta));
    }

    /**
     * Coordinate conversion from code units to Spherical physical units.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param x coordinate array in Cpherical coordinates in physical units (size of the array is D).
     */
    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const;

    /**
     * Vector conversion from hatted to contravariant basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     * @param vi vector in contravariant basis (size of the array is 3).
     */
    Inline void v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;

    /**
     * Vector conversion from contravariant to hatted basis.
     *
     * @param xi coordinate array in code units (size of the array is D).
     * @param vi vector in contravariant basis (size of the array is 3).
     * @param vi_hat vector in hatted basis (size of the array is 3).
     */
    Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const;
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
                                                      coord_t<Dimension::ONE_D>&) const {
    NTTError("1d spherical not defined");
  }

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

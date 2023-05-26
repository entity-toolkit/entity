#ifndef FRAMEWORK_METRICS_KERR_SCHILD_H
#define FRAMEWORK_METRICS_KERR_SCHILD_H

#include "wrapper.h"

#include "metric_base.h"

#include <cassert>
#include <cmath>

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
    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;
    const real_t dr_sqr, dtheta_sqr, dphi_sqr;
    // Spin parameter, in [0,1[
    // and horizon size in units of rg
    // all physical extents are in units of rg
    const real_t rh, a, a_sqr;

  public:
    const real_t dx_min;

    Metric(std::vector<unsigned int> resolution,
           std::vector<real_t>       extent,
           const real_t*             params)
      : MetricBase<D> { "kerr_schild", resolution, extent },
        rh { params[5] },
        a { params[4] },
        a_sqr { SQR(a) },
        dr { (this->x1_max - this->x1_min) / this->nx1 },
        dtheta { (real_t)(constant::PI) / this->nx2 },
        dphi { (real_t)(constant::TWO_PI) / this->nx3 },
        dr_inv { ONE / dr },
        dtheta_inv { ONE / dtheta },
        dphi_inv { ONE / dphi },
        dr_sqr { SQR(dr) },
        dtheta_sqr { SQR(dtheta) },
        dphi_sqr { SQR(dphi) },
        dx_min { findSmallestCell() } {}
    ~Metric() = default;

    [[nodiscard]] auto spin() const -> const real_t& {
      return a;
    }

    [[nodiscard]] auto rhorizon() const -> const real_t& {
      return rh;
    }

    Inline auto h_11(const coord_t<D>& x) const -> real_t {
      return dr_sqr;
    }
    Inline auto h_22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      return dtheta_sqr * SQR(r);
    }
    Inline auto h_33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return SQR(r * math::sin(theta));
      } else {
        return dphi_sqr * SQR(r * math::sin(theta));
      }
    }
    Inline auto h_13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto h11(const coord_t<D>& x) const -> real_t {
      return SQR(dr_inv);
    }
    Inline auto h22(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      return SQR(dtheta_inv / r);
    }
    Inline auto h33(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      if constexpr (D == Dim2) {
        return ONE / (SQR(r * math::sin(theta)));
      } else {
        return SQR(dphi_inv / (r * math::sin(theta)));
      }
    }
    Inline auto h13(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto alpha(const coord_t<D>& x) const -> real_t {
      return ONE;
    }
    Inline auto beta1(const coord_t<D>& x) const -> real_t {
      return ZERO;
    }
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      // ?ASK is this correct?
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(r) * math::sin(theta);
      } else {
        return dr * dtheta * dphi * SQR(r) * math::sin(theta);
      }
    }
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      const real_t r { x[0] * dr + this->x1_min };
      const real_t theta { x[1] * dtheta };
      // ?ASK is this correct?
      if constexpr (D == Dim2) {
        return dr * dtheta * SQR(r);
      } else {
        return dr * dtheta * dphi * SQR(r);
      }
    }

    /**
     * Compute the fiducial minimum cell volume.
     *
     * @returns Minimum cell volume of the grid [code units].
     */
    Inline auto min_cell_volume() const -> real_t {
      return math::pow(dx_min * math::sqrt(static_cast<real_t>(D)), static_cast<short>(D));
    }

    /**
     * Compute the area at the pole (used in axisymmetric solvers).
     * Approximate solution for the polar area.
     *
     * @param x coordinate array in code units
     * @returns Area at the pole.
     */
    Inline auto polar_area(const coord_t<D>& x) const -> real_t {
      real_t r { x[0] * dr + this->x1_min };
      real_t del_theta { x[1] * dtheta };
      return dr * SQR(r) * (ONE - math::cos(del_theta));
    }
/**
 * @note Since kokkos disallows virtual inheritance, we have to
 *       include vector transformations for a non-diagonal metric here
 *       (and not in the base class).
 */
#include "metrics_utils/ks_common.h"
#include "metrics_utils/sph_common.h"

    /**
     * Compute minimum effective cell size for a given metric (in physical units).
     * @returns Minimum cell size of the grid [physical units].
     */
    auto findSmallestCell() const -> real_t {
      if constexpr (D == Dim2) {
        real_t min_dx { -ONE };
        for (int i { 0 }; i < this->nx1; ++i) {
          for (int j { 0 }; j < this->nx2; ++j) {
            real_t        i_ { static_cast<real_t>(i) + HALF };
            real_t        j_ { static_cast<real_t>(j) + HALF };
            coord_t<Dim2> ij { i_, j_ };
            real_t        dx = ONE
                        / (this->alpha(ij) * std::sqrt(this->h11(ij) + this->h22(ij))
                           + this->beta1(ij));
            if ((min_dx > dx) || (min_dx < 0.0)) {
              min_dx = dx;
            }
          }
        }
        return min_dx;
      } else {
        NTTHostError("min cell finding not implemented for 3D");
        return ZERO;
      }
    }

    /**
     * Coordinate conversion from code units to Cartesian physical units.
     *
     * @param xi coordinate array in code units
     * @param x coordinate array in Cartesian physical units
     */
    Inline void x_Code2Cart(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dim2) {
        coord_t<D> x_sph;
        x_Code2Sph(xi, x_sph);
        x[0] = x_sph[0] * math::sin(x_sph[1]);
        x[1] = x_sph[0] * math::cos(x_sph[1]);
      } else if constexpr (D == Dim3) {
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
     * @param x coordinate array in Cartesian coordinates in physical units
     * @param xi coordinate array in code units
     */
    Inline void x_Cart2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dim2) {
        coord_t<D> x_sph;
        x_sph[0] = math::sqrt(x[0] * x[0] + x[1] * x[1]);
        x_sph[1] = math::atan2(x[1], x[0]);
        x_Sph2Code(x_sph, xi);
      } else if constexpr (D == Dim3) {
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
     * @param xi coordinate array in code units
     * @param x coordinate array in Spherical coordinates in physical units
     */
    Inline void x_Code2Sph(const coord_t<D>& xi, coord_t<D>& x) const {
      if constexpr (D == Dim2) {
        x[0] = xi[0] * dr + this->x1_min;
        x[1] = xi[1] * dtheta;
      } else if constexpr (D == Dim3) {
        x[0] = xi[0] * dr + this->x1_min;
        x[1] = xi[1] * dtheta;
        x[2] = xi[2] * dphi;
      }
    }

    /**
     * Coordinate conversion from Spherical physical units to code units.
     *
     * @param x coordinate array in Spherical coordinates in physical units
     * @param xi coordinate array in code units
     */
    Inline void x_Sph2Code(const coord_t<D>& x, coord_t<D>& xi) const {
      if constexpr (D == Dim2) {
        xi[0] = (x[0] - this->x1_min) * dr_inv;
        xi[1] = x[1] * dtheta_inv;
      } else if constexpr (D == Dim3) {
        xi[0] = (x[0] - this->x1_min) * dr_inv;
        xi[1] = x[1] * dtheta_inv;
        xi[2] = x[2] * dphi_inv;
      }
    }

    /**
     * Vector conversion from contravariant to spherical contravariant.
     *
     * @param xi coordinate array in code units
     * @param vi_cntrv vector in contravariant basis
     * @param vsph_cntrv vector in spherical contravariant basis
     */
    Inline void v3_Cntrv2PhysCntrv(const coord_t<D>&,
                                  const vec_t<Dim3>& vi_cntrv,
                                  vec_t<Dim3>&       vsph_cntrv) const {
      vsph_cntrv[0] = vi_cntrv[0] * dr;
      vsph_cntrv[1] = vi_cntrv[1] * dtheta;
      if constexpr (D == Dim2) {
        vsph_cntrv[2] = vi_cntrv[2];
      } else {
        vsph_cntrv[2] = vi_cntrv[2] * dphi;
      }
    }

    /**
     * Vector conversion from spherical contravariant to contravariant.
     *
     * @param xi coordinate array in code units
     * @param vsph_cntrv vector in spherical contravariant basis
     * @param vi_cntrv vector in contravariant basis
     */
    Inline void v3_PhysCntrv2Cntrv(const coord_t<D>&,
                                  const vec_t<Dim3>& vsph_cntrv,
                                  vec_t<Dim3>&       vi_cntrv) const {
      vi_cntrv[0] = vsph_cntrv[0] * dr_inv;
      vi_cntrv[1] = vsph_cntrv[1] * dtheta_inv;
      if constexpr (D == Dim2) {
        vi_cntrv[2] = vsph_cntrv[2];
      } else {
        vi_cntrv[2] = vsph_cntrv[2] * dphi_inv;
      }
    }

    /**
     * Vector conversion from covariant to spherical covariant.
     *
     * @param xi coordinate array in code units
     * @param vi_cov vector in covariant basis
     * @param vsph_cov vector in spherical covariant basis
     */
    Inline void v3_Cov2PhysCov(const coord_t<D>&,
                              const vec_t<Dim3>& vi_cov,
                              vec_t<Dim3>&       vsph_cov) const {
      vsph_cov[0] = vi_cov[0] * dr_inv;
      vsph_cov[1] = vi_cov[1] * dtheta_inv;
      if constexpr (D == Dim2) {
        vsph_cov[2] = vi_cov[2];
      } else {
        vsph_cov[2] = vi_cov[2] * dphi_inv;
      }
    }

    /**
     * Vector conversion from covariant to spherical covariant.
     *
     * @param xi coordinate array in code units
     * @param vsph_cov vector in spherical covariant basis
     * @param vi_cov vector in covariant basis
     */
    Inline void v3_PhysCov2Cov(const coord_t<D>&,
                              const vec_t<Dim3>& vsph_cov,
                              vec_t<Dim3>&       vi_cov) const {
      vi_cov[0] = vsph_cov[0] * dr;
      vi_cov[1] = vsph_cov[1] * dtheta;
      if constexpr (D == Dim2) {
        vi_cov[2] = vsph_cov[2];
      } else {
        vi_cov[2] = vsph_cov[2] * dphi;
      }
    }
  };

}    // namespace ntt

#endif

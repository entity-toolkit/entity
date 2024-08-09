/**
 * @file metrics/spherical.h
 * @brief Flat space-time spherical metric class diag(-1, 1, r^2, r^2, sin(th)^2)
 * @implements
 *   - metric::Spherical<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 * !TODO
 *   - 3D version of find_dxMin
 */

#ifndef METRICS_SPHERICAL_H
#define METRICS_SPHERICAL_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>

namespace metric {

  template <Dimension D>
  class Spherical : public MetricBase<D> {
    static_assert(D != Dim::_1D, "1D spherical not available");
    static_assert(D != Dim::_3D, "3D spherical not fully implemented");

    const real_t dr, dtheta, dphi;
    const real_t dr_inv, dtheta_inv, dphi_inv;

  public:
    static constexpr const char*       Label { "spherical" };
    static constexpr Dimension         PrtlDim { Dim::_3D };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Spherical };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Sph };
    using MetricBase<D>::x1_min;
    using MetricBase<D>::x1_max;
    using MetricBase<D>::x2_min;
    using MetricBase<D>::x2_max;
    using MetricBase<D>::x3_min;
    using MetricBase<D>::x3_max;
    using MetricBase<D>::nx1;
    using MetricBase<D>::nx2;
    using MetricBase<D>::nx3;
    using MetricBase<D>::set_dxMin;

    Spherical(std::vector<std::size_t> res,
              boundaries_t<real_t>     ext,
              const std::map<std::string, real_t>& = {})
      : MetricBase<D> { res, ext }
      , dr((x1_max - x1_min) / nx1)
      , dtheta((x2_max - x2_min) / nx2)
      , dphi((x3_max - x3_min) / nx3)
      , dr_inv { ONE / dr }
      , dtheta_inv { ONE / dtheta }
      , dphi_inv { ONE / dphi } {
      set_dxMin(find_dxMin());
    }

    ~Spherical() = default;

    /**
     * minimum effective cell size for a given metric (in physical units)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      // for 2D
      auto dx1 { dr };
      auto dx2 { x1_min * dtheta };
      return ONE / math::sqrt(ONE / SQR(dx1) + ONE / SQR(dx2));
    }

    /**
     * metric component with lower indices: h_ij
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto h_(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");
      if constexpr (i == 1 && j == 1) {
        return SQR(dr);
      } else if constexpr (i == 2 && j == 2) {
        return SQR(dtheta) * SQR(x[0] * dr + x1_min);
      } else if constexpr (i == 3 && j == 3) {
        if constexpr (D == Dim::_2D) {
          return SQR(x[0] * dr + x1_min) * SQR(math::sin(x[1] * dtheta + x2_min));
        } else {
          return SQR(dphi) * SQR(x[0] * dr + x1_min) *
                 SQR(math::sin(x[1] * dtheta + x2_min));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * sqrt(h_ij)
     * @param x coordinate array in code units
     */
    template <idx_t i, idx_t j>
    Inline auto sqrt_h_(const coord_t<D>& x) const -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(j > 0 && j <= 3, "Invalid index j");
      if constexpr (i == 1 && j == 1) {
        return dr;
      } else if constexpr (i == 2 && j == 2) {
        return dtheta * (x[0] * dr + x1_min);
      } else if constexpr (i == 3 && j == 3) {
        if constexpr (D == Dim::_2D) {
          return (x[0] * dr + x1_min) * (math::sin(x[1] * dtheta + x2_min));
        } else {
          return dphi * (x[0] * dr + x1_min) * (math::sin(x[1] * dtheta + x2_min));
        }
      } else {
        return ZERO;
      }
    }

    /**
     * sqrt(det(h_ij))
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      } else {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min) *
               math::sin(x[1] * dtheta + x2_min);
      }
    }

    /**
     * sqrt(det(h_ij)) / sin(theta)
     * @param x coordinate array in code units
     */
    Inline auto sqrt_det_h_tilde(const coord_t<D>& x) const -> real_t {
      if constexpr (D == Dim::_2D) {
        return dr * dtheta * SQR(x[0] * dr + x1_min);
      } else {
        return dr * dtheta * dphi * SQR(x[0] * dr + x1_min);
      }
    }

    /**
     * differential area at the pole (used in axisymmetric solvers)
     * @param x1 radial coordinate along the axis (code units)
     */
    Inline auto polar_area(const real_t& x1) const -> real_t {
      return dr * SQR(x1 * dr + x1_min) * (ONE - math::cos(HALF * dtheta));
    }

    /**
     * component-wise coordinate conversions
     */
    template <idx_t i, Crd in, Crd out>
    Inline auto convert(const real_t& x_in) const -> real_t {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert((in == Crd::Cd && (out == Crd::Sph || out == Crd::Ph)) ||
                      ((in == Crd::Sph || in == Crd::Ph) && out == Crd::Cd),
                    "Invalid coordinate conversion");
      if constexpr (in == Crd::Cd && (out == Crd::Sph || out == Crd::Ph)) {
        // code -> sph/phys
        if constexpr (i == 1) {
          return x_in * dr + x1_min;
        } else if constexpr (i == 2) {
          return x_in * dtheta + x2_min;
        } else {
          if constexpr (D != Dim::_3D) {
            return x_in;
          } else {
            return x_in * dphi + x3_min;
          }
        }
      } else {
        // sph/phys -> code
        if constexpr (i == 1) {
          return (x_in - x1_min) * dr_inv;
        } else if constexpr (i == 2) {
          return (x_in - x2_min) * dtheta_inv;
        } else {
          if constexpr (D != Dim::_3D) {
            return x_in;
          } else {
            return (x_in - x3_min) * dphi_inv;
          }
        }
      }
    }

    /**
     * full coordinate conversions
     */
    template <Crd in, Crd out>
    Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(in != Crd::XYZ && out != Crd::XYZ,
                    "Invalid coordinate conversion: use convert_xyz");
      // code <-> sph/phys
      if constexpr (D == Dim::_2D) {
        x_out[0] = convert<1, in, out>(x_in[0]);
        x_out[1] = convert<2, in, out>(x_in[1]);
      } else {
        x_out[0] = convert<1, in, out>(x_in[0]);
        x_out[1] = convert<2, in, out>(x_in[1]);
        x_out[2] = convert<3, in, out>(x_in[2]);
      }
    }

    /**
     * full coordinate conversion to/from cartesian
     */
    template <Crd in, Crd out>
    Inline void convert_xyz(const coord_t<PrtlDim>& x_in,
                            coord_t<PrtlDim>&       x_out) const {
      static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                      (in == Crd::XYZ && out == Crd::Cd),
                    "Invalid coordinate conversion");
      if (in == Crd::Cd && out == Crd::XYZ) {
        // code -> cart
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(x_in[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(x_in[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(x_in[2]);
        x_out[0] = x_Sph[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]);
        x_out[1] = x_Sph[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]);
        x_out[2] = x_Sph[0] * math::cos(x_Sph[1]);
      } else {
        // cart -> code
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = math::sqrt(SQR(x_in[0]) + SQR(x_in[1]) + SQR(x_in[2]));
        x_Sph[1] = static_cast<real_t>(constant::HALF_PI) -
                   math::atan2(x_in[2], math::sqrt(SQR(x_in[0]) + SQR(x_in[1])));
        x_Sph[2] = static_cast<real_t>(constant::PI) -
                   math::atan2(x_in[1], -x_in[0]);
        x_out[0] = convert<1, Crd::Sph, Crd::Cd>(x_Sph[0]);
        x_out[1] = convert<2, Crd::Sph, Crd::Cd>(x_Sph[1]);
        x_out[2] = convert<3, Crd::Sph, Crd::Cd>(x_Sph[2]);
      }
    }

    /**
     * component-wise vector transformations
     * @note tetrad/sph <-> cntrv <-> cov
     */
    template <idx_t i, Idx in, Idx out>
    Inline auto transform(const coord_t<D>& xi, const real_t& v_in) const
      -> real_t {
      static_assert(i > 0 && i <= 3, "Invalid index i");
      static_assert(in != out, "Invalid vector transformation");
      if constexpr ((in == Idx::T && out == Idx::Sph) ||
                    (in == Idx::Sph && out == Idx::T)) {
        // tetrad <-> sph
        return v_in;
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::U) {
        // tetrad/sph -> cntrv
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::Sph)) {
        // cntrv -> tetrad/sph
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr ((in == Idx::T || in == Idx::Sph) && out == Idx::D) {
        // tetrad/sph -> cov
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::Sph)) {
        // cov -> tetrad/sph
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && out == Idx::D) {
        // cntrv -> cov
        return v_in * h_<i, i>(xi);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        // cov -> cntrv
        return v_in / h_<i, i>(xi);
      } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                           (in == Idx::PD && out == Idx::D)) {
        // cntrv -> phys cntrv || phys cov -> cov
        if constexpr (i == 1) {
          return v_in * dr;
        } else if constexpr (i == 2) {
          return v_in * dtheta;
        } else if constexpr (D == Dim::_2D) {
          return v_in;
        } else {
          return v_in * dphi;
        }
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                           (in == Idx::D && out == Idx::PD)) {
        // phys cntrv -> cntrv || cov -> phys cov
        if constexpr (i == 1) {
          return v_in * dr_inv;
        } else if constexpr (i == 2) {
          return v_in * dtheta_inv;
        } else if constexpr (D == Dim::_2D) {
          return v_in;
        } else {
          return v_in * dphi_inv;
        }
      } else {
        raise::KernelError(HERE, "Invalid transformation");
      }
    }

    /**
     * full vector transformations
     */
    template <Idx in, Idx out>
    Inline void transform(const coord_t<D>&      xi,
                          const vec_t<Dim::_3D>& v_in,
                          vec_t<Dim::_3D>&       v_out) const {
      static_assert(in != out, "Invalid vector transformation");
      if constexpr (in != Idx::XYZ && out != Idx::XYZ) {
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = transform<2, in, out>(xi, v_in[1]);
        v_out[2] = transform<3, in, out>(xi, v_in[2]);
      } else {
        raise::KernelError(HERE, "Invalid vector transformation");
      }
    }

    /**
     * full vector transformation to/from cartesian
     */
    template <Idx in, Idx out>
    Inline void transform_xyz(const coord_t<PrtlDim>& xi,
                              const vec_t<Dim::_3D>&  v_in,
                              vec_t<Dim::_3D>&        v_out) const {
      static_assert(in != out, "Invalid vector transformation");
      static_assert(in == Idx::XYZ || out == Idx::XYZ,
                    "Invalid vector transformation");
      if constexpr (in == Idx::T && out == Idx::XYZ) {
        // tetrad -> cart
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
        v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) -
                   v_in[2] * math::sin(x_Sph[2]);
        v_out[1] = v_in[0] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[2] * math::cos(x_Sph[2]);
        v_out[2] = v_in[0] * math::cos(x_Sph[1]) - v_in[1] * math::sin(x_Sph[1]);
      } else if constexpr (in == Idx::XYZ && out == Idx::T) {
        // cart -> tetrad
        coord_t<PrtlDim> x_Sph { ZERO };
        x_Sph[0] = convert<1, Crd::Cd, Crd::Sph>(xi[0]);
        x_Sph[1] = convert<2, Crd::Cd, Crd::Sph>(xi[1]);
        x_Sph[2] = convert<3, Crd::Cd, Crd::Sph>(xi[2]);
        v_out[0] = v_in[0] * math::sin(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::sin(x_Sph[1]) * math::sin(x_Sph[2]) +
                   v_in[2] * math::cos(x_Sph[1]);
        v_out[1] = v_in[0] * math::cos(x_Sph[1]) * math::cos(x_Sph[2]) +
                   v_in[1] * math::cos(x_Sph[1]) * math::sin(x_Sph[2]) -
                   v_in[2] * math::sin(x_Sph[1]);
        v_out[2] = -v_in[0] * math::sin(x_Sph[2]) + v_in[1] * math::cos(x_Sph[2]);
      } else if (in == Idx::XYZ) {
        // cart -> cov/cntrv
        vec_t<Dim::_3D> v_Tetrad { ZERO };
        transform_xyz<Idx::XYZ, Idx::T>(xi, v_in, v_Tetrad);
        if constexpr (D == Dim::_2D) {
          transform<Idx::T, out>({ xi[0], xi[1] }, v_Tetrad, v_out);
        } else {
          transform<Idx::T, out>(xi, v_Tetrad, v_out);
        }
      } else if (out == Idx::XYZ) {
        // cov/cntrv -> cart
        vec_t<Dim::_3D> v_Tetrad { ZERO };
        if constexpr (D == Dim::_2D) {
          transform<in, Idx::T>({ xi[0], xi[1] }, v_in, v_Tetrad);
        } else {
          transform<in, Idx::T>(xi, v_in, v_Tetrad);
        }
        transform_xyz<Idx::T, Idx::XYZ>(xi, v_Tetrad, v_out);
      }
    }
  };

} // namespace metric

#endif // METRICS_SPHERICAL_H

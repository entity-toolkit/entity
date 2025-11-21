/**
 * @file   metrics/metric_box.h
 * @brief  Time-dependent anisotropic "expanding/compressing box" metric
 * @implements
 *   - metric::Box<> : metric::MetricBase<>
 * @namespaces:
 *   - metric::
 *
 * @details
 * Spatial map (Cartesian): x_phys = L(t) x_code,  L(t) = diag(a_x(t), a_y(t), a_z(t))
 * with a_i(t) = (1 + q_i t)^{s_i}
 * */

#ifndef METRICS_METRIC_BOX_H
#define METRICS_METRIC_BOX_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/metric_base.h"

#include <map>
#include <string>
#include <vector>

namespace metric {

  template <Dimension D>
  class Box : public MetricBase<D> {
    // per-axis code cell sizes
    const real_t dx1, dx2, dx3;
    const real_t dx1_inv, dx2_inv, dx3_inv;

    // time-dependent scale factors and Hubble rates H_i = (d a_i/dt)/a_i
    real_t ax{ONE}, ay{ONE}, az{ONE};
    real_t Hx{ZERO}, Hy{ZERO}, Hz{ZERO};

    // parameters for a_i(t) = (1 + q_i t)^{s_i}
    const real_t qx, qy, qz;
    const real_t sx, sy, sz;

  public:
    static constexpr const char*       Label { "box" };
    static constexpr Dimension         PrtlDim { D };
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Box };
    static constexpr ntt::Coord::type  CoordType { ntt::Coord::Cart };

    Inline real_t get_ax() const { return ax; }
    Inline real_t get_ay() const { return ay; }
    Inline real_t get_az() const { return az; }
    Inline real_t get_Hx() const { return Hx; }
    Inline real_t get_Hy() const { return Hy; }
    Inline real_t get_Hz() const { return Hz; }

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

    // scale factors L and L^{-1}
    Inline real_t Li(int i) const {
      if (i == 1) return ax;
      if (i == 2) return ay;
      return az;
    }
    Inline real_t Linv(int i) const {
      if (i == 1) return ONE / ax;
      if (i == 2) return ONE / ay;
      return ONE / az;
    }
    Inline real_t Li(idx_t i, const coord_t<D>&) const {
      if (i == 1) return ax;
      if (i == 2) return ay;
      return az;
    }
    Inline real_t Linv(idx_t i, const coord_t<D>&) const {
      if (i == 1) return ONE / ax;
      if (i == 2) return ONE / ay;
      return ONE / az;
    }

    Box(const std::vector<ncells_t>&         res,
        const boundaries_t<real_t>&          ext,
        const std::map<std::string, real_t>& params)
      : MetricBase<D> { res, ext }
      // per-axis code grid spacing (like Minkowski, but anisotropic)
      , dx1 { (x1_max - x1_min) / nx1 }
      , dx2 { (D == Dim::_1D) ? ONE : (x2_max - x2_min) / nx2 }
      , dx3 { (D == Dim::_3D) ? (x3_max - x3_min) / nx3 : ONE }
      , dx1_inv { ONE / dx1 }
      , dx2_inv { (D == Dim::_1D) ? ONE : ONE / dx2 }
      , dx3_inv { (D == Dim::_3D) ? ONE / dx3 : ONE }
      // expansion parameters
      , qx { params.count("qx") ? params.at("qx") : ZERO }
      , qy { params.count("qy") ? params.at("qy") : ZERO }
      , qz { params.count("qz") ? params.at("qz") : ZERO }
      , sx { params.count("sx") ? params.at("sx") : ZERO }
      , sy { params.count("sy") ? params.at("sy") : ZERO }
      , sz { params.count("sz") ? params.at("sz") : ZERO } {
      set_dxMin(find_dxMin());
    }

    ~Box() = default;

    // call every time step with mid-step time (like other metrics that evolve in time)
    Inline void update(real_t t_mid) {
      const real_t fx = ONE + qx * t_mid;
      const real_t fy = ONE + qy * t_mid;
      const real_t fz = ONE + qz * t_mid;

      ax = math::pow(fx, sx);  Hx = (sx * qx) / fx;
      if constexpr (D != Dim::_1D) { ay = math::pow(fy, sy);  Hy = (sy * qy) / fy; }
      if constexpr (D == Dim::_3D) { az = math::pow(fz, sz);  Hz = (sz * qz) / fz; }
    }

    Inline void update(simtime_t t_mid) { update(static_cast<real_t>(t_mid)); }

    // expose H_i if something needs it
    Inline real_t Hi(int i) const {
      return (i == 1 ? Hx : (i == 2 ? Hy : Hz));
    }
    /**
     * minimum effective cell size (in physical units)
     * match Minkowski’s style: take the smallest code spacing, divide by sqrt(D)
     */
    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      real_t dmin = dx1;
      if constexpr (D != Dim::_1D) dmin = math::min(dmin, dx2);
      if constexpr (D == Dim::_3D) dmin = math::min(dmin, dx3);
      if constexpr (D == Dim::_1D) return dmin;
      if constexpr (D == Dim::_2D) return dmin / math::sqrt(static_cast<real_t>(2));
      return dmin / math::sqrt(static_cast<real_t>(3));
    }

    /**
     * total (instantaneous) physical volume of the region
     */
    [[nodiscard]]
    auto totVolume() const -> real_t override {
      const real_t L1 = (x1_max - x1_min) * ax;
      if constexpr (D == Dim::_1D) {
        return L1;
      } else if constexpr (D == Dim::_2D) {
        const real_t L2 = (x2_max - x2_min) * ay;
        return L1 * L2;
      } else {
        const real_t L2 = (x2_max - x2_min) * ay;
        const real_t L3 = (x3_max - x3_min) * az;
        return L1 * L2 * L3;
      }
    }

    /**
    * metric component with lower indices: h_ij
    */
    template <idx_t i, idx_t j>
    Inline auto h_(const coord_t<D>&) const -> real_t {
      static_assert(i >= 1 && i <= 3 && j >= 1 && j <= 3, "Invalid index");
      if constexpr (i > static_cast<idx_t>(D) || j > static_cast<idx_t>(D)) {
        return (i == j) ? ONE : ZERO;
      }
      if constexpr (i == j) {
        if constexpr (i == 1) return SQR(dx1 * ax);
        if constexpr (i == 2) return SQR(dx2 * ay);
        if constexpr (i == 3) return SQR(dx3 * az);
      }
      return ZERO;
    }
    /**
    * sqrt(h_ij)
    */
    template <idx_t i, idx_t j>
    Inline auto sqrt_h_(const coord_t<D>&) const -> real_t {
      static_assert(i >= 1 && i <= 3 && j >= 1 && j <= 3, "Invalid index");
      if constexpr (i > static_cast<idx_t>(D) || j > static_cast<idx_t>(D)) {
        return (i == j) ? ONE : ZERO;
      }
      if constexpr (i == j) {
        if constexpr (i == 1) return dx1 * ax;
        if constexpr (i == 2) return dx2 * ay;
        if constexpr (i == 3) return dx3 * az;
      }
      return ZERO;
    }

    /**
     * sqrt(det(h_ij))
     */
    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t {
      if constexpr (D == Dim::_1D)   return (dx1 * ax);
      if constexpr (D == Dim::_2D)   return (dx1 * ax) * (dx2 * ay);
      /* Dim::_3D */                 return (dx1 * ax) * (dx2 * ay) * (dx3 * az);
    }

    /**
     * identical style to Minkowski
     */
    template <idx_t i, Crd in, Crd out>
    Inline auto convert(const real_t& x_in) const -> real_t {
      static_assert(in != out, "Invalid coordinate conversion");
      static_assert(i > 0 && i <= static_cast<idx_t>(D), "Invalid index i");
      static_assert((in == Crd::Cd && (out == Crd::XYZ || out == Crd::Ph)) ||
                      ((in == Crd::XYZ || in == Crd::Ph) && out == Crd::Cd),
                    "Invalid coordinate conversion");
      const real_t dx_i = (i == 1 ? dx1 : (i == 2 ? dx2 : dx3));
      if constexpr (in == Crd::Cd && (out == Crd::XYZ || out == Crd::Ph)) {
        if constexpr (i == 1)      return x_in * dx_i + x1_min;
        else if constexpr (i == 2) return x_in * dx_i + x2_min;
        else                       return x_in * dx_i + x3_min;
      } else {
        if constexpr (i == 1)      return (x_in - x1_min) / dx_i;
        else if constexpr (i == 2) return (x_in - x2_min) / dx_i;
        else                       return (x_in - x3_min) / dx_i;
      }
    }

    /**
     * full coordinate conversions (code <-> cart/phys)
     */
    template <Crd in, Crd out>
    Inline void convert(const coord_t<D>& x_in, coord_t<D>& x_out) const {
      static_assert(in != out, "Invalid coordinate conversion");
      if constexpr ((in != Crd::Sph) && (out != Crd::Sph)) {
        if constexpr (D == Dim::_1D) {
          x_out[0] = convert<1, in, out>(x_in[0]);
        } else if constexpr (D == Dim::_2D) {
          x_out[0] = convert<1, in, out>(x_in[0]);
          x_out[1] = convert<2, in, out>(x_in[1]);
        } else {
          x_out[0] = convert<1, in, out>(x_in[0]);
          x_out[1] = convert<2, in, out>(x_in[1]);
          x_out[2] = convert<3, in, out>(x_in[2]);
        }
      } else {
        // no Sph support in this Cartesian metric
        raise::Error("Invalid coordinate conversion for Box metric", HERE);
      }
    }

    /**
     * full coordinate conversions to/from cartesian for particles (like Minkowski)
     */
    template <Crd in, Crd out>
    Inline void convert_xyz(const coord_t<PrtlDim>& x_in,
                            coord_t<PrtlDim>&       x_out) const {
      static_assert((in == Crd::Cd && out == Crd::XYZ) ||
                      (in == Crd::XYZ && out == Crd::Cd),
                    "Invalid coordinate conversion");
      // code <-> cart: reuse convert<in,out>
      convert<in, out>(x_in, x_out);
    }

    /**
     * component-wise vector transformations
     * tetrad/cart <-> cntrv <-> cov
     * Same structure as Minkowski; use per-axis sqrt_h_ and h_
     */
    template <idx_t i, Idx in, Idx out>
    Inline auto transform(const coord_t<D>& xi, const real_t& v_in) const -> real_t {
      static_assert(in != out, "Invalid vector transformation");
      if constexpr (i > static_cast<idx_t>(D)) {
        return v_in;
      }
        const real_t dxa =
          (i == 1 ? dx1 * ax :
          (i == 2 ? dx2 * ay :
                    dx3 * az));
        const real_t dxa_inv = ONE / dxa;

  	// ===============================================
  	// DEBUG PRINT — fires ONLY for one probe location
  	// ===============================================
	if (xi[0] == 0 && xi[1] == 0 && i == 1) {
    	  Kokkos::printf(
            "[transform] i=%d  in=%d  out=%d  v_in=%e\n",
            (int)i, (int)in, (int)out, (double)v_in
          );
          Kokkos::printf(
            "[transform] ax=%e ay=%e  Hx=%e Hy=%e\n",
            (double)get_ax(), (double)get_ay(),
            (double)get_Hx(), (double)get_Hy()
          );
          Kokkos::printf(
            "[transform] dx1=%e  dxa=%e  sqrt_h11=%e  h11=%e\n",
            (double)dx1,
            (double)dxa,
            (double)sqrt_h_<1,1>(xi),
            (double)h_<1,1>(xi)
          );
        }
  	// ===============================================	

	

      if constexpr ((in == Idx::T && out == Idx::XYZ) ||
                    (in == Idx::XYZ && out == Idx::T)) {
        return v_in;
      } else if constexpr ((in == Idx::T || in == Idx::XYZ) && out == Idx::U) {
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && (out == Idx::T || out == Idx::XYZ)) {
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr ((in == Idx::T || in == Idx::XYZ) && out == Idx::D) {
        return v_in * sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::D && (out == Idx::T || out == Idx::XYZ)) {
        return v_in / sqrt_h_<i, i>(xi);
      } else if constexpr (in == Idx::U && out == Idx::D) {
        return v_in * h_<i, i>(xi);
      } else if constexpr (in == Idx::D && out == Idx::U) {
        return v_in / h_<i, i>(xi);
      } else if constexpr ((in == Idx::U && out == Idx::PU) ||
                          (in == Idx::PD && out == Idx::D)) {
        return v_in * dxa;
      } else if constexpr ((in == Idx::PU && out == Idx::U) ||
                       (in == Idx::D && out == Idx::PD)) {
        return v_in * dxa_inv;
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

      if constexpr (D == Dim::_1D) {
        // only x1 is physical; pass-through y,z to avoid instantiating i=2,3
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = v_in[1];
        v_out[2] = v_in[2];
      } else if constexpr (D == Dim::_2D) {
        // x1,x2 are physical; pass-through z to avoid instantiating i=3
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = transform<2, in, out>(xi, v_in[1]);
        v_out[2] = v_in[2];
      } else { // Dim::_3D
        v_out[0] = transform<1, in, out>(xi, v_in[0]);
        v_out[1] = transform<2, in, out>(xi, v_in[1]);
        v_out[2] = transform<3, in, out>(xi, v_in[2]);
      }
    }

    /**
     * full vector transformations to cartesian (compatibility, like Minkowski)
     */
    template <Idx in, Idx out>
    Inline void transform_xyz(const coord_t<PrtlDim>& xi,
                              const vec_t<Dim::_3D>&  v_in,
                              vec_t<Dim::_3D>&        v_out) const {
      static_assert(in == Idx::XYZ || out == Idx::XYZ,
                    "Invalid vector transformation");
      transform<in, out>(xi, v_in, v_out);
    }
  };

} // namespace metric

#endif // METRICS_METRIC_BOX_H

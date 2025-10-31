/**
 * @file   metrics/metric_box.h
 * @brief  Time-dependent anisotropic "expanding/compressing box" metric
 * @implements
 *   - metric::Box<>
 * @namespaces:
 *   - metric::
 *
 * @details
 * Spatial map:  x_phys = L(t) x_code,  L(t) = diag(a_x(t), a_y(t), a_z(t))
 * with a_i(t) = (1 + q_i t)^{s_i},  H_i(t) = (da_i/dt)/a_i = (q_i s_i)/(1+q_i t).
 *
 * Diagonal spatial metric:
 *     h_ij = a_i^2 δ_ij,        h^{ij} = δ^{ij} / a_i^2
 * Determinant and volume factor:
 *     det(h) = (a_x a_y a_z)^2 = Δ^2,   sqrt(det h) = Δ
 *
 * Notes:
 * - Call update(t + 0.5*dt) each step so fields & pushers use mid-step values.
 * - We store a_i, H_i, and Δ (Delta) for fast access by kernels.
 * - For Cartesian coordinates, the various transform/convert helpers are identity
 *   maps; they are defined for interface compatibility with existing kernels.
 */

#ifndef METRICS_METRIC_BOX_H
#define METRICS_METRIC_BOX_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <array>
#include <cmath>

namespace metric {
  using namespace ntt;

  /**
   * @tparam D Simulation dimension (Dim::_1D, _2D, _3D)
   */
  template <Dimension D>
  struct Box : public MetricBase<D> {
    /* ---------------------------------------------------------------------- */
    /* Static interface expected by kernels                                   */
    /* ---------------------------------------------------------------------- */
    static constexpr const char*       Label { "box" };
    static constexpr bool              is_metric  = true;
    static constexpr auto              Dim        = D;
    static constexpr auto              PrtlDim    = D;            // 3D particles on 2D mesh can be enabled elsewhere if needed
    static constexpr Coord::type       CoordType  = Coord::Cart;  // Cartesian basis
    static constexpr ntt::Metric::type MetricType { ntt::Metric::Box };

    /* ---------------------------------------------------------------------- */
    /* Public data                                                            */
    /* ---------------------------------------------------------------------- */
    // Parameters of the scale factors: a_i(t) = (1 + q_i t)^{s_i}
    std::array<real_t, 3> q { ZERO, ZERO, ZERO };
    std::array<real_t, 3> s { ZERO, ZERO, ZERO };

    // Mid-step dynamic state (after update())
    std::array<real_t, 3> a { ONE, ONE, ONE };  // a_i(t_mid)
    std::array<real_t, 3> H { ZERO, ZERO, ZERO }; // H_i(t_mid) = (da_i/dt)/a_i
    real_t                Delta { ONE };          // Δ = a_x a_y a_z

    // Optional extents / resolution if you mirror other metrics’ constructors
    // (kept for interface parity; not used by the simple Cartesian transforms)
    coord_t<D> domain_extent { ZERO };  // physical size along each axis (optional)
    coord_t<D> domain_origin { ZERO };  // physical origin (optional)

    /* ---------------------------------------------------------------------- */
    /* Lifecycle                                                              */
    /* ---------------------------------------------------------------------- */
    Box() = default;

    Box(const coord_t<D>& extent_physical,
              const coord_t<D>& origin_physical = coord_t<D>{ ZERO })
      : domain_extent(extent_physical)
      , domain_origin(origin_physical) {}

    /**
     * @brief Recompute scale factors at mid-step time.
     * @param t_mid  time at n+1/2 (call with t + 0.5*dt)
     */
    Inline void update(real_t t_mid) {
      // a_i = (1 + q_i t)^s_i, H_i = (s_i q_i)/(1 + q_i t)
      for (int i = 0; i < 3; ++i) {
        const real_t f = ONE + q[i] * t_mid;
        a[i] = math::pow(f, s[i]);
        H[i] = (s[i] * q[i]) / f;
      }
      Delta = a[0] * a[1] * a[2];
    }

    /* ---------------------------------------------------------------------- */
    /* Metric accessors used by curl/volume kernels                           */
    /* ---------------------------------------------------------------------- */

    // h_<I,J>(x) : spatial metric components h_{ij} (diagonal)
    template <int I, int J>
    Inline real_t h_(const coord_t<D>&) const {
      static_assert(I >= 1 && I <= 3 && J >= 1 && J <= 3, "Index out of range");
      if constexpr (I == J) {
        return SQR(a[I - 1]);  // h_ii = a_i^2
      } else {
        return ZERO;           // h_ij = 0 for i != j
      }
    }

    // Inverse metric h^{ij} if needed
    template <int I, int J>
    Inline real_t hU_(const coord_t<D>&) const {
      static_assert(I >= 1 && I <= 3 && J >= 1 && J <= 3, "Index out of range");
      if constexpr (I == J) {
        return ONE / SQR(a[I - 1]);  // h^ii = 1/a_i^2
      } else {
        return ZERO;
      }
    }

    // sqrt(det h) = Δ (note: det h = Δ^2)
    Inline real_t sqrt_det_h(const coord_t<D>&) const {
      return Delta;
    }

    /* ---------------------------------------------------------------------- */
    /* Coordinate & vector transforms (Cartesian -> identity)                 */
    /* ---------------------------------------------------------------------- */

    // Convert between coordinate representations (Cd/XYZ/Ph).
    // For Cartesian, treat as identity to match Minkowski behavior.
    template <Crd From, Crd To>
    Inline void convert_xyz(const coord_t<D>& xin, coord_t<D>& xout) const {
      for (unsigned i = 0; i < static_cast<unsigned>(D); ++i) xout[i] = xin[i];
    }

    // Convert a single coordinate component (axis AX) between systems.
    template <int AX, Crd From, Crd To>
    Inline real_t convert(real_t x) const {
      static_assert(AX >= 1 && AX <= 3, "Axis out of range");
      return x; // identity in Cartesian
    }

    // Transform a vector between index/basis types (Idx::U, Idx::XYZ).
    // In Cartesian, this is the identity mapping component-wise.
    template <Idx From, Idx To>
    Inline void transform_xyz(const coord_t<D>&,
                              const vec_t<Dim::_3D>& vin,
                              vec_t<Dim::_3D>&       vout) const {
      vout[0] = vin[0];
      vout[1] = vin[1];
      vout[2] = vin[2];
    }

    /* ---------------------------------------------------------------------- */
    /* Helpers for E/B scaling in primed variables                             */
    /* ---------------------------------------------------------------------- */

    Inline real_t Li(int i)   const { return a[i]; }       // multiply primed -> physical
    Inline real_t Linv(int i) const { return ONE / a[i]; } // physical -> primed

    [[nodiscard]]
    auto find_dxMin() const -> real_t override {
      return 0;
    }

    /**
     * total volume of the region described by the metric (in physical units)
     */
    [[nodiscard]]
    auto totVolume() const -> real_t override {
      return 0;
    }
  };

   

} // namespace metric

#endif // METRICS_METRIC_BOX_H

#ifndef FRAMEWORK_METRICS_MINKOWSKI_H
#define FRAMEWORK_METRICS_MINKOWSKI_H

#include "global.h"
#include "metric.h"

#include <cmath>

namespace ntt {
  /**
   * Minkowski metric (cartesian system): diag(-1, 1, 1, 1).
   * Cell sizes in each direction dx1 = dx2 = dx3 are equal.
   *
   * @tparam D dimension.
   */
  template <Dimension D>
  class Minkowski : virtual public Metric<D> {
  private:
    const real_t dx, dx_sqr, inv_dx;

  public:
    Minkowski(std::vector<std::size_t> resolution, std::vector<real_t> extent)
      : Metric<D> {"minkowski", resolution, extent},
        dx((this->x1_max - this->x1_min) / this->nx1),
        dx_sqr(dx * dx),
        inv_dx(ONE / dx) {}
    ~Minkowski() = default;

    auto findSmallestCell() const -> real_t override { return dx / std::sqrt(static_cast<real_t>(D)); }

    Inline auto h_11(const coord_t<D>&) const -> real_t override { return dx_sqr; }
    Inline auto h_22(const coord_t<D>&) const -> real_t override { return dx_sqr; }
    Inline auto h_33(const coord_t<D>&) const -> real_t override { return dx_sqr; }

    Inline auto sqrt_det_h(const coord_t<D>&) const -> real_t override { return dx_sqr * dx; }

    Inline void x_Code2Cart(const coord_t<D>&, coord_t<D>&) const override;

    Inline void v_Hat2Cntrv(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const override;
    Inline void v_Cntrv2Hat(const coord_t<D>&, const vec_t<Dimension::THREE_D>&, vec_t<Dimension::THREE_D>&) const override;

    // todo
    Inline void x_Code2Sph(const coord_t<D>&, coord_t<D>&) const override {};

    // defaults
    Inline auto h_12(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_12(x); }
    Inline auto h_13(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_13(x); }
    Inline auto h_21(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_21(x); }
    Inline auto h_23(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_23(x); }
    Inline auto h_31(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_31(x); }
    Inline auto h_32(const coord_t<D>& x) const -> real_t override { return Metric<D>::h_32(x); }
    Inline auto polar_area(const coord_t<D>& x) const -> real_t override { return Metric<D>::polar_area(x); }
  };

  // * * * * * * * * * * * * * * *
  // vector transformations
  // * * * * * * * * * * * * * * *
  template <Dimension D>
  Inline void Minkowski<D>::v_Hat2Cntrv(const coord_t<D>& xi,
                                        const vec_t<Dimension::THREE_D>& vi_hat,
                                        vec_t<Dimension::THREE_D>& vi) const {
    vi[0] = vi_hat[0] / std::sqrt(h_11(xi));
    vi[1] = vi_hat[1] / std::sqrt(h_22(xi));
    vi[2] = vi_hat[2] / std::sqrt(h_33(xi));
  }
  
  template <Dimension D>
  Inline void Minkowski<D>::v_Cntrv2Hat(const coord_t<D>& xi,
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
  Inline void Minkowski<Dimension::ONE_D>::x_Code2Cart(const coord_t<Dimension::ONE_D>& xi,
                                                       coord_t<Dimension::ONE_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
  }

  // * * * * * * * * * * * * * * *
  // 2D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Minkowski<Dimension::TWO_D>::x_Code2Cart(const coord_t<Dimension::TWO_D>& xi,
                                                     coord_t<Dimension::TWO_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
  }

  // * * * * * * * * * * * * * * *
  // 3D:
  // * * * * * * * * * * * * * * *
  template <>
  Inline void Minkowski<Dimension::THREE_D>::x_Code2Cart(const coord_t<Dimension::THREE_D>& xi,
                                                       coord_t<Dimension::THREE_D>& x) const {
    x[0] = xi[0] * dx + this->x1_min;
    x[1] = xi[1] * dx + this->x2_min;
    x[2] = xi[2] * dx + this->x3_min;
  }

  } // namespace ntt

#endif

#ifndef OBJECTS_GEOMETRY_QSPHERICAL_H
#define OBJECTS_GEOMETRY_QSPHERICAL_H

#include "global.h"
#include "grid.h"

#include <tuple>
#include <cassert>

namespace ntt {

  // xi, eta, phi = log(r-r0), f(h, theta), phi
  template <Dimension D>
  struct QSphericalSystem : public Grid<D> {
    const real_t r0;
    const real_t h;
    const real_t dxi, deta, dphi;
    const real_t xi_min;
    const real_t dxi_sqr, deta_sqr, dphi_sqr;

    QSphericalSystem(std::vector<std::size_t> resolution,
                     std::vector<real_t> extent,
                     const real_t& r0_,
                     const real_t& h_)
      : Grid<D> {"qspherical", resolution, extent},
        r0 {r0_},
        h {h_},
        xi_min {std::log(this->x1_min - r0)},
        dxi((std::log(this->x1_max - r0) - xi_min) / (real_t)(this->Nx1)),
        deta(PI / (real_t)(this->Nx2)),
        dphi(TWO_PI / (real_t)(this->Nx3)),
        dxi_sqr(dxi * dxi),
        deta_sqr(deta * deta),
        dphi_sqr(dphi * dphi) {}
    ~QSphericalSystem() = default;

    auto findSmallestCell() const -> real_t {
      if constexpr (D == TWO_D) {
        using index_t = NTTArray<real_t**>::size_type;
        real_t min_dx {-1.0};
        for (index_t i {0}; i < this->Nx1; ++i) {
          for (index_t j {0}; j < this->Nx2; ++j) {
            auto i_ {(real_t)(i)};
            auto j_ {(real_t)(j)};
            real_t dx1_ {this->h11(i_, j_)};
            real_t dx2_ {this->h22(i_, j_)};
            real_t dx = 1.0 / std::sqrt(1.0 / dx1_ + 1.0 / dx2_);
            if ((min_dx >= dx) || (min_dx < 0.0)) {
              min_dx = dx;
            }
          }
        }
        return min_dx;
      } else {
        throw std::logic_error("# Error: min cell finding not implemented for 3D qspherical.");
      }
    }

    // * * * * * * * * * * * * * * *
    // 2D:
    // * * * * * * * * * * * * * * *
    // coordinate transformations
    // conversion from code units (CU) to cartesian (Cart)
    Inline auto coord_CU_to_Cart(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> override {
      auto [r, theta] = coord_CU_to_Sph(x1, x2);
      return {r * std::sin(theta), r * std::cos(theta)};
    }

    // conversion to spherical
    Inline auto coord_CU_to_Sph(const real_t& x1, const real_t& x2) const -> std::tuple<real_t, real_t> {
      auto xi {x1 * dxi + xi_min};
      auto eta {x2 * deta};
      return {r0 + std::exp(xi), eta + 2.0 * h * eta * (PI - 2.0 * eta) * (PI - eta) * INV_PI_SQR};
    }

    // metric coefficients
    Inline auto h11(const real_t& x1, const real_t&) const -> real_t {
      auto xi {x1 * dxi + xi_min};
      return dxi_sqr * std::exp(2.0 * xi);
    }

    Inline auto h22(const real_t& x1, const real_t& x2) const -> real_t {
      auto xi {x1 * dxi + xi_min};
      auto r {r0 + std::exp(xi)};

      auto eta {x2 * deta};
      auto dtheta_deta {(ONE + 2.0 * h + 12.0 * h * (eta * INV_PI) * ((eta * INV_PI) - ONE))};

      return deta_sqr * r * r * dtheta_deta * dtheta_deta;
    }

    Inline auto h33(const real_t& x1, const real_t& x2) const -> real_t {
      auto xi {x1 * dxi + xi_min};
      auto r {r0 + std::exp(xi)};
      auto eta {x2 * deta};
      auto theta {eta + 2.0 * h * eta * (PI - 2.0 * eta) * (PI - eta) * INV_PI_SQR};
      auto sin_theta {std::sin(theta)};
      return r * r * sin_theta * sin_theta;
    }

    // det of metric
    Inline auto sqrt_det_h(const real_t& x1, const real_t& x2) const -> real_t {
      auto xi {x1 * dxi + xi_min};
      auto r {r0 + std::exp(xi)};
      auto eta {x2 * deta};
      auto theta {eta + 2.0 * h * eta * (PI - 2.0 * eta) * (PI - eta) * INV_PI_SQR};
      auto sin_theta {std::sin(theta)};
      auto dtheta_deta {(ONE + 2.0 * h + 12.0 * h * (eta * INV_PI) * ((eta * INV_PI) - ONE))};
      return dxi * deta * std::exp(xi) * r * r * sin_theta * dtheta_deta;
    }

    // area at poles
    Inline auto polar_area(const real_t& x1, const real_t& x2) const -> real_t {
      auto xi {x1 * dxi + xi_min};
      auto r {r0 + std::exp(xi)};
      auto eta {x2 * deta};
      auto theta {eta + 2.0 * h * eta * (PI - 2.0 * eta) * (PI - eta) * INV_PI_SQR};
      return std::exp(xi) * r * r * (ONE - std::cos(theta));
    }

    // * * * * * * * * * * * * * * *
    // 3D:
    // * * * * * * * * * * * * * * *
    // TODO
    // - - - - - - - - - - - - - - -
  };

} // namespace ntt

#endif

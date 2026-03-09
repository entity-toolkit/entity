/**
 * @file kernels/particle_moments.hpp
 * @brief Algorithm for computing different moments from particle distribution
 * @implements
 *   - kernel::ParticleMoments_kernel<>
 * @namespaces:
 *   - kernel::
 */

#ifndef KERNELS_PARTICLE_MOMENTS_HPP
#define KERNELS_PARTICLE_MOMENTS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "metrics/traits.h"

#include <vector>

namespace kernel {
  using namespace ntt;

  template <FldsID::type F>
  auto get_contrib(float mass, float charge) -> real_t {
    if constexpr (F == FldsID::Rho) {
      return mass;
    } else if constexpr (F == FldsID::Charge) {
      return charge;
    } else {
      return ONE;
    }
  }

  template <SimEngine::type S, class M, FldsID::type F, unsigned short N>
    requires metric::traits::HasD<M> && metric::traits::HasSqrtDetH<M> &&
             ((S == SimEngine::SRPIC && metric::traits::HasTransformXYZ<M>) ||
              (S == SimEngine::GRPIC && metric::traits::HasTransform<M>))
  class ParticleMoments_kernel {
    static constexpr auto D = M::Dim;

    static_assert((F == FldsID::Rho) || (F == FldsID::Charge) || (F == FldsID::N) ||
                    (F == FldsID::Nppc) || (F == FldsID::T) || (F == FldsID::V),
                  "Invalid field ID");

    const unsigned short     c1, c2;
    scatter_ndfield_t<D, N>  Buff;
    const idx_t              buff_idx;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const array_t<short*>    tag;
    const float              mass;
    const float              charge;
    const bool               use_weights;
    const M                  metric;
    const int                ni2;
    const unsigned short     window;

    const real_t contrib;
    const real_t smooth;
    bool         is_axis_i2min { false }, is_axis_i2max { false };

  public:
    ParticleMoments_kernel(const std::vector<unsigned short>& components,
                           const scatter_ndfield_t<D, N>&     scatter_buff,
                           idx_t                              buff_idx,
                           const array_t<int*>&               i1,
                           const array_t<int*>&               i2,
                           const array_t<int*>&               i3,
                           const array_t<prtldx_t*>&          dx1,
                           const array_t<prtldx_t*>&          dx2,
                           const array_t<prtldx_t*>&          dx3,
                           const array_t<real_t*>&            ux1,
                           const array_t<real_t*>&            ux2,
                           const array_t<real_t*>&            ux3,
                           const array_t<real_t*>&            phi,
                           const array_t<real_t*>&            weight,
                           const array_t<short*>&             tag,
                           float                              mass,
                           float                              charge,
                           bool                               use_weights,
                           const M&                           metric,
                           const boundaries_t<FldsBC>&        boundaries,
                           ncells_t                           ni2,
                           real_t                             inv_n0,
                           unsigned short                     window)
      : c1 { (components.size() > 0) ? components[0]
                                     : static_cast<unsigned short>(0) }
      , c2 { (components.size() == 2) ? components[1]
                                      : static_cast<unsigned short>(0) }
      , Buff { scatter_buff }
      , buff_idx { buff_idx }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , weight { weight }
      , tag { tag }
      , mass { mass }
      , charge { charge }
      , use_weights { use_weights }
      , metric { metric }
      , ni2 { static_cast<int>(ni2) }
      , window { window }
      , contrib { get_contrib<F>(mass, charge) }
      , smooth { inv_n0 / (real_t)(math::pow(TWO * (real_t)window + ONE,
                                             static_cast<int>(D))) } {
      raise::ErrorIf(buff_idx >= N, "Invalid buffer index", HERE);
      raise::ErrorIf(window > N_GHOSTS, "Window size too large", HERE);
      raise::ErrorIf(((F == FldsID::Rho) || (F == FldsID::Charge)) && (mass == ZERO),
                     "Rho & Charge for massless particles not defined",
                     HERE);
      if constexpr ((M::CoordType != Coord::Cart) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
      }
    }

    Inline void operator()(index_t p) const {
      if (tag(p) == ParticleTag::dead) {
        return;
      }
      real_t coeff { ZERO };
      if constexpr (F == FldsID::T) {
        real_t          u0 { ZERO };
        // for stress-energy tensor
        vec_t<Dim::_3D> u_Phys { ZERO };
        if constexpr (S == SimEngine::SRPIC) {
          // SR
          // stress-energy tensor for SR is computed in the tetrad (hatted) basis
          if constexpr (M::CoordType == Coord::Cart) {
            u_Phys[0] = ux1(p);
            u_Phys[1] = ux2(p);
            u_Phys[2] = ux3(p);
          } else {
            static_assert(D != Dim::_1D, "non-Cartesian SRPIC 1D");
            coord_t<M::PrtlDim> x_Code { ZERO };
            x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            if constexpr (D == Dim::_3D) {
              x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
            } else {
              x_Code[2] = phi(p);
            }
            metric.template transform_xyz<Idx::XYZ, Idx::T>(
              x_Code,
              { ux1(p), ux2(p), ux3(p) },
              u_Phys);
          }
          if (mass == ZERO) {
            u0 = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
          } else {
            u0 = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
          }
        } else {
          // GR
          // stress-energy tensor for GR is computed in contravariant basis
          static_assert(D != Dim::_1D, "GRPIC 1D");
          coord_t<D> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          }
          // raise full covariant 4-vector to get correct contravariant u^i
          // u^i != h^{ij} u_j
          const real_t    u_0_cov { metric.u_0(x_Code,
                                               { ux1(p), ux2(p), ux3(p) },
                                            (mass == ZERO) ? ZERO : ONE) };
          vec_t<Dim::_4D> u_cntrv_4d { ZERO };
          metric.template transform_4d<Idx::D, Idx::U>(
            x_Code,
            { u_0_cov, ux1(p), ux2(p), ux3(p) },
            u_cntrv_4d);
          // in GR: u^0 = Gamma/alpha
          u0 = u_cntrv_4d[0];
          metric.template transform<Idx::U, Idx::PU>(
            x_Code,
            { u_cntrv_4d[1], u_cntrv_4d[2], u_cntrv_4d[3] },
            u_Phys);
        }
        // compute the corresponding moment
        // T^μν = m * u^0 * v^{c1} * v^{c2},
        // where v^0 = 1, v^i = u^i / u^0 (coordinate 3-velocity)
        coeff = ((mass == ZERO) ? ONE : mass) * u0;
#pragma unroll
        for (const auto& c : { c1, c2 }) {
          if (c != 0) {
            coeff *= u_Phys[c - 1] / u0;
          }
        }
      } else if constexpr (F == FldsID::V) {
        if constexpr (S == SimEngine::SRPIC) {
          // SR: bulk velocity
          real_t          gamma { ZERO };
          // for bulk 3vel (tetrad basis)
          vec_t<Dim::_3D> u_Phys { ZERO };
          if constexpr (M::CoordType == Coord::Cart) {
            u_Phys[0] = ux1(p);
            u_Phys[1] = ux2(p);
            u_Phys[2] = ux3(p);
          } else {
            coord_t<M::PrtlDim> x_Code { ZERO };
            x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
            x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
            if constexpr (D == Dim::_3D) {
              x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
            } else {
              x_Code[2] = phi(p);
            }
            metric.template transform_xyz<Idx::XYZ, Idx::T>(x_Code,
                                                            { ux1(p), ux2(p), ux3(p) },
                                                            u_Phys);
          }
          if (mass == ZERO) {
            gamma = NORM(u_Phys[0], u_Phys[1], u_Phys[2]);
          } else {
            gamma = math::sqrt(ONE + NORM_SQR(u_Phys[0], u_Phys[1], u_Phys[2]));
          }
          // compute the corresponding moment
          coeff = (mass == ZERO ? ONE : mass) * u_Phys[c1 - 1] / gamma;
        } else {
          // GR: Eckart frame flux N^μ = m * u^μ / u^0
          static_assert(D != Dim::_1D, "GRPIC 1D");
          vec_t<Dim::_3D> u_Phys { ZERO };
          real_t          u0 { ZERO };
          coord_t<D> x_Code { ZERO };
          x_Code[0] = static_cast<real_t>(i1(p)) + static_cast<real_t>(dx1(p));
          x_Code[1] = static_cast<real_t>(i2(p)) + static_cast<real_t>(dx2(p));
          if constexpr (D == Dim::_3D) {
            x_Code[2] = static_cast<real_t>(i3(p)) + static_cast<real_t>(dx3(p));
          }
          // raise full covariant 4-vector to get correct contravariant u^i
          // u^i != h^{ij} u_j
          const real_t    u_0_cov { metric.u_0(x_Code,
                                               { ux1(p), ux2(p), ux3(p) },
                                            (mass == ZERO) ? ZERO : ONE) };
          vec_t<Dim::_4D> u_cntrv_4d { ZERO };
          metric.template transform_4d<Idx::D, Idx::U>(
            x_Code,
            { u_0_cov, ux1(p), ux2(p), ux3(p) },
            u_cntrv_4d);
          // in GR: u^0 = Gamma/alpha
          u0 = u_cntrv_4d[0];
          // Deposit flux N^μ = mass * u^μ / u^0
          if (c1 == 0) {
            // u^0 component
            coeff = (mass == ZERO ? ONE : mass);
          } else {
            // u^i component: mass * u^i / u^0
            coeff = (mass == ZERO ? ONE : mass) * u_cntrv_4d[c1] / u0;
          }
        }
      } else {
        // for other cases, use the `contrib` defined above
        coeff = contrib;
      }
      if constexpr (F != FldsID::Nppc) {
        // for nppc calculation ...
        // ... do not take volume, weights or smoothing into account
        if constexpr (D == Dim::_1D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF });
        } else if constexpr (D == Dim::_2D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF });
        } else if constexpr (D == Dim::_3D) {
          coeff *= smooth /
                   metric.sqrt_det_h({ static_cast<real_t>(i1(p)) + HALF,
                                       static_cast<real_t>(i2(p)) + HALF,
                                       static_cast<real_t>(i3(p)) + HALF });
        }
        if (use_weights) {
          coeff *= weight(p);
        }
      }
      auto buff_access = Buff.access();
      if constexpr (D == Dim::_1D) {
        for (auto di1 { -window }; di1 <= window; ++di1) {
          buff_access(i1(p) + di1 + N_GHOSTS, buff_idx) += coeff;
        }
      } else if constexpr (D == Dim::_2D) {
        for (auto di2 { -window }; di2 <= window; ++di2) {
          for (auto di1 { -window }; di1 <= window; ++di1) {
            if constexpr (M::CoordType == Coord::Cart) {
              buff_access(i1(p) + di1 + N_GHOSTS,
                          i2(p) + di2 + N_GHOSTS,
                          buff_idx) += coeff;
            } else {
              // reflect contribution at axes
              if (is_axis_i2min && (i2(p) + di2 < 0)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            N_GHOSTS - (i2(p) + di2),
                            buff_idx) += coeff;
              } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            buff_idx) += coeff;
              }
            }
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (auto di3 { -window }; di3 <= window; ++di3) {
          for (auto di2 { -window }; di2 <= window; ++di2) {
            for (auto di1 { -window }; di1 <= window; ++di1) {
              if constexpr (M::CoordType == Coord::Cart) {
                buff_access(i1(p) + di1 + N_GHOSTS,
                            i2(p) + di2 + N_GHOSTS,
                            i3(p) + di3 + N_GHOSTS,
                            buff_idx) += coeff;
              } else {
                // reflect contribution at axes
                if (is_axis_i2min && (i2(p) + di2 < 0)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              N_GHOSTS - (i2(p) + di2),
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else if (is_axis_i2max && (i2(p) + di2 >= ni2)) {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              2 * ni2 - (i2(p) + di2) + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                } else {
                  buff_access(i1(p) + di1 + N_GHOSTS,
                              i2(p) + di2 + N_GHOSTS,
                              i3(p) + di3 + N_GHOSTS,
                              buff_idx) += coeff;
                }
              }
            }
          }
        }
      }
    }
  };

  template <Dimension D, unsigned short N>
  class NormalizeVectorByRho_kernel {
    const ndfield_t<D, N> Rho;
    ndfield_t<D, N>       Vector;
    const unsigned short  c_rho, c_v1, c_v2, c_v3;

  public:
    NormalizeVectorByRho_kernel(const ndfield_t<D, N>& rho,
                                const ndfield_t<D, N>& vector,
                                unsigned short         crho,
                                unsigned short         cv1,
                                unsigned short         cv2,
                                unsigned short         cv3)
      : Rho { rho }
      , Vector { vector }
      , c_rho { crho }
      , c_v1 { cv1 }
      , c_v2 { cv2 }
      , c_v3 { cv3 } {
      raise::ErrorIf(c_rho >= N or c_v1 >= N or c_v2 >= N or c_v3 >= N,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_rho == c_v1 or c_rho == c_v2 or c_rho == c_v3,
                     "Invalid component index",
                     HERE);
      raise::ErrorIf(c_v1 == c_v2 or c_v1 == c_v3 or c_v2 == c_v3,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (not cmp::AlmostZero(Rho(i1, c_rho))) {
          Vector(i1, c_v1) /= Rho(i1, c_rho);
          Vector(i1, c_v2) /= Rho(i1, c_rho);
          Vector(i1, c_v3) /= Rho(i1, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "1D implementation of NormalizeVectorByRho_kernel called for non-1D");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (not cmp::AlmostZero(Rho(i1, i2, c_rho))) {
          Vector(i1, i2, c_v1) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v2) /= Rho(i1, i2, c_rho);
          Vector(i1, i2, c_v3) /= Rho(i1, i2, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of NormalizeVectorByRho_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (not cmp::AlmostZero(Rho(i1, i2, i3, c_rho))) {
          Vector(i1, i2, i3, c_v1) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v2) /= Rho(i1, i2, i3, c_rho);
          Vector(i1, i2, i3, c_v3) /= Rho(i1, i2, i3, c_rho);
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of NormalizeVectorByRho_kernel called for non-3D");
      }
    }
  };

  template <Dimension D, class M, unsigned short N>
    requires metric::traits::HasD<M> && metric::traits::HasTransform<M>
  class Normalize4VelocityByNorm_kernel {
    // Normalizes 4-momentum flux to Eckart frame velocity
    // V^μ = N^μ / sqrt(-N_ν N^ν)
    const ndfield_t<D, N> Flux;    // momentum flux N^μ
    ndfield_t<D, N>       Vector;  // Eckart 4-velocity
    const unsigned short  c_u0, c_u1, c_u2, c_u3;  // 4-velocity component indices
    const M               metric;

  public:
    Normalize4VelocityByNorm_kernel(const ndfield_t<D, N>& flux,
                                    const ndfield_t<D, N>& vector,
                                    unsigned short         cu0,
                                    unsigned short         cu1,
                                    unsigned short         cu2,
                                    unsigned short         cu3,
                                    const M&               metric)
      : Flux { flux }
      , Vector { vector }
      , c_u0 { cu0 }
      , c_u1 { cu1 }
      , c_u2 { cu2 }
      , c_u3 { cu3 }
      , metric { metric } {
      raise::ErrorIf(c_u0 >= N or c_u1 >= N or c_u2 >= N or c_u3 >= N,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;

        vec_t<Dim::_4D> N_cntrv { ZERO };
        N_cntrv[0] = Flux(i1, i2, c_u0);
        N_cntrv[1] = Flux(i1, i2, c_u1);
        N_cntrv[2] = Flux(i1, i2, c_u2);
        N_cntrv[3] = Flux(i1, i2, c_u3);

        // ZAMO fallback for empty or pathological cells
        const auto zamo_fallback_2d = [&]() {
          const real_t al = metric.alpha(x_Code);
          const real_t b1 = metric.beta1(x_Code);
          Vector(i1, i2, c_u0) = ONE / al;
          Vector(i1, i2, c_u1) = -b1 / al;
          Vector(i1, i2, c_u2) = ZERO;
          Vector(i1, i2, c_u3) = ZERO;
        };

        // rest frame for empty cells
        if (cmp::AlmostZero(N_cntrv[0])) {
          zamo_fallback_2d();
          return;
        }

        vec_t<Dim::_4D> N_cov { ZERO };
        // Compute N_i = g_ij N^j
        metric.template transform_4d<Idx::U, Idx::D>(x_Code, N_cntrv, N_cov);

        // Compute N_ν N^ν = g_μν N^μ N^ν (should be negative for timelike)
        real_t N_norm_sq { N_cov[0] * N_cntrv[0] + N_cov[1] * N_cntrv[1] + N_cov[2] * N_cntrv[2] + N_cov[3] * N_cntrv[3] };

        // ZAMO fallback for cells with insufficient particles
        if (cmp::AlmostZero(N_norm_sq) || N_norm_sq >= ZERO) {
          zamo_fallback_2d();
          return;
        }

        real_t norm = math::sqrt(-N_norm_sq);

        if (not cmp::AlmostZero(norm)) {
          Vector(i1, i2, c_u0) = N_cntrv[0] / norm;
          Vector(i1, i2, c_u1) = N_cntrv[1] / norm;
          Vector(i1, i2, c_u2) = N_cntrv[2] / norm;
          Vector(i1, i2, c_u3) = N_cntrv[3] / norm;
        } else {
          Vector(i1, i2, c_u0) = ONE;
          Vector(i1, i2, c_u1) = ZERO;
          Vector(i1, i2, c_u2) = ZERO;
          Vector(i1, i2, c_u3) = ZERO;
        }
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of Normalize4VelocityByNorm_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;
        x_Code[2] = COORD(i3) + HALF;

        vec_t<Dim::_4D> N_cntrv { ZERO };
        N_cntrv[0] = Flux(i1, i2, i3, c_u0);  // N^0
        N_cntrv[1] = Flux(i1, i2, i3, c_u1);  // N^1
        N_cntrv[2] = Flux(i1, i2, i3, c_u2);  // N^2
        N_cntrv[3] = Flux(i1, i2, i3, c_u3);  // N^3

        // Rest frame for empty cells
        if (cmp::AlmostZero(N_cntrv[0])) {
          Vector(i1, i2, i3, c_u0) = ONE;
          Vector(i1, i2, i3, c_u1) = ZERO;
          Vector(i1, i2, i3, c_u2) = ZERO;
          Vector(i1, i2, i3, c_u3) = ZERO;
          return;
        }

        vec_t<Dim::_4D> N_cov { ZERO };
        // Compute N_i = g_ij N^j
        metric.template transform_4d<Idx::U, Idx::D>(x_Code, N_cntrv, N_cov);

        // Compute N_ν N^ν = g_μν N^μ N^ν (should be negative for timelike)
        real_t N_norm_sq { N_cov[0] * N_cntrv[0] + N_cov[1] * N_cntrv[1] + N_cov[2] * N_cntrv[2] + N_cov[3] * N_cntrv[3] };

        // Set to rest frame for cells with insufficient particles
        if (cmp::AlmostZero(N_norm_sq) || N_norm_sq >= ZERO) {
          Vector(i1, i2, i3, c_u0) = ONE;
          Vector(i1, i2, i3, c_u1) = ZERO;
          Vector(i1, i2, i3, c_u2) = ZERO;
          Vector(i1, i2, i3, c_u3) = ZERO;
          return;
        }

        real_t norm = math::sqrt(-N_norm_sq);  // sqrt(-N_ν N^ν) for timelike vector

        if (not cmp::AlmostZero(norm)) {
          Vector(i1, i2, i3, c_u0) = N_cntrv[0] / norm;
          Vector(i1, i2, i3, c_u1) = N_cntrv[1] / norm;
          Vector(i1, i2, i3, c_u2) = N_cntrv[2] / norm;
          Vector(i1, i2, i3, c_u3) = N_cntrv[3] / norm;
        } else {
          Vector(i1, i2, i3, c_u0) = ONE;
          Vector(i1, i2, i3, c_u1) = ZERO;
          Vector(i1, i2, i3, c_u2) = ZERO;
          Vector(i1, i2, i3, c_u3) = ZERO;
        }
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of Normalize4VelocityByNorm_kernel called for non-3D");
      }
    }
  };


  template <Dimension D, class M, unsigned short N_T, unsigned short N_U, unsigned short N_em, unsigned short N_aux>
    requires metric::traits::HasD<M> && metric::traits::HasTransform<M>
  class FluidFrameStressEnergy_kernel {
    // Stress-energy tensor in fluid frame, rotated to B-field aligned coordinates
    // T_phys and U_phys are in physical basis
    ndfield_t<D, N_T>   T_in;
    ndfield_t<D, N_U>   U_in;
    ndfield_t<D, N_em>  EM_in;
    ndfield_t<D, N_aux> AUX_in;
    ndfield_t<D, N_T>   T_out;

    unsigned short t_comp[10];  // T components: T00, T01, T02, T03, T11, T12, T13, T22, T23, T33
    unsigned short u_comp[4];   // U components: U0, U1, U2, U3
    unsigned short out_comp[10];  // Output components
    const M metric;

  public:
    FluidFrameStressEnergy_kernel(ndfield_t<D, N_T>&   t_in, ndfield_t<D, N_U>&   u_in,
                                  ndfield_t<D, N_em>&  em_in, ndfield_t<D, N_aux>& aux_in,
                                  ndfield_t<D, N_T>&   t_out,
                                  unsigned short t00, unsigned short t01, unsigned short t02,
                                  unsigned short t03, unsigned short t11, unsigned short t12,
                                  unsigned short t13, unsigned short t22, unsigned short t23,
                                  unsigned short t33, unsigned short u0, unsigned short u1,
                                  unsigned short u2, unsigned short u3, unsigned short out00,
                                  unsigned short out01, unsigned short out02, unsigned short out03,
                                  unsigned short out11, unsigned short out12, unsigned short out13,
                                  unsigned short out22, unsigned short out23, unsigned short out33,
                                  const M& m)
      : T_in { t_in }, U_in { u_in }, EM_in { em_in }, AUX_in { aux_in }, T_out { t_out }
      , metric { m } {
      t_comp[0] = t00;  t_comp[1] = t01;  t_comp[2] = t02;  t_comp[3] = t03;
      t_comp[4] = t11;  t_comp[5] = t12;  t_comp[6] = t13;
      t_comp[7] = t22;  t_comp[8] = t23;  t_comp[9] = t33;
      u_comp[0] = u0;  u_comp[1] = u1;  u_comp[2] = u2;  u_comp[3] = u3;
      out_comp[0] = out00;  out_comp[1] = out01;  out_comp[2] = out02;  out_comp[3] = out03;
      out_comp[4] = out11;  out_comp[5] = out12;  out_comp[6] = out13;
      out_comp[7] = out22;  out_comp[8] = out23;  out_comp[9] = out33;
    }

    Inline real_t getTComp(index_t i1, index_t i2, int i, int j) const {
      if (i > j) { return getTComp(i1, i2, j, i); }
      int idx = i * 4 - i * (i - 1) / 2 + j - i;
      if constexpr (D == Dim::_2D) {
        return T_in(i1, i2, t_comp[idx]);
      } else {
        return T_in(i1, i2, 0, t_comp[idx]);
      }
    }

    Inline real_t getTComp(index_t i1, index_t i2, index_t i3, int i, int j) const {
      if (i > j) { return getTComp(i1, i2, i3, j, i); }
      int idx = i * 4 - i * (i - 1) / 2 + j - i;
      if constexpr (D == Dim::_3D) {
        return T_in(i1, i2, i3, t_comp[idx]);
      } else {
        return T_in(i1, i2, t_comp[idx]);
      }
    }

    Inline void setTComp(index_t i1, index_t i2, int i, int j, real_t val) const {
      if (i > j) { setTComp(i1, i2, j, i, val); return; }
      int idx = i * 4 - i * (i - 1) / 2 + j - i;
      if constexpr (D == Dim::_2D) {
        T_out(i1, i2, out_comp[idx]) = val;
      } else {
        T_out(i1, i2, 0, out_comp[idx]) = val;
      }
    }

    Inline void setTComp(index_t i1, index_t i2, index_t i3, int i, int j, real_t val) const {
      if (i > j) { setTComp(i1, i2, i3, j, i, val); return; }
      int idx = i * 4 - i * (i - 1) / 2 + j - i;
      if constexpr (D == Dim::_3D) {
        T_out(i1, i2, i3, out_comp[idx]) = val;
      } else {
        T_out(i1, i2, out_comp[idx]) = val;
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x { static_cast<real_t>(i1) + HALF, static_cast<real_t>(i2) + HALF };

        // ── 1. Load U (code contravariant) and lower ─────────────────────────
        real_t U[4] { U_in(i1, i2, u_comp[0]), U_in(i1, i2, u_comp[1]),
                      U_in(i1, i2, u_comp[2]), U_in(i1, i2, u_comp[3]) };
        vec_t<Dim::_4D> Uup4 { U[0], U[1], U[2], U[3] };
        vec_t<Dim::_4D> Udn4 { ZERO };
        metric.template transform_4d<Idx::U, Idx::D>(x, Uup4, Udn4);
        real_t Ud[4] { Udn4[0], Udn4[1], Udn4[2], Udn4[3] };

        // ── 2. Metric quantities ──────────────────────────────────────────────
        // sqrt_g  = sqrt(det h_code), sqrt_mg = sqrt(-g_code) = sqrt_g / alpha
        real_t sq_g  = metric.sqrt_det_h(x);
        real_t al    = metric.alpha(x);
        real_t sq_mg = sq_g / al;

        // ── 3. Convert T from physical to code basis ──────────────────────────
        // T deposited with physical spatial indices (u^i_phys); u^0 is code time.
        // T^{μν}_code = J^μ_α J^ν_β T^{αβ}_phys  where J = transform<PU,U> Jacobian.
        // Two-pass: first convert row index (spatial), then column index (spatial).
        // T^{00} and T^{0i}/T^{i0} temporal components handled by including index 0
        // in each 4-vector pass; the transform only touches spatial {1,2,3}.
        real_t T[4][4];
        for (int i = 0; i < 4; ++i)
          for (int j = 0; j < 4; ++j)
            T[i][j] = getTComp(i1, i2, i, j);
        // Pass 1: convert first index (spatial rows i=1,2,3) from phys to code
        for (int j = 0; j < 4; ++j) {
          vec_t<Dim::_3D> col_phys { T[1][j], T[2][j], T[3][j] };
          vec_t<Dim::_3D> col_code { ZERO };
          metric.template transform<Idx::PU, Idx::U>(x, col_phys, col_code);
          T[1][j] = col_code[0]; T[2][j] = col_code[1]; T[3][j] = col_code[2];
        }
        // Pass 2: convert second index (spatial columns j=1,2,3) from phys to code
        for (int i = 0; i < 4; ++i) {
          vec_t<Dim::_3D> row_phys { T[i][1], T[i][2], T[i][3] };
          vec_t<Dim::_3D> row_code { ZERO };
          metric.template transform<Idx::PU, Idx::U>(x, row_phys, row_code);
          T[i][1] = row_code[0]; T[i][2] = row_code[1]; T[i][3] = row_code[2];
        }

        // ── 4. Magnetic 4-vector b^μ (code basis) ────────────────────────────
        // From F_{ij}_code = ε_{ijk} B^k sq_g and F_{0i}_code = E_i_code:
        // b^μ = (1/2)/sq_mg * ε^{μνρσ} Ud_ν F_{ρσ}  (derived analytically)
        real_t B[3] { EM_in(i1, i2, em::bx1), EM_in(i1, i2, em::bx2),
                      EM_in(i1, i2, em::bx3) };
        real_t E[3] { AUX_in(i1, i2, em::ex1), AUX_in(i1, i2, em::ex2),
                      AUX_in(i1, i2, em::ex3) };

        real_t bv[4];
        bv[0] = al * (Ud[1] * B[0] + Ud[2] * B[1] + Ud[3] * B[2]);
        bv[1] = -al * Ud[0] * B[0] + (Ud[2] * E[2] - Ud[3] * E[1]) / sq_mg;
        bv[2] = -al * Ud[0] * B[1] - (Ud[1] * E[2] - Ud[3] * E[0]) / sq_mg;
        bv[3] = -al * Ud[0] * B[2] + (Ud[1] * E[1] - Ud[2] * E[0]) / sq_mg;

        // Lower b, compute norm, normalize
        vec_t<Dim::_4D> bup4 { bv[0], bv[1], bv[2], bv[3] };
        vec_t<Dim::_4D> bdn4 { ZERO };
        metric.template transform_4d<Idx::U, Idx::D>(x, bup4, bdn4);
        real_t b2 = bv[0]*bdn4[0] + bv[1]*bdn4[1] + bv[2]*bdn4[2] + bv[3]*bdn4[3];

        if (cmp::AlmostZero(b2) || b2 <= ZERO) {
          // No meaningful B field: output zeros (tetrad undefined)
          for (int i = 0; i < 4; ++i)
            for (int j = i; j < 4; ++j)
              setTComp(i1, i2, i, j, ZERO);
          return;
        }
        real_t bni = ONE / math::sqrt(b2);
        for (int i = 0; i < 4; ++i) { bv[i] *= bni; bdn4[i] *= bni; }
        real_t bd[4] { bdn4[0], bdn4[1], bdn4[2], bdn4[3] };

        // ── 6. Perpendicular vector p (spatial 3D cross product in code basis) ─
        // p_i_code = ε_{ijk} U^j b^k sq_g  (spatial i,j,k ∈ {1,2,3})
        // Then raise with code metric; p^0 = g^{0i} p_i ≠ 0 after raising.
        real_t pdn[4] { ZERO,
                        (U[2] * bv[3] - U[3] * bv[2]) * sq_g,
                        (U[3] * bv[1] - U[1] * bv[3]) * sq_g,
                        (U[1] * bv[2] - U[2] * bv[1]) * sq_g };
        vec_t<Dim::_4D> pdn4 { pdn[0], pdn[1], pdn[2], pdn[3] };
        vec_t<Dim::_4D> pup4 { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(x, pdn4, pup4);
        real_t pv[4] { pup4[0], pup4[1], pup4[2], pup4[3] };
        // Re-lower for accurate norm
        vec_t<Dim::_4D> pdn4b { ZERO };
        metric.template transform_4d<Idx::U, Idx::D>(x, pup4, pdn4b);
        real_t p2 = pv[0]*pdn4b[0] + pv[1]*pdn4b[1] + pv[2]*pdn4b[2] + pv[3]*pdn4b[3];

        if (cmp::AlmostZero(p2) || p2 <= ZERO) {
          for (int i = 0; i < 4; ++i)
            for (int j = i; j < 4; ++j)
              setTComp(i1, i2, i, j, ZERO);
          return;
        }
        real_t pni = ONE / math::sqrt(p2);
        for (int i = 0; i < 4; ++i) { pv[i] *= pni; pdn4b[i] *= pni; }
        real_t pd[4] { pdn4b[0], pdn4b[1], pdn4b[2], pdn4b[3] };

        // ── 7. s vector: 4D cross product s_μ = ε_{μνρσ} U^ν P^ρ B^σ sq_mg ──
        real_t sdn[4];
        sdn[0] = sq_mg * ( U[1]*(pv[2]*bv[3] - pv[3]*bv[2])
                         - U[2]*(pv[1]*bv[3] - pv[3]*bv[1])
                         + U[3]*(pv[1]*bv[2] - pv[2]*bv[1]) );
        sdn[1] = sq_mg * ( -U[0]*(pv[2]*bv[3] - pv[3]*bv[2])
                          + U[2]*(pv[0]*bv[3] - pv[3]*bv[0])
                          - U[3]*(pv[0]*bv[2] - pv[2]*bv[0]) );
        sdn[2] = sq_mg * (  U[0]*(pv[1]*bv[3] - pv[3]*bv[1])
                          - U[1]*(pv[0]*bv[3] - pv[3]*bv[0])
                          + U[3]*(pv[0]*bv[1] - pv[1]*bv[0]) );
        sdn[3] = sq_mg * ( -U[0]*(pv[1]*bv[2] - pv[2]*bv[1])
                          + U[1]*(pv[0]*bv[2] - pv[2]*bv[0])
                          - U[2]*(pv[0]*bv[1] - pv[1]*bv[0]) );
        vec_t<Dim::_4D> sdn4 { sdn[0], sdn[1], sdn[2], sdn[3] };
        vec_t<Dim::_4D> sup4 { ZERO };
        metric.template transform_4d<Idx::D, Idx::U>(x, sdn4, sup4);
        real_t sv[4] { sup4[0], sup4[1], sup4[2], sup4[3] };
        vec_t<Dim::_4D> sdn4b { ZERO };
        metric.template transform_4d<Idx::U, Idx::D>(x, sup4, sdn4b);
        real_t s2 = sv[0]*sdn4b[0] + sv[1]*sdn4b[1] + sv[2]*sdn4b[2] + sv[3]*sdn4b[3];

        if (cmp::AlmostZero(s2) || s2 <= ZERO) {
          for (int i = 0; i < 4; ++i)
            for (int j = i; j < 4; ++j)
              setTComp(i1, i2, i, j, ZERO);
          return;
        }
        real_t sni = ONE / math::sqrt(s2);
        for (int i = 0; i < 4; ++i) { sv[i] *= sni; sdn4b[i] *= sni; }
        real_t sd[4] { sdn4b[0], sdn4b[1], sdn4b[2], sdn4b[3] };

        // ── 8. Project T directly onto tetrad {u(0), p(1), s(2), b(3)} ───────
        // T^{(a)(b)} = Σ_{μν} T^{μν}_code (e_a)_μ (e_b)_ν
        // e_{(0)}=u gives fluid-frame automatically; spatial vectors p,s,b are ⊥ u
        // Covariant tetrad: a=0: Ud(u), a=1: pd(p), a=2: sd(s), a=3: bd(b)
        // Output 10 upper-triangular components:
        //   (0,0)=uu (W), (0,1)=up, (0,2)=us, (0,3)=ub (q_b),
        //   (1,1)=pp, (1,2)=ps, (1,3)=pb, (2,2)=ss, (2,3)=sb, (3,3)=bb (p_par)
        const real_t* edn[4] = { Ud, pd, sd, bd };
        for (int a = 0; a < 4; ++a)
          for (int b = a; b < 4; ++b) {
            real_t val = ZERO;
            for (int mu = 0; mu < 4; ++mu)
              for (int nu = 0; nu < 4; ++nu)
                val += T[mu][nu] * edn[a][mu] * edn[b][nu];
            setTComp(i1, i2, a, b, val);
          }
      } else {
        raise::KernelError(HERE, "2D implementation of FluidFrameStressEnergy_kernel called for non-2D");
      }
    }
  };

  template <Dimension D, class M, unsigned short N>
    requires metric::traits::HasD<M> && metric::traits::HasTransform<M>
  class Transform4VelocitySpatialToPhysical_kernel {
    // Transforms spatial components of 4-velocity from coordinate to physical basis
    // u^0 (Gamma/alpha) remains unchanged as it's unitless
    ndfield_t<D, N>       Vector;
    const unsigned short  c_u1, c_u2, c_u3;
    const M               metric;

  public:
    Transform4VelocitySpatialToPhysical_kernel(ndfield_t<D, N>&       vector,
                                               unsigned short         cu1,
                                               unsigned short         cu2,
                                               unsigned short         cu3,
                                               const M&               metric)
      : Vector { vector }
      , c_u1 { cu1 }
      , c_u2 { cu2 }
      , c_u3 { cu3 }
      , metric { metric } {
      raise::ErrorIf(c_u1 >= N or c_u2 >= N or c_u3 >= N,
                     "Invalid component index",
                     HERE);
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;

        vec_t<Dim::_3D> u_cntrv { Vector(i1, i2, c_u1),
                                   Vector(i1, i2, c_u2),
                                   Vector(i1, i2, c_u3) };

        vec_t<Dim::_3D> u_phys { ZERO };
        // Transform spatial components from coordinate contravariant to physical contravariant
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_cntrv, u_phys);

        Vector(i1, i2, c_u1) = u_phys[0];
        Vector(i1, i2, c_u2) = u_phys[1];
        Vector(i1, i2, c_u3) = u_phys[2];
      } else {
        raise::KernelError(
          HERE,
          "2D implementation of Transform4VelocitySpatialToPhysical_kernel called for non-2D");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        coord_t<D> x_Code { ZERO };
        x_Code[0] = COORD(i1) + HALF;
        x_Code[1] = COORD(i2) + HALF;
        x_Code[2] = COORD(i3) + HALF;

        vec_t<Dim::_3D> u_cntrv { Vector(i1, i2, i3, c_u1),
                                   Vector(i1, i2, i3, c_u2),
                                   Vector(i1, i2, i3, c_u3) };

        vec_t<Dim::_3D> u_phys { ZERO };
        // Transform spatial components from coordinate contravariant to physical contravariant
        metric.template transform<Idx::U, Idx::PU>(x_Code, u_cntrv, u_phys);

        Vector(i1, i2, i3, c_u1) = u_phys[0];
        Vector(i1, i2, i3, c_u2) = u_phys[1];
        Vector(i1, i2, i3, c_u3) = u_phys[2];
      } else {
        raise::KernelError(
          HERE,
          "3D implementation of Transform4VelocitySpatialToPhysical_kernel called for non-3D");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_PARTICLE_MOMENTS_HPP

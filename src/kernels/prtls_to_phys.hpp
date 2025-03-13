/**
 * @file kernels/prtls_to_phys.hpp
 * @brief Convert particle positions & velocities to physical units
 * @implements
 *   - kernel::PrtlToPhys_kernel<>
 * @namespaces:
 *   - kernel::
 * @note
 * This is a kernel that converts particle positions
 * and velocities to physical units
 * @note SR : to the corresponding tetrad basis
 * @note GR : x -- coordinate basis, u -- covariant basis
 */

#ifndef KERNELS_PRTLS_TO_PHYS_HPP
#define KERNELS_PRTLS_TO_PHYS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <SimEngine::type S, class M>
  class PrtlToPhys_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr Dimension D = M::Dim;

  protected:
    const npart_t            stride;
    array_t<real_t*>         buff_x1;
    array_t<real_t*>         buff_x2;
    array_t<real_t*>         buff_x3;
    array_t<real_t*>         buff_ux1;
    array_t<real_t*>         buff_ux2;
    array_t<real_t*>         buff_ux3;
    array_t<real_t*>         buff_wei;
    const array_t<int*>      i1, i2, i3;
    const array_t<prtldx_t*> dx1, dx2, dx3;
    const array_t<real_t*>   ux1, ux2, ux3;
    const array_t<real_t*>   phi;
    const array_t<real_t*>   weight;
    const M                  metric;

  public:
    PrtlToPhys_kernel(npart_t                   stride,
                      array_t<real_t*>&         buff_x1,
                      array_t<real_t*>&         buff_x2,
                      array_t<real_t*>&         buff_x3,
                      array_t<real_t*>&         buff_ux1,
                      array_t<real_t*>&         buff_ux2,
                      array_t<real_t*>&         buff_ux3,
                      array_t<real_t*>&         buff_wei,
                      const array_t<int*>&      i1,
                      const array_t<int*>&      i2,
                      const array_t<int*>&      i3,
                      const array_t<prtldx_t*>& dx1,
                      const array_t<prtldx_t*>& dx2,
                      const array_t<prtldx_t*>& dx3,
                      const array_t<real_t*>&   ux1,
                      const array_t<real_t*>&   ux2,
                      const array_t<real_t*>&   ux3,
                      const array_t<real_t*>&   phi,
                      const array_t<real_t*>&   weight,
                      const M&                  metric)
      : stride { stride }
      , buff_x1 { buff_x1 }
      , buff_x2 { buff_x2 }
      , buff_x3 { buff_x3 }
      , buff_ux1 { buff_ux1 }
      , buff_ux2 { buff_ux2 }
      , buff_ux3 { buff_ux3 }
      , buff_wei { buff_wei }
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
      , metric { metric } {
      if constexpr ((D == Dim::_1D) || (D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(buff_x1.extent(0) == 0, "Invalid buffer size", HERE);
      }
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(buff_x2.extent(0) == 0, "Invalid buffer size", HERE);
      }
      if constexpr (((D == Dim::_2D) && (M::CoordType != Coord::Cart)) ||
                    (D == Dim::_3D)) {
        raise::ErrorIf(buff_x3.extent(0) == 0, "Invalid buffer size", HERE);
      }
      raise::ErrorIf(buff_ux1.extent(0) == 0, "Invalid buffer size", HERE);
      raise::ErrorIf(buff_ux2.extent(0) == 0, "Invalid buffer size", HERE);
      raise::ErrorIf(buff_ux3.extent(0) == 0, "Invalid buffer size", HERE);
    }

    Inline void operator()(index_t p) const {
      bufferX(p);
      bufferU(p);
      buff_wei(p) = weight(p * stride);
    }

    Inline void bufferX(index_t& p) const {
      if constexpr ((D == Dim::_1D) || (D == Dim::_2D) || (D == Dim::_3D)) {
        buff_x1(p) = metric.template convert<1, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i1(p * stride)) +
          static_cast<real_t>(dx1(p * stride)));
      }
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        buff_x2(p) = metric.template convert<2, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i2(p * stride)) +
          static_cast<real_t>(dx2(p * stride)));
      }
      if constexpr ((D == Dim::_2D) && (M::CoordType != Coord::Cart)) {
        buff_x3(p) = phi(p * stride);
      }
      if constexpr (D == Dim::_3D) {
        buff_x3(p) = metric.template convert<3, Crd::Cd, Crd::Ph>(
          static_cast<real_t>(i3(p * stride)) +
          static_cast<real_t>(dx3(p * stride)));
      }
    }

    Inline void bufferU(index_t& p) const {
      vec_t<Dim::_3D> u_Phys { ZERO };
      if constexpr (D == Dim::_1D) {
        if constexpr (M::CoordType == Coord::Cart) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p * stride)) +
              static_cast<real_t>(dx1(p * stride)) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unsupported coordinate system in 1D");
        }
      } else if constexpr (D == Dim::_2D) {
        if constexpr (M::CoordType == Coord::Cart) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p * stride)) +
                static_cast<real_t>(dx1(p * stride)),
              static_cast<real_t>(i2(p * stride)) +
                static_cast<real_t>(dx2(p * stride)) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p * stride)) +
                static_cast<real_t>(dx1(p * stride)),
              static_cast<real_t>(i2(p * stride)) +
                static_cast<real_t>(dx2(p * stride)),
              phi(p * stride) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else if constexpr (S == SimEngine::GRPIC) {
          metric.template transform<Idx::D, Idx::PD>(
            { static_cast<real_t>(i1(p * stride)) +
                static_cast<real_t>(dx1(p * stride)),
              static_cast<real_t>(i2(p * stride)) +
                static_cast<real_t>(dx2(p * stride)) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unrecognized simulation engine");
        }
      } else if constexpr (D == Dim::_3D) {
        if constexpr (S == SimEngine::SRPIC) {
          metric.template transform_xyz<Idx::XYZ, Idx::T>(
            { static_cast<real_t>(i1(p * stride)) +
                static_cast<real_t>(dx1(p * stride)),
              static_cast<real_t>(i2(p * stride)) +
                static_cast<real_t>(dx2(p * stride)),
              static_cast<real_t>(i3(p * stride)) +
                static_cast<real_t>(dx3(p * stride)) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else if constexpr (S == SimEngine::GRPIC) {
          metric.template transform<Idx::D, Idx::PD>(
            { static_cast<real_t>(i1(p * stride)) +
                static_cast<real_t>(dx1(p * stride)),
              static_cast<real_t>(i2(p * stride)) +
                static_cast<real_t>(dx2(p * stride)),
              static_cast<real_t>(i3(p * stride)) +
                static_cast<real_t>(dx3(p * stride)) },
            { ux1(p * stride), ux2(p * stride), ux3(p * stride) },
            u_Phys);
        } else {
          raise::KernelError(HERE, "Unrecognized simulation engine");
        }
      }
      buff_ux1(p) = u_Phys[0];
      buff_ux2(p) = u_Phys[1];
      buff_ux3(p) = u_Phys[2];
    }
  };
} // namespace kernel

#endif // KERNELS_PRTLS_TO_PHYS_HPP

/**
 * @file particles_bc.cpp
 * @brief Special "reflecting" boundary conditions on the axis ...
 *        ... for particles in 2D axisymmetric simulations.
 * @implements: `ParticlesBoundaryConditions` method of the `PIC` class
 * @includes: --
 * @depends: `pic.h`
 *
 * @notes: - Periodic boundary conditions are implemented in `exchange.cpp`
 *
 */

#include "wrapper.h"

#include "particle_macros.h"
#include "pic.h"

namespace ntt {

#ifdef MINKOWSKI_METRIC
  template <Dimension D>
  void PIC<D>::ParticlesBoundaryConditions() {}
#else
  /**
   * @brief 2d particle bc.
   */
  template <>
  void PIC<Dim2>::ParticlesBoundaryConditions() {
    auto& mblock = this->meshblock;
    for (auto& species : mblock.particles) {
      auto ni1 = mblock.Ni1();
      auto ni2 = mblock.Ni2();
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          // radial boundary conditions
          if ((species.i1(p) < 0) || (species.i1(p) >= ni1)) {
            species.tag(p) = static_cast<short>(ParticleTag::dead);
          }
          // axis boundaries
          if ((species.i2(p) < 0) || (species.i2(p) >= ni2)) {
            if (species.i2(p) < 0) {
              // reflect particle coordinate
              species.i2(p) = 0;
            } else {
              // reflect particle coordinate
              species.i2(p) = ni2 - 1;
            }
            species.dx2(p) = static_cast<prtldx_t>(1.0) - species.dx2(p);
            species.phi(p) = species.phi(p) + constant::PI;
            // reverse u^theta
            coord_t<Dim3> x_p { get_prtl_x1(species, p),
                                get_prtl_x2(species, p),
                                species.phi(p) };
            vec_t<Dim3>   u_hat, u_cart;
            mblock.metric.v3_Cart2Hat(
              x_p, { species.ux1(p), species.ux2(p), species.ux3(p) }, u_hat);
            mblock.metric.v3_Hat2Cart(x_p, { u_hat[0], -u_hat[1], u_hat[2] }, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
        });
    }
  }

  /**
   * @brief 1d particle bc.
   */
  template <>
  void PIC<Dim1>::ParticlesBoundaryConditions() {
    NTTHostError("not applicable");
  }
  /**
   * @brief 3d particle bc.
   */
  template <>
  void PIC<Dim3>::ParticlesBoundaryConditions() {
    NTTHostError("not implemented");
  }

#endif

}    // namespace ntt

template void ntt::PIC<ntt::Dim1>::ParticlesBoundaryConditions();
template void ntt::PIC<ntt::Dim2>::ParticlesBoundaryConditions();
template void ntt::PIC<ntt::Dim3>::ParticlesBoundaryConditions();
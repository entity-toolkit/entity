/**
 * @file particles_bc.cpp
 * @brief Special "reflecting" boundary conditions on the axis ...
 *        ... for particles in 2D axisymmetric simulations.
 * @implements: `ParticlesBoundaryConditions` method of the `PIC` class
 * @includes: --
 * @depends: `pic.h`
 *
 * @notes: - Periodic boundary conditions are implemented in `particles_exch.cpp`
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
    auto& mblock   = this->meshblock;
    auto  params   = *(this->params());
    auto  r_absorb = params.metricParameters()[2];
    auto  r_max    = mblock.metric.x1_max;
    for (auto& species : mblock.particles) {
      auto ni1 = mblock.Ni1();
      auto ni2 = mblock.Ni2();
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          // radial boundary conditions
          if ((species.i1(p) < -1) || (species.i1(p) >= ni1 + 1)) {
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
            species.phi(p) = species.phi(p) + constant::PI;
            species.dx2(p) = 1.0f - species.dx2(p);
            // reverse u^theta
            coord_t<Dim3> x_p { get_prtl_x1(species, p),
                                get_prtl_x2(species, p),
                                species.phi(p) };
            vec_t<Dim3>   u_hat, u_cart;
            mblock.metric.v_Cart2Hat(
              x_p, { species.ux1(p), species.ux2(p), species.ux3(p) }, u_hat);
            mblock.metric.v_Hat2Cart(x_p, { u_hat[0], -u_hat[1], u_hat[2] }, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
          // absorbing boundaries
          coord_t<Dim2> x_sph { ZERO };
          mblock.metric.x_Code2Sph({ get_prtl_x1(species, p), ZERO }, x_sph);
          // particles penetrate 80% of the absorbing region
          if ((x_sph[0] > r_absorb + (real_t)(0.8) * (r_max - r_absorb))) {
            species.tag(p) = static_cast<short>(ParticleTag::dead);
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
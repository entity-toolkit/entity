#include "wrapper.h"
#include "pic.h"

#include "particle_macros.h"

namespace ntt {
  /**
   * @brief 1d particle bc.
   *
   */
  template <>
  void PIC<Dim1>::ParticlesBoundaryConditions() {}

  /**
   * @brief 2d particle bc.
   *
   */
  template <>
  void PIC<Dim2>::ParticlesBoundaryConditions() {
#ifndef MINKOWSKI_METRIC
    auto& mblock = this->meshblock;
    for (auto& species : mblock.particles) {
      auto ni = mblock.Ni1();
      auto nj = mblock.Ni2();
      Kokkos::parallel_for(
        "prtl_bc", species.rangeActiveParticles(), Lambda(index_t p) {
          // radial boundary conditions
          species.is_dead(p) = ((species.i1(p) < -1) || (species.i1(p) >= ni + 1));
          if (species.i2(p) < 0) {
            // reflect particle coordinate
            species.i2(p)  = 0;
            species.dx2(p) = 1.0f - species.dx2(p);
            // reverse u^theta
            coord_t<Dim3> x_p {
              get_prtl_x1(species, p), get_prtl_x2(species, p), species.phi(p)};
            vec_t<Dim3> u_hat, u_cart;
            mblock.metric.v_Cart2Hat(
              x_p, {species.ux1(p), species.ux2(p), species.ux3(p)}, u_hat);
            mblock.metric.v_Hat2Cart(x_p, {u_hat[0], -u_hat[1], u_hat[2]}, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
          if (species.i2(p) >= nj) {
            // reflect particle coordinate
            species.i2(p)  = nj - 1;
            species.dx2(p) = 1.0f - species.dx2(p);
            // reverse u^theta
            coord_t<Dim3> x_p {
              get_prtl_x1(species, p), get_prtl_x2(species, p), species.phi(p)};
            vec_t<Dim3> u_hat, u_cart;
            mblock.metric.v_Cart2Hat(
              x_p, {species.ux1(p), species.ux2(p), species.ux3(p)}, u_hat);
            mblock.metric.v_Hat2Cart(x_p, {u_hat[0], -u_hat[1], u_hat[2]}, u_cart);
            species.ux1(p) = u_cart[0];
            species.ux2(p) = u_cart[1];
            species.ux3(p) = u_cart[2];
          }
        });
    }
#endif
  }

  /**
   * @brief 3d particle bc.
   */
  template <>
  void PIC<Dim3>::ParticlesBoundaryConditions() {}

} // namespace ntt

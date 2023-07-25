#include "wrapper.h"

#include "particle_macros.h"

#include "meshblock/mesh.h"
#include "meshblock/particles.h"

namespace ntt {
  template <>
  auto Particles<Dim1, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim1>&) -> void {}

  template <>
  auto Particles<Dim2, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim2>&) -> void {}

  template <>
  auto Particles<Dim3, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim3>&) -> void {}

#ifndef MPI_ENABLED
  template <>
  auto Particles<Dim1, PICEngine>::BoundaryConditions(const Mesh<Dim1>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC),
                       "1D SR only supports periodic boundaries");
      }
    }
    const auto ni1 = mesh.Ni1();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        i1(p) += ni1 * static_cast<int>(i1(p) < 0) - ni1 * static_cast<int>(i1(p) >= (int)ni1);
      });
  }
#  ifdef MINKOWSKI_METRIC
  template <>
  auto Particles<Dim2, PICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC),
                       "2D Minkowski SR only supports periodic boundaries");
      }
    }
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2();
    Kokkos::parallel_for(
      "Exchange_particles", rangeActiveParticles(), ClassLambda(index_t p) {
        i1(p) += ni1 * static_cast<int>(i1(p) < 0) - ni1 * static_cast<int>(i1(p) >= (int)ni1);
        i2(p) += ni2 * static_cast<int>(i2(p) < 0) - ni2 * static_cast<int>(i2(p) >= (int)ni2);
      });
  }
  template <>
  auto Particles<Dim3, PICEngine>::BoundaryConditions(const Mesh<Dim3>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC),
                       "3D Minkowski only supports periodic boundaries");
      }
    }
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2(), ni3 = mesh.Ni3();
    Kokkos::parallel_for(
      "Exchange_particles", rangeActiveParticles(), ClassLambda(index_t p) {
        i1(p) += ni1 * static_cast<int>(i1(p) < 0) - ni1 * static_cast<int>(i1(p) >= (int)ni1);
        i2(p) += ni2 * static_cast<int>(i2(p) < 0) - ni2 * static_cast<int>(i2(p) >= (int)ni2);
        i3(p) += ni3 * static_cast<int>(i3(p) < 0) - ni3 * static_cast<int>(i3(p) >= (int)ni3);
      });
  }
#  else     // not MINKOWSKI_METRIC
  template <>
  auto Particles<Dim2, PICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    NTTHostErrorIf((mesh.boundaries[0][0] != BoundaryCondition::OPEN)
                     && (mesh.boundaries[0][0] != BoundaryCondition::CUSTOM),
                   "2D non-Minkowski SR must have open or custom boundaries in x1_min");
    NTTHostErrorIf(
      (mesh.boundaries[0][1] != BoundaryCondition::OPEN)
        && (mesh.boundaries[0][1] != BoundaryCondition::CUSTOM)
        && (mesh.boundaries[0][1] != BoundaryCondition::ABSORB),
      "2D non-Minkowski SR must have open or custom or absorb boundaries in x1_max");
    NTTHostErrorIf(mesh.boundaries[1][0] != BoundaryCondition::AXIS,
                   "2D non-Minkowski SR must have axis boundaries in x2");
    const auto ni1 = static_cast<int>(mesh.Ni1());
    const auto ni2 = static_cast<int>(mesh.Ni2());
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        // radial boundary conditions
        if ((i1(p) < 0) || (i1(p) >= ni1)) {
          tag(p) = static_cast<short>(ParticleTag::dead);
        }
        // axis boundaries
        if ((i2(p) < 0) || (i2(p) >= ni2)) {
          i2(p)  = math::min(math::max(i2(p), 0), (int)ni2 - 1);
          dx2(p) = static_cast<prtldx_t>(1.0) - dx2(p);
          phi(p) = phi(p) + constant::PI;
          // reverse u^theta
          coord_t<Dim3> x_p { get_prtl_x1(*this, p), get_prtl_x2(*this, p), phi(p) };
          vec_t<Dim3>   u_hat, u_cart;
          mesh.metric.v3_Cart2Hat(x_p, { ux1(p), ux2(p), ux3(p) }, u_hat);
          // reverse u^theta
          mesh.metric.v3_Hat2Cart(x_p, { u_hat[0], -u_hat[1], u_hat[2] }, u_cart);
          ux1(p) = u_cart[0];
          ux2(p) = u_cart[1];
          ux3(p) = u_cart[2];
        }
      });
  }
  template <>
  auto Particles<Dim3, PICEngine>::BoundaryConditions(const Mesh<Dim3>&) -> void {
    NTTHostError("not implemented");
  }
#  endif    // MINKOWSKI_METRIC

  template <>
  auto Particles<Dim2, GRPICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    NTTHostErrorIf((mesh.boundaries[0][0] != BoundaryCondition::OPEN)
                     && (mesh.boundaries[0][0] != BoundaryCondition::CUSTOM),
                   "2D GR must have open or custom boundaries in x1_min");
    NTTHostErrorIf((mesh.boundaries[0][1] != BoundaryCondition::OPEN)
                     && (mesh.boundaries[0][1] != BoundaryCondition::CUSTOM)
                     && (mesh.boundaries[0][1] != BoundaryCondition::ABSORB),
                   "2D GR must have open or custom or absorb boundaries in x1_max");
    NTTHostErrorIf(mesh.boundaries[1][0] != BoundaryCondition::AXIS,
                   "2D GR must have axis boundaries in x2");

    const auto    rh = mesh.metric.getParameter("rh");
    coord_t<Dim2> xh_CU { ZERO };
    mesh.metric.x_Sph2Code({ rh, ZERO }, xh_CU);
    auto       i1h      = static_cast<int>(xh_CU[0]);
    // !TODO: make this more rigorous
    const auto buffer_h = 5;
    i1h -= buffer_h;
    if (i1h < 0) {
      NTTWarn("not enough buffer at rmin below the horizon, setting i1 of horizon to 0");
    }
    i1h            = std::max(i1h, 0);

    const auto ni1 = static_cast<int>(mesh.Ni1());
    const auto ni2 = static_cast<int>(mesh.Ni2());
    Kokkos::parallel_for(
      "prtl_bc", rangeActiveParticles(), ClassLambda(index_t p) {
        // radial boundary conditions
        if ((i1(p) < i1h) || (i1(p) >= ni1)) {
          tag(p) = static_cast<short>(ParticleTag::dead);
        }
        // axis boundaries
        if ((i2(p) < 0) || (i2(p) >= ni2)) {
          if (i2(p) < 0) {
            // reflect particle coordinate
            i2(p) = 0;
          } else {
            // reflect particle coordinate
            i2(p) = ni2 - 1;
          }
          dx2(p) = static_cast<prtldx_t>(1.0) - dx2(p);
          phi(p) = phi(p) + constant::PI;
          // reverse u^theta
          ux2(p) = -ux2(p);
        }
      });
  }

  template <>
  auto Particles<Dim3, GRPICEngine>::BoundaryConditions(const Mesh<Dim3>&) -> void {
    NTTHostError("not implemented");
  }

#else    // MPI_ENABLED
  template <>
  auto Particles<Dim1, PICEngine>::BoundaryConditions(const Mesh<Dim1>& mesh) -> void {
    const auto ni1 { mesh.Ni1() };
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        const auto move_i1m = (i1(p) < 0);
        const auto move_i1p = (i1(p) >= ni1);
        if (move_i1m) {
          tag(p) = static_cast<short>(ParticleTag::dead);
        }
      });
  }
#endif
}    // namespace ntt
#ifdef MPI_ENABLED

#  include "wrapper.h"

#  include "particle_macros.h"

#  include "meshblock/mesh.h"
#  include "meshblock/particles.h"

namespace ntt {

  Inline auto SendTag(short tag, bool im1, bool ip1) -> short {
    return ((im1) * (PrtlSendTag<Dim1>::im1 - 1) + (ip1) * (PrtlSendTag<Dim1>::ip1 - 1) + 1)
           * tag;
  }

  Inline auto SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1) -> short {
    return ((im1 && jm1) * (PrtlSendTag<Dim2>::im1_jm1 - 1)
            + (im1 && jp1) * (PrtlSendTag<Dim2>::im1_jp1 - 1)
            + (ip1 && jm1) * (PrtlSendTag<Dim2>::ip1_jm1 - 1)
            + (ip1 && jp1) * (PrtlSendTag<Dim2>::ip1_jp1 - 1)
            + (im1 && !jp1 && !jm1) * (PrtlSendTag<Dim2>::im1__j0 - 1)
            + (ip1 && !jp1 && !jm1) * (PrtlSendTag<Dim2>::ip1__j0 - 1)
            + (jm1 && !ip1 && !im1) * (PrtlSendTag<Dim2>::i0__jm1 - 1)
            + (jp1 && !ip1 && !im1) * (PrtlSendTag<Dim2>::i0__jp1 - 1) + 1)
           * tag;
  }

  Inline auto SendTag(short tag, bool im1, bool ip1, bool jm1, bool jp1, bool km1, bool kp1)
    -> short {
    return ((im1 && jm1 && km1) * (PrtlSendTag<Dim3>::im1_jm1_km1 - 1)
            + (im1 && jm1 && kp1) * (PrtlSendTag<Dim3>::im1_jm1_kp1 - 1)
            + (im1 && jp1 && km1) * (PrtlSendTag<Dim3>::im1_jp1_km1 - 1)
            + (im1 && jp1 && kp1) * (PrtlSendTag<Dim3>::im1_jp1_kp1 - 1)
            + (ip1 && jm1 && km1) * (PrtlSendTag<Dim3>::ip1_jm1_km1 - 1)
            + (ip1 && jm1 && kp1) * (PrtlSendTag<Dim3>::ip1_jm1_kp1 - 1)
            + (ip1 && jp1 && km1) * (PrtlSendTag<Dim3>::ip1_jp1_km1 - 1)
            + (ip1 && jp1 && kp1) * (PrtlSendTag<Dim3>::ip1_jp1_kp1 - 1)
            + (im1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::im1_jm1__k0 - 1)
            + (im1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::im1_jp1__k0 - 1)
            + (ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::ip1_jm1__k0 - 1)
            + (ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::ip1_jp1__k0 - 1)
            + (im1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim3>::im1__j0_km1 - 1)
            + (im1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim3>::im1__j0_kp1 - 1)
            + (ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim3>::ip1__j0_km1 - 1)
            + (ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim3>::ip1__j0_kp1 - 1)
            + (!im1 && !ip1 && jm1 && km1) * (PrtlSendTag<Dim3>::i0__jm1_km1 - 1)
            + (!im1 && !ip1 && jm1 && kp1) * (PrtlSendTag<Dim3>::i0__jm1_kp1 - 1)
            + (!im1 && !ip1 && jp1 && km1) * (PrtlSendTag<Dim3>::i0__jp1_km1 - 1)
            + (!im1 && !ip1 && jp1 && kp1) * (PrtlSendTag<Dim3>::i0__jp1_kp1 - 1)
            + (!im1 && !ip1 && !jm1 && !jp1 && km1) * (PrtlSendTag<Dim3>::i0___j0_km1 - 1)
            + (!im1 && !ip1 && !jm1 && !jp1 && kp1) * (PrtlSendTag<Dim3>::i0___j0_kp1 - 1)
            + (!im1 && !ip1 && jm1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::i0__jm1__k0 - 1)
            + (!im1 && !ip1 && jp1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::i0__jp1__k0 - 1)
            + (im1 && !jm1 && !jp1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::im1__j0__k0 - 1)
            + (ip1 && !jm1 && !jp1 && !km1 && !kp1) * (PrtlSendTag<Dim3>::ip1__j0__k0 - 1) + 1)
           * tag;
  }

  template <>
  auto Particles<Dim1, PICEngine>::BoundaryConditions(const Mesh<Dim1>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC) && (bc != BoundaryCondition::COMM),
                       "1D SR only supports periodic or comm boundaries");
      }
    }
    const auto ni1 = mesh.Ni1();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        tag(p) = SendTag(tag(p), i1(p) < 0, i1(p) >= ni1);
      });
  }

#  ifdef MINKOWSKI_METRIC
  template <>
  auto Particles<Dim2, PICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC) && (bc != BoundaryCondition::COMM),
                       "2D Minkowski SR only supports periodic or comm boundaries");
      }
    }
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        tag(p) = SendTag(tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2);
      });
  }
#  else     // not MINKOWSKI_METRIC
  template <>
  auto Particles<Dim2, PICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    const auto rm_im1 = ((mesh.boundaries[0][0] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[0][0] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[0][0] == BoundaryCondition::CUSTOM));
    const auto rm_ip1 = ((mesh.boundaries[0][1] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[0][1] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[0][1] == BoundaryCondition::CUSTOM));
    const auto rm_jm1 = ((mesh.boundaries[1][0] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[1][0] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[1][0] == BoundaryCondition::CUSTOM));
    const auto rm_jp1 = ((mesh.boundaries[1][1] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[1][1] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[1][1] == BoundaryCondition::CUSTOM));
    const auto ax_jm1 = (mesh.boundaries[1][0] == BoundaryCondition::AXIS);
    const auto ax_jp1 = (mesh.boundaries[1][1] == BoundaryCondition::AXIS);
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        if (tag(p) != ParticleTag::alive) {
          return;
        }
        // remove at r_min or r_max
        if (((i1(p) < 0) && rm_im1) || ((i1(p) >= ni1) && rm_ip1)) {
          tag(p) = ParticleTag::dead;
          return;
        }
        if ((ax_jm1 && (i2(p) < 1)) || (ax_jp1 && (i2(p) >= ni2 - 1))) {
          // reflect: u_x -> -u_x
          ux1(p) = -ux1(p);
          return;
        }
        if (((i2(p) < 0) && rm_jm1) || ((i2(p) >= ni2) && rm_jp1)) {
          // remove at theta_min or theta_max (just in case; should not be called)
          tag(p) = ParticleTag::dead;
          return;
        }
        // otherwise, communicate the particle
        tag(p) = SendTag(tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2);
      });
  }
#  endif    // MINKOWSKI_METRIC

  template <>
  auto Particles<Dim3, PICEngine>::BoundaryConditions(const Mesh<Dim3>& mesh) -> void {
    for (auto& bcs : mesh.boundaries) {
      for (auto& bc : bcs) {
        NTTHostErrorIf((bc != BoundaryCondition::PERIODIC) && (bc != BoundaryCondition::COMM),
                       "3D SR only supports periodic or comm boundaries");
      }
    }
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2(), ni3 = mesh.Ni3();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        tag(p) = SendTag(
          tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2, i3(p) < 0, i3(p) >= ni3);
      });
  }

  template <>
  auto Particles<Dim2, GRPICEngine>::BoundaryConditions(const Mesh<Dim2>& mesh) -> void {
    const auto rm_im1 = ((mesh.boundaries[0][0] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[0][0] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[0][0] == BoundaryCondition::CUSTOM));
    const auto rm_ip1 = ((mesh.boundaries[0][1] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[0][1] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[0][1] == BoundaryCondition::CUSTOM));
    const auto rm_jm1 = ((mesh.boundaries[1][0] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[1][0] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[1][0] == BoundaryCondition::CUSTOM));
    const auto rm_jp1 = ((mesh.boundaries[1][1] == BoundaryCondition::OPEN)
                         || (mesh.boundaries[1][1] == BoundaryCondition::ABSORB)
                         || (mesh.boundaries[1][1] == BoundaryCondition::CUSTOM));
    const auto ax_jm1 = (mesh.boundaries[1][0] == BoundaryCondition::AXIS);
    const auto ax_jp1 = (mesh.boundaries[1][1] == BoundaryCondition::AXIS);
    const auto ni1 = mesh.Ni1(), ni2 = mesh.Ni2();
    Kokkos::parallel_for(
      "BoundaryConditions", rangeActiveParticles(), ClassLambda(index_t p) {
        if (tag(p) != ParticleTag::alive) {
          return;
        }
        // remove at r_min or r_max
        if (((i1(p) < 0) && rm_im1) || ((i1(p) >= ni1) && rm_ip1)) {
          tag(p) = ParticleTag::dead;
          return;
        }
        if ((ax_jm1 && (i2(p) < 1)) || (ax_jp1 && (i2(p) >= ni2 - 1))) {
          // reflect: u_theta -> -u_theta
          ux2(p) = -ux2(p);
          return;
        }
        if (((i2(p) < 0) && rm_jm1) || ((i2(p) >= ni2) && rm_jp1)) {
          // remove at theta_min or theta_max (just in case; should not be called)
          tag(p) = ParticleTag::dead;
          return;
        }
        // otherwise, communicate the particle
        tag(p) = SendTag(tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2);
      });
  }

  template <>
  auto Particles<Dim3, GRPICEngine>::BoundaryConditions(const Mesh<Dim3>&) -> void {
    NTTHostError("not implemented");
  }

  template <>
  auto Particles<Dim1, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim1>&) -> void {
    NTTHostError("not implemented");
  }

  template <>
  auto Particles<Dim2, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim2>&) -> void {
    NTTHostError("not implemented");
  }

  template <>
  auto Particles<Dim3, SANDBOXEngine>::BoundaryConditions(const Mesh<Dim3>&) -> void {
    NTTHostError("not implemented");
  }

}    // namespace ntt

#endif    // MPI_ENABLED
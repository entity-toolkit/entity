#ifdef MPI_ENABLED

#  include "wrapper.h"

#  include "particle_macros.h"

#  include "meshblock/mesh.h"
#  include "meshblock/particles.h"

namespace ntt {
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
        if (tag(p) == ParticleTag::alive) {
          tag(p) = (short)(i1(p) < 0) * (PrtlSendTag<Dim1>::im1 - 1)
                   + (short)(i1(p) >= ni1) * (PrtlSendTag<Dim1>::ip1 - 1) + 1;
        }
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
        if (tag(p) == ParticleTag::alive) {
          if (i1(p) < 0) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::im1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::im1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::im1__j0;
            }
          } else if (i1(p) >= ni1) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::ip1__j0;
            }
          } else {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::i0__jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::i0__jp1;
            }
          }
        }
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
        // check if needs removing
        if ((tag(p) == ParticleTag::alive) && (i1(p) < 0) && rm_im1) {
          tag(p) = ParticleTag::dead;
          return;
        }
        if ((tag(p) == ParticleTag::alive) && (i1(p) >= ni1) && rm_ip1) {
          tag(p) = ParticleTag::dead;
          return;
        }
        auto reflect = false;
        if ((tag(p) == ParticleTag::alive) && (i2(p) < 0)) {
          if (rm_jm1) {
            tag(p) = ParticleTag::dead;
            return;
          } else if (ax_jm1) {
            i2(p)   = 0;
            reflect = true;
          }
        }
        if ((tag(p) == ParticleTag::alive) && (i2(p) >= ni2) && rm_jp1) {
          if (rm_jp1) {
            tag(p) = ParticleTag::dead;
            return;
          } else if (ax_jp1) {
            i2(p)   = (int)ni2 - 1;
            reflect = true;
          }
        }
        if (reflect) {
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
        if (tag(p) == ParticleTag::alive) {
          if (i1(p) < 0) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::im1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::im1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::im1__j0;
            }
          } else if (i1(p) >= ni1) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::ip1__j0;
            }
          } else {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::i0__jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::i0__jp1;
            }
          }
        }
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
        if (tag(p) == ParticleTag::alive) {
          if (i1(p) < 0) {
            if (i2(p) < 0) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::im1_jm1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::im1_jm1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::im1_jm1__k0;
              }
            } else if (i2(p) >= ni2) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::im1_jp1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::im1_jp1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::im1_jp1__k0;
              }
            } else {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::im1__j0_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::im1__j0_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::im1__j0__k0;
              }
            }
          } else if (i1(p) >= ni1) {
            if (i2(p) < 0) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::ip1_jm1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::ip1_jm1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::ip1_jm1__k0;
              }
            } else if (i2(p) >= ni2) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::ip1_jp1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::ip1_jp1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::ip1_jp1__k0;
              }
            } else {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::ip1__j0_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::ip1__j0_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::ip1__j0__k0;
              }
            }
          } else {
            if (i2(p) < 0) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::i0__jm1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::i0__jm1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::i0__jm1__k0;
              }
            } else if (i2(p) >= ni2) {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::i0__jp1_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::i0__jp1_kp1;
              } else {
                tag(p) = PrtlSendTag<Dim3>::i0__jp1__k0;
              }
            } else {
              if (i3(p) < 0) {
                tag(p) = PrtlSendTag<Dim3>::i0___j0_km1;
              } else if (i3(p) >= ni3) {
                tag(p) = PrtlSendTag<Dim3>::i0___j0_kp1;
              }
            }
          }
        }
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
        // check if needs removing
        if ((tag(p) == ParticleTag::alive) && (i1(p) < 0) && rm_im1) {
          tag(p) = ParticleTag::dead;
          return;
        }
        if ((tag(p) == ParticleTag::alive) && (i1(p) >= ni1) && rm_ip1) {
          tag(p) = ParticleTag::dead;
          return;
        }
        auto reflect = false;
        if ((tag(p) == ParticleTag::alive) && (i2(p) < 0)) {
          if (rm_jm1) {
            tag(p) = ParticleTag::dead;
            return;
          } else if (ax_jm1) {
            i2(p)   = 0;
            reflect = true;
          }
        }
        if ((tag(p) == ParticleTag::alive) && (i2(p) >= ni2) && rm_jp1) {
          if (rm_jp1) {
            tag(p) = ParticleTag::dead;
            return;
          } else if (ax_jp1) {
            i2(p)   = (int)ni2 - 1;
            reflect = true;
          }
        }
        if (reflect) {
          dx2(p) = static_cast<prtldx_t>(1.0) - dx2(p);
          phi(p) = phi(p) + constant::PI;
          // reverse u^theta
          ux2(p) = -ux2(p);
        }
        if (tag(p) == ParticleTag::alive) {
          if (i1(p) < 0) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::im1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::im1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::im1__j0;
            }
          } else if (i1(p) >= ni1) {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::ip1_jp1;
            } else {
              tag(p) = PrtlSendTag<Dim2>::ip1__j0;
            }
          } else {
            if (i2(p) < 0) {
              tag(p) = PrtlSendTag<Dim2>::i0__jm1;
            } else if (i2(p) >= ni2) {
              tag(p) = PrtlSendTag<Dim2>::i0__jp1;
            }
          }
        }
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
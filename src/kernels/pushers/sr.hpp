/**
 * @file kernels/pushers/sr.hpp
 * @brief Particle pusher for the SR
 * @implements
 *   - kernel::sr::Pusher_kernel<>
 * @namespaces:
 *   - kernel::sr::
 * @macros:
 *   - MPI_ENABLED
 * @note
 * At the end of the boundary condition call, if MPI is enabled particles
 * are additionally tagged depending on which direction they are leaving
 */

#ifndef KERNELS_PUSHERS_SR_HPP
#define KERNELS_PUSHERS_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/archetypes.h"
#include "traits/metric.h"
#include "traits/policies.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h"
#include "kernels/pushers/sr_policies.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    I = static_cast<int>((XI + 1)) - 1;                                        \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

/* -------------------------------------------------------------------------- */

namespace kernel::sr {
  using namespace ntt;

  /**
   * @tparam M Metric
   * @tparam P Extra policies
   */
  template <SRMetricClass M, class P = PusherPolicy<M>>
  struct Pusher_kernel {
    using E                   = typename P::EmissionPolicy;
    using PUPD                = typename P::CustomParticleUpdatePolicy;
    using F                   = typename P::ExternalFieldsPolicy;
    static constexpr auto Atm = P::ApplyAtmosphere;

    static constexpr auto D            = M::Dim;
    static constexpr auto HasExtFx1    = ::traits::fieldsetter::HasFx1<F, D>;
    static constexpr auto HasExtFx2    = ::traits::fieldsetter::HasFx2<F, D>;
    static constexpr auto HasExtFx3    = ::traits::fieldsetter::HasFx3<F, D>;
    static constexpr auto HasExtForce  = HasExtFx1 or HasExtFx2 or HasExtFx3;
    static constexpr auto HasExtEx1    = ::traits::fieldsetter::HasEx1<F, D>;
    static constexpr auto HasExtEx2    = ::traits::fieldsetter::HasEx2<F, D>;
    static constexpr auto HasExtEx3    = ::traits::fieldsetter::HasEx3<F, D>;
    static constexpr auto HasExtEfield = HasExtEx1 or HasExtEx2 or HasExtEx3;
    static constexpr auto HasExtBx1    = ::traits::fieldsetter::HasBx1<F, D>;
    static constexpr auto HasExtBx2    = ::traits::fieldsetter::HasBx2<F, D>;
    static constexpr auto HasExtBx3    = ::traits::fieldsetter::HasBx3<F, D>;
    static constexpr auto HasExtBfield = HasExtBx1 or HasExtBx2 or HasExtBx3;
    static constexpr auto HasEmission  = not ::traits::emission::IsNoPolicy<E>;

    static constexpr auto HasCustomPrtlUpdate =
      not ::traits::custom_prtl_update::IsNoPolicy<PUPD>;

    const PusherContext       ctx;
    const PusherBoundaries<D> bc;
    PusherArrays              particles;

    const randacc_ndfield_t<D, 6> EB;

    const M metric;
    const P policies;

    const real_t normalized_dt_half;

    Pusher_kernel(const PusherContext&           pusher_ctx,
                  const PusherBoundaries<D>&     pusher_boundaries,
                  PusherArrays&                  pusher_arrays,
                  const randacc_ndfield_t<D, 6>& EB,
                  const M&                       metric,
                  const P&                       policies = {})
      : ctx { pusher_ctx }
      , bc { pusher_boundaries }
      , particles { pusher_arrays }
      , EB { EB }
      , metric { metric }
      , policies { policies }
      , normalized_dt_half { HALF * (ctx.charge / ctx.mass) * ctx.omegaB0 * ctx.dt } {
      raise::ErrorIf(ctx.pusher_flags == ParticlePusher::NONE,
                     "No particle pusher specified",
                     HERE);
    }

    Inline void operator()(index_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        if (particles.tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      getParticlePosition(p, xp_Cd);
      if (ctx.pusher_flags == ParticlePusher::PHOTON) {
        /**
         * Procedure for massless particles
         */
        if constexpr (HasEmission) {
          // get Cartesian position
          coord_t<M::PrtlDim> xp_Ph { ZERO };
          if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                        M::PrtlDim == Dim::_3D) {
            xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
          }
          if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
            xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
          }
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
          }

          // get Cartesian velocity
          vec_t<Dim::_3D> u_prime { ZERO };
          u_prime[0] = particles.ux1(p);
          u_prime[1] = particles.ux2(p);
          u_prime[2] = particles.ux3(p);

          // get Cartesian fields
          vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
          vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
          getInterpolatedEMFields<SHAPE_ORDER>(p, ei, bi);

          metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
          metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);

          processEmission(p, u_prime, xp_Cd, xp_Ph, ei_Cart, bi_Cart);
        }
        // finally update the position
        positionPush(false, p, xp_Cd);
      } else {
        /**
         * Procedure for massive particles
         */
        // update cartesian velocity
        vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
        vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
        vec_t<Dim::_3D> external_force_Cart { ZERO };
        vec_t<Dim::_3D> u_prime { ZERO };
        vec_t<Dim::_3D> ei_Cart_rad { ZERO }, bi_Cart_rad { ZERO };
        bool            is_gca { false };

        // field interpolation 0th-11th order
        getInterpolatedEMFields<SHAPE_ORDER>(p, ei, bi);

        metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
        metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);

        coord_t<M::PrtlDim> xp_Ph { ZERO };

        if constexpr (HasExtForce or Atm or HasExtEfield or HasExtBfield or
                      HasEmission) {
          if constexpr (M::PrtlDim == Dim::_1D or M::PrtlDim == Dim::_2D or
                        M::PrtlDim == Dim::_3D) {
            xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
          }
          if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
            xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
          }
          if constexpr (M::PrtlDim == Dim::_3D) {
            xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
          }
        }

        // add the user-provided external fields
        if constexpr (HasExtEfield) {
          vec_t<Dim::_3D> ext_e_Ph { ZERO };
          vec_t<Dim::_3D> ext_e_Cart { ZERO };
          if constexpr (HasExtEx1) {
            ext_e_Ph[0] = policies.external_fields_policy.ex1(xp_Ph);
          }
          if constexpr (HasExtEx2) {
            ext_e_Ph[1] = policies.external_fields_policy.ex2(xp_Ph);
          }
          if constexpr (HasExtEx3) {
            ext_e_Ph[2] = policies.external_fields_policy.ex3(xp_Ph);
          }
          metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd, ext_e_Ph, ext_e_Cart);
          ei_Cart[0] += ext_e_Cart[0];
          ei_Cart[1] += ext_e_Cart[1];
          ei_Cart[2] += ext_e_Cart[2];
        }

        if constexpr (HasExtBfield) {
          vec_t<Dim::_3D> ext_b_Ph { ZERO };
          vec_t<Dim::_3D> ext_b_Cart { ZERO };
          if constexpr (HasExtBx1) {
            ext_b_Ph[0] = policies.external_fields_policy.bx1(xp_Ph);
          }
          if constexpr (HasExtBx2) {
            ext_b_Ph[1] = policies.external_fields_policy.bx2(xp_Ph);
          }
          if constexpr (HasExtBx3) {
            ext_b_Ph[2] = policies.external_fields_policy.bx3(xp_Ph);
          }
          metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd, ext_b_Ph, ext_b_Cart);
          bi_Cart[0] += ext_b_Cart[0];
          bi_Cart[1] += ext_b_Cart[1];
          bi_Cart[2] += ext_b_Cart[2];
        }

        // backup fields & velocities to use later in radiative drag
        if ((ctx.radiative_drag_flags != RadiativeDrag::NONE) or HasEmission) {
          ei_Cart_rad[0] = ei_Cart[0];
          ei_Cart_rad[1] = ei_Cart[1];
          ei_Cart_rad[2] = ei_Cart[2];
          bi_Cart_rad[0] = bi_Cart[0];
          bi_Cart_rad[1] = bi_Cart[1];
          bi_Cart_rad[2] = bi_Cart[2];
          u_prime[0]     = particles.ux1(p);
          u_prime[1]     = particles.ux2(p);
          u_prime[2]     = particles.ux3(p);
        }

        // compute the external force either user-provided or from the atmosphere model
        if constexpr (HasExtForce or Atm) {
          getExternalForce(xp_Cd, xp_Ph, external_force_Cart);
        }

        if (ctx.pusher_flags & ParticlePusher::GCA) {
          /* hybrid GCA/conventional mode --------------------------------- */
          const auto E2 { NORM_SQR(ei_Cart[0], ei_Cart[1], ei_Cart[2]) };
          const auto B2 { NORM_SQR(bi_Cart[0], bi_Cart[1], bi_Cart[2]) };
          const auto rL {
            math::sqrt(
              ONE + NORM_SQR(particles.ux1(p), particles.ux2(p), particles.ux3(p))) *
            ctx.dt / (TWO * math::abs(normalized_dt_half) * math::sqrt(B2))
          };
          if (B2 > ZERO && rL < ctx.gca.larmor_max &&
              (E2 / B2) < ctx.gca.e_ovr_b_sqr_max) {
            is_gca = true;
            // update with GCA
            if constexpr (HasExtForce or Atm) {
              velocityEMPush_GCA_ExtForce(p, external_force_Cart, ei_Cart, bi_Cart);
            } else {
              velocityEMPush_GCA(p, ei_Cart, bi_Cart);
            }
          } else {
            // update with conventional pusher
            if constexpr (HasExtForce or Atm) {
              particles.ux1(p) += HALF * ctx.dt * external_force_Cart[0];
              particles.ux2(p) += HALF * ctx.dt * external_force_Cart[1];
              particles.ux3(p) += HALF * ctx.dt * external_force_Cart[2];
            }
            if (ctx.pusher_flags & ParticlePusher::BORIS) {
              velocityEMPush_Boris(p, ei_Cart, bi_Cart);
            } else if (ctx.pusher_flags & ParticlePusher::VAY) {
              velocityEMPush_Vay(p, ei_Cart, bi_Cart);
            } else {
              raise::KernelError(HERE, "Invalid pusher algorithm for GCA mode");
            }
            if constexpr (HasExtForce or Atm) {
              particles.ux1(p) += HALF * ctx.dt * external_force_Cart[0];
              particles.ux2(p) += HALF * ctx.dt * external_force_Cart[1];
              particles.ux3(p) += HALF * ctx.dt * external_force_Cart[2];
            }
          }
        } else {
          /* conventional pusher mode ------------------------------------- */
          // update with conventional pusher
          if constexpr (HasExtForce or Atm) {
            particles.ux1(p) += HALF * ctx.dt * external_force_Cart[0];
            particles.ux2(p) += HALF * ctx.dt * external_force_Cart[1];
            particles.ux3(p) += HALF * ctx.dt * external_force_Cart[2];
          }
          if (ctx.pusher_flags & ParticlePusher::BORIS) {
            velocityEMPush_Boris(p, ei_Cart, bi_Cart);
          } else if (ctx.pusher_flags & ParticlePusher::VAY) {
            velocityEMPush_Vay(p, ei_Cart, bi_Cart);
          } else {
            raise::KernelError(HERE, "Invalid pusher algorithm for GCA mode");
          }
          if constexpr (HasExtForce or Atm) {
            particles.ux1(p) += HALF * ctx.dt * external_force_Cart[0];
            particles.ux2(p) += HALF * ctx.dt * external_force_Cart[1];
            particles.ux3(p) += HALF * ctx.dt * external_force_Cart[2];
          }
        }
        // radiative drag
        if constexpr (not HasEmission) {
          if ((not is_gca) and (ctx.radiative_drag_flags != RadiativeDrag::NONE)) {
            u_prime[0] = HALF * (u_prime[0] + particles.ux1(p));
            u_prime[1] = HALF * (u_prime[1] + particles.ux2(p));
            u_prime[2] = HALF * (u_prime[2] + particles.ux3(p));
            if (ctx.radiative_drag_flags & RadiativeDrag::SYNCHROTRON) {
              synchrotronDrag(p, u_prime, ei_Cart_rad, bi_Cart_rad);
            }
            if (ctx.radiative_drag_flags & RadiativeDrag::COMPTON) {
              inverseComptonDrag(p, u_prime);
            }
          }
        } else {
          u_prime[0] = HALF * (u_prime[0] + particles.ux1(p));
          u_prime[1] = HALF * (u_prime[1] + particles.ux2(p));
          u_prime[2] = HALF * (u_prime[2] + particles.ux3(p));
          processEmission(p, u_prime, xp_Cd, xp_Ph, ei_Cart_rad, bi_Cart_rad);
        }
        // update position
        positionPush(true, p, xp_Cd);
      }
    }

    // .......................
    // velocity pushers
    // .......................
    Inline void velocityEMPush_Boris(index_t          p,
                                     vec_t<Dim::_3D>& e0,
                                     vec_t<Dim::_3D>& b0) const {
      real_t COEFF { normalized_dt_half };

      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;
      vec_t<Dim::_3D> u0 { particles.ux1(p) + e0[0],
                           particles.ux2(p) + e0[1],
                           particles.ux3(p) + e0[2] };

      COEFF *= ONE / math::sqrt(ONE + NORM_SQR(u0[0], u0[1], u0[2]));
      b0[0] *= COEFF;
      b0[1] *= COEFF;
      b0[2] *= COEFF;
      COEFF  = TWO / (ONE + NORM_SQR(b0[0], b0[1], b0[2]));

      vec_t<Dim::_3D> u1 {
        (u0[0] + CROSS_x1(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
        (u0[1] + CROSS_x2(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF,
        (u0[2] + CROSS_x3(u0[0], u0[1], u0[2], b0[0], b0[1], b0[2])) * COEFF
      };

      u0[0] += CROSS_x1(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[0];
      u0[1] += CROSS_x2(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[1];
      u0[2] += CROSS_x3(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) + e0[2];

      particles.ux1(p) = u0[0];
      particles.ux2(p) = u0[1];
      particles.ux3(p) = u0[2];
    }

    Inline void velocityEMPush_Vay(index_t          p,
                                   vec_t<Dim::_3D>& e0,
                                   vec_t<Dim::_3D>& b0) const {
      auto COEFF { normalized_dt_half };
      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;

      b0[0] *= COEFF;
      b0[1] *= COEFF;
      b0[2] *= COEFF;

      COEFF = ONE / math::sqrt(ONE + NORM_SQR(particles.ux1(p),
                                              particles.ux2(p),
                                              particles.ux3(p)));

      vec_t<Dim::_3D> u1 { (particles.ux1(p) + TWO * e0[0] +
                            CROSS_x1(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF),
                           (particles.ux2(p) + TWO * e0[1] +
                            CROSS_x2(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF),
                           (particles.ux3(p) + TWO * e0[2] +
                            CROSS_x3(particles.ux1(p),
                                     particles.ux2(p),
                                     particles.ux3(p),
                                     b0[0],
                                     b0[1],
                                     b0[2]) *
                              COEFF) };
      COEFF = DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]);
      auto COEFF2 { ONE + NORM_SQR(u1[0], u1[1], u1[2]) -
                    NORM_SQR(b0[0], b0[1], b0[2]) };

      COEFF = ONE /
              math::sqrt(
                INV_2 * (COEFF2 + math::sqrt(SQR(COEFF2) +
                                             FOUR * (SQR(b0[0]) + SQR(b0[1]) +
                                                     SQR(b0[2]) + SQR(COEFF)))));
      COEFF2 = ONE / (ONE + SQR(b0[0] * COEFF) + SQR(b0[1] * COEFF) +
                      SQR(b0[2] * COEFF));

      particles.ux1(p) = COEFF2 *
                         (u1[0] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[0] * COEFF) +
                          u1[1] * b0[2] * COEFF - u1[2] * b0[1] * COEFF);
      particles.ux2(p) = COEFF2 *
                         (u1[1] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[1] * COEFF) +
                          u1[2] * b0[0] * COEFF - u1[0] * b0[2] * COEFF);
      particles.ux3(p) = COEFF2 *
                         (u1[2] +
                          COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                            (b0[2] * COEFF) +
                          u1[0] * b0[1] * COEFF - u1[1] * b0[0] * COEFF);
    }

    Inline void velocityEMPush_GCA(index_t          p,
                                   vec_t<Dim::_3D>& e0,
                                   vec_t<Dim::_3D>& b0) const {
      const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                          NORM_SQR(b0[0], b0[1], b0[2]) };

      const vec_t<Dim::_3D> wE {
        CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
      };

      {
        const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
        b0[0] *= b_norm_inv;
        b0[1] *= b_norm_inv;
        b0[2] *= b_norm_inv;
      }
      auto upar {
        DOT(particles.ux1(p), particles.ux2(p), particles.ux3(p), b0[0], b0[1], b0[2]) +
        normalized_dt_half * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2])
      };

      real_t factor;
      {
        const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
        if (wE_sqr < static_cast<real_t>(0.01)) {
          factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
        } else {
          factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
        }
      }
      const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
      const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                         math::sqrt(
                           ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
      particles.ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      particles.ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      particles.ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    Inline void velocityEMPush_GCA_ExtForce(index_t          p,
                                            vec_t<Dim::_3D>& f0,
                                            vec_t<Dim::_3D>& e0,
                                            vec_t<Dim::_3D>& b0) const {
      const auto eb_sqr { NORM_SQR(e0[0], e0[1], e0[2]) +
                          NORM_SQR(b0[0], b0[1], b0[2]) };

      const vec_t<Dim::_3D> wE {
        CROSS_x1(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x2(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr,
        CROSS_x3(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) / eb_sqr
      };

      {
        const auto b_norm_inv { ONE / NORM(b0[0], b0[1], b0[2]) };
        b0[0] *= b_norm_inv;
        b0[1] *= b_norm_inv;
        b0[2] *= b_norm_inv;
      }
      auto upar {
        DOT(particles.ux1(p), particles.ux2(p), particles.ux3(p), b0[0], b0[1], b0[2]) +
        normalized_dt_half * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) +
        ctx.dt * DOT(f0[0], f0[1], f0[2], b0[0], b0[1], b0[2])
      };

      real_t factor;
      {
        const auto wE_sqr { NORM_SQR(wE[0], wE[1], wE[2]) };
        if (wE_sqr < static_cast<real_t>(0.01)) {
          factor = ONE + wE_sqr + TWO * SQR(wE_sqr) + FIVE * SQR(wE_sqr) * wE_sqr;
        } else {
          factor = (ONE - math::sqrt(ONE - FOUR * wE_sqr)) / (TWO * wE_sqr);
        }
      }
      const vec_t<Dim::_3D> vE_Cart { wE[0] * factor, wE[1] * factor, wE[2] * factor };
      const auto Gamma { math::sqrt(ONE + SQR(upar)) /
                         math::sqrt(
                           ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
      particles.ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      particles.ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      particles.ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    // .......................
    // position pusher & bcs
    // .......................
    Inline void positionPush(bool massive, index_t p, coord_t<M::PrtlDim>& xp) const {
      // get cartesian velocity
      if constexpr (M::CoordType == Coord::Cartesian) {
        // i+di push for Cartesian basis
        const real_t dt_inv_energy {
          massive
            ? (ctx.dt / math::sqrt(ONE + SQR(particles.ux1(p)) +
                                   SQR(particles.ux2(p)) + SQR(particles.ux3(p))))
            : (ctx.dt / math::sqrt(SQR(particles.ux1(p)) + SQR(particles.ux2(p)) +
                                   SQR(particles.ux3(p))))
        };
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          particles.i1_prev(p)  = particles.i1(p);
          particles.dx1_prev(p) = particles.dx1(p);
          particles.dx1(p) += metric.template transform<1, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux1(p)) *
                              dt_inv_energy;
          particles.i1(p) += static_cast<int>(particles.dx1(p) >= ONE) -
                             static_cast<int>(particles.dx1(p) < ZERO);
          particles.dx1(p) -= (particles.dx1(p) >= ONE);
          particles.dx1(p) += (particles.dx1(p) < ZERO);
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          particles.i2_prev(p)  = particles.i2(p);
          particles.dx2_prev(p) = particles.dx2(p);
          particles.dx2(p) += metric.template transform<2, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux2(p)) *
                              dt_inv_energy;
          particles.i2(p) += static_cast<int>(particles.dx2(p) >= ONE) -
                             static_cast<int>(particles.dx2(p) < ZERO);
          particles.dx2(p) -= (particles.dx2(p) >= ONE);
          particles.dx2(p) += (particles.dx2(p) < ZERO);
        }
        if constexpr (D == Dim::_3D) {
          particles.i3_prev(p)  = particles.i3(p);
          particles.dx3_prev(p) = particles.dx3(p);
          particles.dx3(p) += metric.template transform<3, Idx::XYZ, Idx::U>(
                                xp,
                                particles.ux3(p)) *
                              dt_inv_energy;
          particles.i3(p) += static_cast<int>(particles.dx3(p) >= ONE) -
                             static_cast<int>(particles.dx3(p) < ZERO);
          particles.dx3(p) -= (particles.dx3(p) >= ONE);
          particles.dx3(p) += (particles.dx3(p) < ZERO);
        }
      } else {
        // full Cartesian coordinate push in non-Cartesian basis
        const real_t inv_energy {
          massive
            ? ONE / math::sqrt(ONE + SQR(particles.ux1(p)) +
                               SQR(particles.ux2(p)) + SQR(particles.ux3(p)))
            : ONE / math::sqrt(SQR(particles.ux1(p)) + SQR(particles.ux2(p)) +
                               SQR(particles.ux3(p)))
        };
        vec_t<Dim::_3D>     vp_Cart { particles.ux1(p) * inv_energy,
                                  particles.ux2(p) * inv_energy,
                                  particles.ux3(p) * inv_energy };
        // get cartesian position
        coord_t<M::PrtlDim> xp_Cart { ZERO };
        metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xp, xp_Cart);
        // update cartesian position
        for (auto d = 0u; d < M::PrtlDim; ++d) {
          xp_Cart[d] += vp_Cart[d] * ctx.dt;
        }
        // transform back to code
        metric.template convert_xyz<Crd::XYZ, Crd::Cd>(xp_Cart, xp);

        // update x1
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          particles.i1_prev(p)  = particles.i1(p);
          particles.dx1_prev(p) = particles.dx1(p);
          from_Xi_to_i_di(xp[0], particles.i1(p), particles.dx1(p));
        }

        // update x2 & phi
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          particles.i2_prev(p)  = particles.i2(p);
          particles.dx2_prev(p) = particles.dx2(p);
          from_Xi_to_i_di(xp[1], particles.i2(p), particles.dx2(p));
          if constexpr (D == Dim::_2D && M::PrtlDim == Dim::_3D) {
            particles.phi(p) = xp[2];
          }
        }

        // update x3
        if constexpr (D == Dim::_3D) {
          particles.i3_prev(p)  = particles.i3(p);
          particles.dx3_prev(p) = particles.dx3(p);
          from_Xi_to_i_di(xp[2], particles.i3(p), particles.dx3(p));
        }

        {
          coord_t<M::PrtlDim> xp_old { ZERO };
          getParticlePrevPosition(p, xp_old);
          if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
            xp[0] = HALF * (xp[0] + xp_old[0]);
          }
          if constexpr (D == Dim::_2D || D == Dim::_3D) {
            xp[1] = HALF * (xp[1] + xp_old[1]);
          }
          if constexpr (D == Dim::_3D) {
            xp[2] = HALF * (xp[2] + xp_old[2]);
          }
        }
      }

      // call custom particle position update
      if constexpr (HasCustomPrtlUpdate) {
        policies.custom_particle_update_policy(p, ctx, bc, particles, metric);
        getParticlePosition(p, xp);
      }

      if constexpr (M::CoordType != Coord::Cartesian) {
        // align xp with velocity @ n+1/2
        coord_t<M::PrtlDim> xp_old { ZERO };
        getParticlePrevPosition(p, xp_old);
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          xp[0] = HALF * (xp[0] + xp_old[0]);
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          xp[1] = HALF * (xp[1] + xp_old[1]);
        }
        if constexpr (D == Dim::_3D) {
          xp[2] = HALF * (xp[2] + xp_old[2]);
        }
      }

      // apply boundary conditions
      boundaryConditions(p, xp);
    }

    Inline void boundaryConditions(index_t              p,
                                   coord_t<M::PrtlDim>& xp_VelAligned) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i1(p) < 0) {
          if (bc.is_periodic_i1min) {
            particles.i1(p)      += ctx.ni1;
            particles.i1_prev(p) += ctx.ni1;
          } else if (bc.is_absorb_i1min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i1min) {
            particles.i1(p)  = 0;
            particles.dx1(p) = ONE - particles.dx1(p);
            invert_vel       = true;
          }
        } else if (particles.i1(p) >= ctx.ni1) {
          if (bc.is_periodic_i1max) {
            particles.i1(p)      -= ctx.ni1;
            particles.i1_prev(p) -= ctx.ni1;
          } else if (bc.is_absorb_i1max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i1max) {
            particles.i1(p)  = ctx.ni1 - 1;
            particles.dx1(p) = ONE - particles.dx1(p);
            invert_vel       = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cartesian) {
            particles.ux1(p) = -particles.ux1(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp_VelAligned,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[0] = -v[0];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_VelAligned, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i2(p) < 0) {
          if (bc.is_periodic_i2min) {
            particles.i2(p)      += ctx.ni2;
            particles.i2_prev(p) += ctx.ni2;
          } else if (bc.is_absorb_i2min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i2min) {
            particles.i2(p)  = 0;
            particles.dx2(p) = ONE - particles.dx2(p);
            invert_vel       = true;
          } else if (bc.is_axis_i2min) {
            particles.i2(p)  = 0;
            particles.dx2(p) = ONE - particles.dx2(p);
          }
        } else if (particles.i2(p) >= ctx.ni2) {
          if (bc.is_periodic_i2max) {
            particles.i2(p)      -= ctx.ni2;
            particles.i2_prev(p) -= ctx.ni2;
          } else if (bc.is_absorb_i2max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i2max) {
            particles.i2(p)  = ctx.ni2 - 1;
            particles.dx2(p) = ONE - particles.dx2(p);
            invert_vel       = true;
          } else if (bc.is_axis_i2max) {
            particles.i2(p)  = ctx.ni2 - 1;
            particles.dx2(p) = ONE - particles.dx2(p);
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cartesian) {
            particles.ux2(p) = -particles.ux2(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp_VelAligned,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[1] = -v[1];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_VelAligned, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_3D) {
        auto invert_vel = false;
        if (particles.i3(p) < 0) {
          if (bc.is_periodic_i3min) {
            particles.i3(p)      += ctx.ni3;
            particles.i3_prev(p) += ctx.ni3;
          } else if (bc.is_absorb_i3min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i3min) {
            particles.i3(p)  = 0;
            particles.dx3(p) = ONE - particles.dx3(p);
            invert_vel       = true;
          }
        } else if (particles.i3(p) >= ctx.ni3) {
          if (bc.is_periodic_i3max) {
            particles.i3(p)      -= ctx.ni3;
            particles.i3_prev(p) -= ctx.ni3;
          } else if (bc.is_absorb_i3max) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i3max) {
            particles.i3(p)  = ctx.ni3 - 1;
            particles.dx3(p) = ONE - particles.dx3(p);
            invert_vel       = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cartesian) {
            particles.ux3(p) = -particles.ux3(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp_VelAligned,
              { particles.ux1(p), particles.ux2(p), particles.ux3(p) },
              v);
            v[2] = -v[2];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_VelAligned, v, vXYZ);
            particles.ux1(p) = vXYZ[0];
            particles.ux2(p) = vXYZ[1];
            particles.ux3(p) = vXYZ[2];
          }
        }
      }
#if defined(MPI_ENABLED)
      if constexpr (D == Dim::_1D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1);
      } else if constexpr (D == Dim::_2D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1,
                                        particles.i2(p) < 0,
                                        particles.i2(p) >= ctx.ni2);
      } else if constexpr (D == Dim::_3D) {
        particles.tag(p) = mpi::SendTag(particles.tag(p),
                                        particles.i1(p) < 0,
                                        particles.i1(p) >= ctx.ni1,
                                        particles.i2(p) < 0,
                                        particles.i2(p) >= ctx.ni2,
                                        particles.i3(p) < 0,
                                        particles.i3(p) >= ctx.ni3);
      }
#endif
    }

    // .......................
    // helper functions
    // .......................
    Inline void getParticlePosition(index_t p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xp[0] = i_di_to_Xi(particles.i1(p), particles.dx1(p));
      }
      if constexpr (D == Dim::_2D) {
        xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp[2] = particles.phi(p);
        }
      }
      if constexpr (D == Dim::_3D) {
        xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));
        xp[2] = i_di_to_Xi(particles.i3(p), particles.dx3(p));
      }
    }

    Inline void getParticlePrevPosition(index_t p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xp[0] = i_di_to_Xi(particles.i1_prev(p), particles.dx1_prev(p));
      }
      if constexpr (D == Dim::_2D) {
        xp[1] = i_di_to_Xi(particles.i2_prev(p), particles.dx2_prev(p));
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp[2] = particles.phi(p);
        }
      }
      if constexpr (D == Dim::_3D) {
        xp[1] = i_di_to_Xi(particles.i2_prev(p), particles.dx2_prev(p));
        xp[2] = i_di_to_Xi(particles.i3_prev(p), particles.dx3_prev(p));
      }
    }

    template <unsigned short O>
    Inline void getInterpolatedEMFields(index_t          p,
                                        vec_t<Dim::_3D>& e0,
                                        vec_t<Dim::_3D>& b0) const {

      // Zig-zag interpolation
      if constexpr (O == 0u) {

        if constexpr (D == Dim::_1D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);

          // first order
          real_t c0, c1;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;

          // Ex1
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::ex1);
          c1    = EB(i + indx, em::ex1);
          e0[0] = c0 * pondmx + c1 * pondpx;
          // Ex2
          // Interpolate --- (primal)
          c0    = EB(i, em::ex2);
          c1    = EB(i + 1, em::ex2);
          e0[1] = c0 * ponpmx + c1 * ponppx;
          // Ex3
          // Interpolate --- (primal)
          c0    = EB(i, em::ex3);
          c1    = EB(i + 1, em::ex3);
          e0[2] = c0 * ponpmx + c1 * ponppx;
          // Bx1
          // Interpolate --- (primal)
          c0    = EB(i, em::bx1);
          c1    = EB(i + 1, em::bx1);
          b0[0] = c0 * ponpmx + c1 * ponppx;
          // Bx2
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::bx2);
          c1    = EB(i + indx, em::bx2);
          b0[1] = c0 * pondmx + c1 * pondpx;
          // Bx3
          // Interpolate --- (dual)
          c0    = EB(i - 1 + indx, em::bx3);
          c1    = EB(i + indx, em::bx3);
          b0[2] = c0 * pondmx + c1 * pondpx;
        } else if constexpr (D == Dim::_2D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);
          int indy = static_cast<int>(dx2_ + HALF);

          // first order
          real_t c000, c100, c010, c110, c00, c10;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;
          real_t ponpmy = ONE - dx2_;
          real_t ponppy = dx2_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;
          real_t pondmy = static_cast<real_t>(indy + ONE) - (dx2_ + HALF);
          real_t pondpy = ONE - pondmy;

          // Ex1
          // Interpolate --- (dual, primal)
          c000  = EB(i - 1 + indx, j, em::ex1);
          c100  = EB(i + indx, j, em::ex1);
          c010  = EB(i - 1 + indx, j + 1, em::ex1);
          c110  = EB(i + indx, j + 1, em::ex1);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          e0[0] = c00 * ponpmy + c10 * ponppy;
          // Ex2
          // Interpolate -- (primal, dual)
          c000  = EB(i, j - 1 + indy, em::ex2);
          c100  = EB(i + 1, j - 1 + indy, em::ex2);
          c010  = EB(i, j + indy, em::ex2);
          c110  = EB(i + 1, j + indy, em::ex2);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          e0[1] = c00 * pondmy + c10 * pondpy;
          // Ex3
          // Interpolate -- (primal, primal)
          c000  = EB(i, j, em::ex3);
          c100  = EB(i + 1, j, em::ex3);
          c010  = EB(i, j + 1, em::ex3);
          c110  = EB(i + 1, j + 1, em::ex3);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          e0[2] = c00 * ponpmy + c10 * ponppy;

          // Bx1
          // Interpolate -- (primal, dual)
          c000  = EB(i, j - 1 + indy, em::bx1);
          c100  = EB(i + 1, j - 1 + indy, em::bx1);
          c010  = EB(i, j + indy, em::bx1);
          c110  = EB(i + 1, j + indy, em::bx1);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          b0[0] = c00 * pondmy + c10 * pondpy;
          // Bx2
          // Interpolate -- (dual, primal)
          c000  = EB(i - 1 + indx, j, em::bx2);
          c100  = EB(i + indx, j, em::bx2);
          c010  = EB(i - 1 + indx, j + 1, em::bx2);
          c110  = EB(i + indx, j + 1, em::bx2);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          b0[1] = c00 * ponpmy + c10 * ponppy;
          // Bx3
          // Interpolate -- (dual, dual)
          c000  = EB(i - 1 + indx, j - 1 + indy, em::bx3);
          c100  = EB(i + indx, j - 1 + indy, em::bx3);
          c010  = EB(i - 1 + indx, j + indy, em::bx3);
          c110  = EB(i + indx, j + indy, em::bx3);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          b0[2] = c00 * pondmy + c10 * pondpy;
        } else if constexpr (D == Dim::_3D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const int  k { particles.i3(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };
          const auto dx3_ { static_cast<real_t>(particles.dx3(p)) };

          // direct interpolation - Arno
          int indx = static_cast<int>(dx1_ + HALF);
          int indy = static_cast<int>(dx2_ + HALF);
          int indz = static_cast<int>(dx3_ + HALF);

          // first order
          real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01,
            c11, c0, c1;

          real_t ponpmx = ONE - dx1_;
          real_t ponppx = dx1_;
          real_t ponpmy = ONE - dx2_;
          real_t ponppy = dx2_;
          real_t ponpmz = ONE - dx3_;
          real_t ponppz = dx3_;

          real_t pondmx = static_cast<real_t>(indx + ONE) - (dx1_ + HALF);
          real_t pondpx = ONE - pondmx;
          real_t pondmy = static_cast<real_t>(indy + ONE) - (dx2_ + HALF);
          real_t pondpy = ONE - pondmy;
          real_t pondmz = static_cast<real_t>(indz + ONE) - (dx3_ + HALF);
          real_t pondpz = ONE - pondmz;

          // Ex1
          // Interpolate --- (dual, primal, primal)
          c000  = EB(i - 1 + indx, j, k, em::ex1);
          c100  = EB(i + indx, j, k, em::ex1);
          c010  = EB(i - 1 + indx, j + 1, k, em::ex1);
          c110  = EB(i + indx, j + 1, k, em::ex1);
          c001  = EB(i - 1 + indx, j, k + 1, em::ex1);
          c101  = EB(i + indx, j, k + 1, em::ex1);
          c011  = EB(i - 1 + indx, j + 1, k + 1, em::ex1);
          c111  = EB(i + indx, j + 1, k + 1, em::ex1);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          e0[0] = c0 * ponpmz + c1 * ponppz;
          // Ex2
          // Interpolate -- (primal, dual, primal)
          c000  = EB(i, j - 1 + indy, k, em::ex2);
          c100  = EB(i + 1, j - 1 + indy, k, em::ex2);
          c010  = EB(i, j + indy, k, em::ex2);
          c110  = EB(i + 1, j + indy, k, em::ex2);
          c001  = EB(i, j - 1 + indy, k + 1, em::ex2);
          c101  = EB(i + 1, j - 1 + indy, k + 1, em::ex2);
          c011  = EB(i, j + indy, k + 1, em::ex2);
          c111  = EB(i + 1, j + indy, k + 1, em::ex2);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * pondmy + c10 * pondpy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * pondmy + c11 * pondpy;
          e0[1] = c0 * ponpmz + c1 * ponppz;
          // Ex3
          // Interpolate -- (primal, primal, dual)
          c000  = EB(i, j, k - 1 + indz, em::ex3);
          c100  = EB(i + 1, j, k - 1 + indz, em::ex3);
          c010  = EB(i, j + 1, k - 1 + indz, em::ex3);
          c110  = EB(i + 1, j + 1, k - 1 + indz, em::ex3);
          c001  = EB(i, j, k + indz, em::ex3);
          c101  = EB(i + 1, j, k + indz, em::ex3);
          c011  = EB(i, j + 1, k + indz, em::ex3);
          c111  = EB(i + 1, j + 1, k + indz, em::ex3);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * ponpmy + c11 * ponppy;
          e0[2] = c0 * pondmz + c1 * pondpz;

          // Bx1
          // Interpolate -- (primal, dual, dual)
          c000  = EB(i, j - 1 + indy, k - 1 + indz, em::bx1);
          c100  = EB(i + 1, j - 1 + indy, k - 1 + indz, em::bx1);
          c010  = EB(i, j + indy, k - 1 + indz, em::bx1);
          c110  = EB(i + 1, j + indy, k - 1 + indz, em::bx1);
          c001  = EB(i, j - 1 + indy, k + indz, em::bx1);
          c101  = EB(i + 1, j - 1 + indy, k + indz, em::bx1);
          c011  = EB(i, j + indy, k + indz, em::bx1);
          c111  = EB(i + 1, j + indy, k + indz, em::bx1);
          c00   = c000 * ponpmx + c100 * ponppx;
          c10   = c010 * ponpmx + c110 * ponppx;
          c0    = c00 * pondmy + c10 * pondpy;
          c01   = c001 * ponpmx + c101 * ponppx;
          c11   = c011 * ponpmx + c111 * ponppx;
          c1    = c01 * pondmy + c11 * pondpy;
          b0[0] = c0 * pondmz + c1 * pondpz;
          // Bx2
          // Interpolate -- (dual, primal, dual)
          c000  = EB(i - 1 + indx, j, k - 1 + indz, em::bx2);
          c100  = EB(i + indx, j, k - 1 + indz, em::bx2);
          c010  = EB(i - 1 + indx, j + 1, k - 1 + indz, em::bx2);
          c110  = EB(i + indx, j + 1, k - 1 + indz, em::bx2);
          c001  = EB(i - 1 + indx, j, k + indz, em::bx2);
          c101  = EB(i + indx, j, k + indz, em::bx2);
          c011  = EB(i - 1 + indx, j + 1, k + indz, em::bx2);
          c111  = EB(i + indx, j + 1, k + indz, em::bx2);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          b0[1] = c0 * pondmz + c1 * pondpz;
          // Bx3
          // Interpolate -- (dual, dual, primal)
          c000  = EB(i - 1 + indx, j - 1 + indy, k, em::bx3);
          c100  = EB(i + indx, j - 1 + indy, k, em::bx3);
          c010  = EB(i - 1 + indx, j + indy, k, em::bx3);
          c110  = EB(i + indx, j + indy, k, em::bx3);
          c001  = EB(i - 1 + indx, j - 1 + indy, k + 1, em::bx3);
          c101  = EB(i + indx, j - 1 + indy, k + 1, em::bx3);
          c011  = EB(i - 1 + indx, j + indy, k + 1, em::bx3);
          c111  = EB(i + indx, j + indy, k + 1, em::bx3);
          c00   = c000 * pondmx + c100 * pondpx;
          c10   = c010 * pondmx + c110 * pondpx;
          c0    = c00 * ponpmy + c10 * ponppy;
          c01   = c001 * pondmx + c101 * pondpx;
          c11   = c011 * pondmx + c111 * pondpx;
          c1    = c01 * ponpmy + c11 * ponppy;
          b0[2] = c0 * ponpmz + c1 * ponppz;
        }
      } else if constexpr (O >= 1u) {

        if constexpr (D == Dim::_1D) {
          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          // primal and dual shape function
          real_t     Sp[O + 1], Sd[O + 1];
          // minimum contributing cells
          int        ip_min, id_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, Sp);

          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, Sd);

          // Ex1 -- dual
          e0[0] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[0] += Sd[idx1] * EB(id_min + idx1, em::ex1);
          }

          // Ex2 -- primal
          e0[1] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[1] += Sp[idx1] * EB(ip_min + idx1, em::ex2);
          }

          // Ex3 -- primal
          e0[2] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            e0[2] += Sp[idx1] * EB(ip_min + idx1, em::ex3);
          }

          // Bx1 -- primal
          b0[0] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[0] += Sp[idx1] * EB(ip_min + idx1, em::bx1);
          }

          // Bx2 -- dual
          b0[1] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[1] += Sd[idx1] * EB(id_min + idx1, em::bx2);
          }

          // Bx3 -- dual
          b0[2] = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            b0[2] += Sd[idx1] * EB(id_min + idx1, em::bx3);
          }

        } else if constexpr (D == Dim::_2D) {

          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };

          // primal and dual shape function
          real_t S1p[O + 1], S1d[O + 1];
          real_t S2p[O + 1], S2d[O + 1];
          // minimum contributing cells
          int    ip_min, id_min;
          int    jp_min, jd_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, S1p);
          prtl_shape::order<false, O>(j, dx2_, jp_min, S2p);
          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, S1d);
          prtl_shape::order<true, O>(j, dx2_, jd_min, S2d);

          // Ex1 -- dual, primal
          e0[0] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::ex1);
            }
            e0[0] += c0 * S2p[idx2];
          }

          // Ex2 -- primal, dual
          e0[1] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::ex2);
            }
            e0[1] += c0 * S2d[idx2];
          }

          // Ex3 -- primal, primal
          e0[2] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jp_min + idx2, em::ex3);
            }
            e0[2] += c0 * S2p[idx2];
          }

          // Bx1 -- primal, dual
          b0[0] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1p[idx1] * EB(ip_min + idx1, jd_min + idx2, em::bx1);
            }
            b0[0] += c0 * S2d[idx2];
          }

          // Bx2 -- dual, primal
          b0[1] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jp_min + idx2, em::bx2);
            }
            b0[1] += c0 * S2p[idx2];
          }

          // Bx3 -- dual, dual
          b0[2] = ZERO;
          for (int idx2 = 0; idx2 < O + 1; idx2++) {
            real_t c0 = ZERO;
            for (int idx1 = 0; idx1 < O + 1; idx1++) {
              c0 += S1d[idx1] * EB(id_min + idx1, jd_min + idx2, em::bx3);
            }
            b0[2] += c0 * S2d[idx2];
          }

        } else if constexpr (D == Dim::_3D) {

          const int  i { particles.i1(p) + static_cast<int>(N_GHOSTS) };
          const int  j { particles.i2(p) + static_cast<int>(N_GHOSTS) };
          const int  k { particles.i3(p) + static_cast<int>(N_GHOSTS) };
          const auto dx1_ { static_cast<real_t>(particles.dx1(p)) };
          const auto dx2_ { static_cast<real_t>(particles.dx2(p)) };
          const auto dx3_ { static_cast<real_t>(particles.dx3(p)) };

          // primal and dual shape function
          real_t S1p[O + 1], S1d[O + 1];
          real_t S2p[O + 1], S2d[O + 1];
          real_t S3p[O + 1], S3d[O + 1];

          // minimum contributing cells
          int ip_min, id_min;
          int jp_min, jd_min;
          int kp_min, kd_min;

          // primal shape function - not staggered
          prtl_shape::order<false, O>(i, dx1_, ip_min, S1p);
          prtl_shape::order<false, O>(j, dx2_, jp_min, S2p);
          prtl_shape::order<false, O>(k, dx3_, kp_min, S3p);
          // dual shape function - staggered
          prtl_shape::order<true, O>(i, dx1_, id_min, S1d);
          prtl_shape::order<true, O>(j, dx2_, jd_min, S2d);
          prtl_shape::order<true, O>(k, dx3_, kd_min, S3d);

          // Ex1 -- dual, primal, primal
          e0[0] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jp_min + idx2, kp_min + idx3, em::ex1);
              }
              c0 += c00 * S2p[idx2];
            }
            e0[0] += c0 * S3p[idx3];
          }

          // Ex2 -- primal, dual, primal
          e0[1] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jd_min + idx2, kp_min + idx3, em::ex2);
              }
              c0 += c00 * S2d[idx2];
            }
            e0[1] += c0 * S3p[idx3];
          }

          // Ex3 -- primal, primal, dual
          e0[2] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jp_min + idx2, kd_min + idx3, em::ex3);
              }
              c0 += c00 * S2p[idx2];
            }
            e0[2] += c0 * S3d[idx3];
          }

          // Bx1 -- primal, dual, dual
          b0[0] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1p[idx1] *
                       EB(ip_min + idx1, jd_min + idx2, kd_min + idx3, em::bx1);
              }
              c0 += c00 * S2d[idx2];
            }
            b0[0] += c0 * S3d[idx3];
          }

          // Bx2 -- dual, primal, dual
          b0[1] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jp_min + idx2, kd_min + idx3, em::bx2);
              }
              c0 += c00 * S2p[idx2];
            }
            b0[1] += c0 * S3d[idx3];
          }

          // Bx3 -- dual, dual, primal
          b0[2] = ZERO;
          for (int idx3 = 0; idx3 < O + 1; idx3++) {
            real_t c0 = ZERO;
            for (int idx2 = 0; idx2 < O + 1; idx2++) {
              real_t c00 = ZERO;
              for (int idx1 = 0; idx1 < O + 1; idx1++) {
                c00 += S1d[idx1] *
                       EB(id_min + idx1, jd_min + idx2, kp_min + idx3, em::bx3);
              }
              c0 += c00 * S2d[idx2];
            }
            b0[2] += c0 * S3p[idx3];
          }
        }
      }
    }

    Inline void synchrotronDrag(index_t                p,
                                vec_t<Dim::_3D>&       u_prime,
                                const vec_t<Dim::_3D>& e0,
                                const vec_t<Dim::_3D>& b0) const {
      real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
      u_prime[0]             *= gamma_prime_sqr;
      u_prime[1]             *= gamma_prime_sqr;
      u_prime[2]             *= gamma_prime_sqr;
      gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);
      const real_t beta_dot_e {
        DOT(u_prime[0], u_prime[1], u_prime[2], e0[0], e0[1], e0[2])
      };
      vec_t<Dim::_3D> e_plus_beta_cross_b {
        e0[0] + CROSS_x1(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[1] + CROSS_x2(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2]),
        e0[2] + CROSS_x3(u_prime[0], u_prime[1], u_prime[2], b0[0], b0[1], b0[2])
      };
      vec_t<Dim::_3D> kappaR {
        CROSS_x1(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[0],
        CROSS_x2(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[1],
        CROSS_x3(e_plus_beta_cross_b[0],
                 e_plus_beta_cross_b[1],
                 e_plus_beta_cross_b[2],
                 b0[0],
                 b0[1],
                 b0[2]) +
          beta_dot_e * e0[2],
      };
      const real_t chiR_sqr { NORM_SQR(e_plus_beta_cross_b[0],
                                       e_plus_beta_cross_b[1],
                                       e_plus_beta_cross_b[2]) -
                              SQR(beta_dot_e) };
      particles.ux1(p) += ctx.synchrotron_drag.coeff *
                          (kappaR[0] - gamma_prime_sqr * u_prime[0] * chiR_sqr);
      particles.ux2(p) += ctx.synchrotron_drag.coeff *
                          (kappaR[1] - gamma_prime_sqr * u_prime[1] * chiR_sqr);
      particles.ux3(p) += ctx.synchrotron_drag.coeff *
                          (kappaR[2] - gamma_prime_sqr * u_prime[2] * chiR_sqr);
    }

    Inline void getExternalForce(const coord_t<M::PrtlDim>& xp_Cd,
                                 const coord_t<M::PrtlDim>& xp_Ph,
                                 vec_t<Dim::_3D>& external_force_Cart) const
      requires(Atm or HasExtForce)
    {
      real_t f_x1 = ZERO, f_x2 = ZERO, f_x3 = ZERO;
      if constexpr (HasExtFx1) {
        f_x1 = policies.external_fields_policy.fx1(xp_Ph);
      }
      if constexpr (HasExtFx2) {
        f_x2 = policies.external_fields_policy.fx2(xp_Ph);
      }
      if constexpr (HasExtFx3) {
        f_x3 = policies.external_fields_policy.fx3(xp_Ph);
      }
      if constexpr (Atm) {
        if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
          if (not cmp::AlmostZero(ctx.atmosphere.gx1) and
              ((ctx.atmosphere.ds < ZERO or
                xp_Ph[0] <= ctx.atmosphere.x_surf + ctx.atmosphere.ds) and
               (ctx.atmosphere.ds > ZERO or
                xp_Ph[0] >= ctx.atmosphere.x_surf + ctx.atmosphere.ds))) {
            if constexpr (M::CoordType == Coord::Cartesian) {
              f_x1 += ctx.atmosphere.gx1;
            } else {
              f_x1 += ctx.atmosphere.gx1 * SQR(ctx.atmosphere.x_surf / xp_Ph[0]);
            }
          }
        }
        if constexpr (D == Dim::_2D or D == Dim::_3D) {
          if (not cmp::AlmostZero(ctx.atmosphere.gx2) and
              ((ctx.atmosphere.ds < ZERO or
                xp_Ph[1] <= ctx.atmosphere.x_surf + ctx.atmosphere.ds) and
               (ctx.atmosphere.ds > ZERO or
                xp_Ph[1] >= ctx.atmosphere.x_surf + ctx.atmosphere.ds))) {
            if constexpr (M::CoordType == Coord::Cartesian) {
              f_x2 += ctx.atmosphere.gx2;
            } else {
              raise::KernelError(HERE, "Invalid force for coordinate system");
            }
          }
        }
        if constexpr (D == Dim::_3D) {
          if (not cmp::AlmostZero(ctx.atmosphere.gx3) and
              ((ctx.atmosphere.ds < ZERO or
                xp_Ph[2] <= ctx.atmosphere.x_surf + ctx.atmosphere.ds) and
               (ctx.atmosphere.ds > ZERO or
                xp_Ph[2] >= ctx.atmosphere.x_surf + ctx.atmosphere.ds))) {
            if constexpr (M::CoordType == Coord::Cartesian) {
              f_x3 += ctx.atmosphere.gx3;
            } else {
              raise::KernelError(HERE, "Invalid force for coordinate system");
            }
          }
        }
      }
      metric.template transform_xyz<Idx::T, Idx::XYZ>(xp_Cd,
                                                      { f_x1, f_x2, f_x3 },
                                                      external_force_Cart);
    }

    Inline void inverseComptonDrag(index_t p, vec_t<Dim::_3D>& u_prime) const {
      real_t gamma_prime_sqr  = ONE / math::sqrt(ONE + NORM_SQR(u_prime[0],
                                                               u_prime[1],
                                                               u_prime[2]));
      u_prime[0]             *= gamma_prime_sqr;
      u_prime[1]             *= gamma_prime_sqr;
      u_prime[2]             *= gamma_prime_sqr;
      gamma_prime_sqr         = SQR(ONE / gamma_prime_sqr);

      particles.ux1(p) -= ctx.compton_drag.coeff * gamma_prime_sqr * u_prime[0];
      particles.ux2(p) -= ctx.compton_drag.coeff * gamma_prime_sqr * u_prime[1];
      particles.ux3(p) -= ctx.compton_drag.coeff * gamma_prime_sqr * u_prime[2];
    }

    Inline void processEmission(index_t                    p,
                                vec_t<Dim::_3D>&           u_prime,
                                const coord_t<M::PrtlDim>& xp_Cd,
                                const coord_t<M::PrtlDim>& xp_Ph,
                                const vec_t<Dim::_3D>&     ep_Cart,
                                const vec_t<Dim::_3D>&     bp_Cart) const
      requires(not ::traits::emission::IsNoPolicy<E>)
    {
      typename E::Payload payload;
      vec_t<Dim::_3D>     delta_u_Ph { ZERO };
      const auto emission_response = policies.emission_policy.shouldEmit(xp_Cd,
                                                                         xp_Ph,
                                                                         u_prime,
                                                                         ep_Cart,
                                                                         bp_Cart,
                                                                         delta_u_Ph,
                                                                         payload);

      if (emission_response.second) {
        particles.ux1(p) += delta_u_Ph[0];
        particles.ux2(p) += delta_u_Ph[1];
        particles.ux3(p) += delta_u_Ph[2];
      }

      if (emission_response.first) {
        vec_t<Dim::_3D> direction { ZERO };
        const auto      delta_u_Ph_mag = NORM(delta_u_Ph[0],
                                         delta_u_Ph[1],
                                         delta_u_Ph[2]);
        direction[0]                   = -delta_u_Ph[0] / delta_u_Ph_mag;
        direction[1]                   = -delta_u_Ph[1] / delta_u_Ph_mag;
        direction[2]                   = -delta_u_Ph[2] / delta_u_Ph_mag;
        tuple_t<int, M::Dim>      xi_Cd { 0 };
        tuple_t<prtldx_t, M::Dim> dxi_Cd { static_cast<prtldx_t>(0) };
        real_t                    prtl_phi = ZERO;
        if constexpr (M::Dim == Dim::_1D or M::Dim == Dim::_2D or
                      M::Dim == Dim::_3D) {
          xi_Cd[0]  = particles.i1(p);
          dxi_Cd[0] = particles.dx1(p);
        }
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          xi_Cd[1]  = particles.i2(p);
          dxi_Cd[1] = particles.dx2(p);
          if constexpr (M::CoordType != Coord::Cartesian) {
            prtl_phi = particles.phi(p);
          }
        }
        if constexpr (M::Dim == Dim::_3D) {
          xi_Cd[2]  = particles.i3(p);
          dxi_Cd[2] = particles.dx3(p);
        }
        policies.emission_policy
          .emit(xi_Cd, dxi_Cd, direction, particles.weight(p), prtl_phi, payload);
      }
    }
  };

} // namespace kernel::sr

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // KERNELS_PUSHERS_SR_HPP

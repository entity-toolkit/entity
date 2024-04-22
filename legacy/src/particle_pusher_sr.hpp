/**
 * @file kernels/particle_pusher_sr.h
 * @brief Particle pusher for the SR
 * @implements
 *   - kernel::sr::Pusher_kernel<>
 *   - kernel::sr::PusherBase_kernel<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - arch/traits.h
 *   - utils/error.h
 *   - utils/numeric.h
 * @namespaces:
 *   - kernel::sr::
 */

#ifndef KERNELS_PARTICLE_PUSHER_SR_HPP
#define KERNELS_PARTICLE_PUSHER_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <tuple>
#include <type_traits>

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  { I = static_cast<int>((XI)); }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    DI = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);               \
  }

#define i_di_to_Xi(I, DI) static_cast<real_t>((I)) + static_cast<real_t>((DI))

/* -------------------------------------------------------------------------- */

namespace kernel::sr {
  using namespace ntt;

  // Pushers
  struct Boris_t {};

  struct Vay_t {};

  struct Photon_t {};

  struct Extforce_t {};

  struct GCA_t {};

  // Cooling
  struct NoCooling_t {};

  struct Synchrotron_t {};

  template <typename... Tags>
  struct Union_t {};

  template <typename T, typename... Ts>
  struct is_contained;

  template <typename T, typename F, typename... Ts>
  struct is_contained<T, F, Ts...> : is_contained<T, Ts...> {};

  template <typename T, typename... Ts>
  struct is_contained<T, T, Ts...> : std::true_type {};

  template <typename T>
  struct is_contained<T> : std::false_type {};

  struct Massive_t {};

  struct Massless_t {};

  /**
   * @tparam M Metric
   * @tparam PG Problem generator
   * @tparam P Particle pusher
   * @tparam Cs Cooling algorithms
   */
  template <class M>
  struct Pusher_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    // static_assert(PG::is_pgen, "PG must be a problem generator class");
    static constexpr auto D = M::Dim;
    // using base_t::defines_fx1;
    // using base_t::defines_fx2;
    // using base_t::defines_fx3;

  private:
    const ndfield_t<D, 6> EB;
    const unsigned short  sp;
    array_t<int*>         i1, i2, i3;
    array_t<int*>         i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>    dx1, dx2, dx3;
    array_t<prtldx_t*>    dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>      ux1, ux2, ux3;
    array_t<real_t*>      phi;
    array_t<short*>       tag;
    const M               metric;
    // const PG              pgen;

    const real_t time, coeff, dt;
    const int    ni1, ni2, ni3;
    bool         is_absorb_i1min { false }, is_absorb_i1max { false };
    bool         is_absorb_i2min { false }, is_absorb_i2max { false };
    bool         is_absorb_i3min { false }, is_absorb_i3max { false };
    bool         is_periodic_i1min { false }, is_periodic_i1max { false };
    bool         is_periodic_i2min { false }, is_periodic_i2max { false };
    bool         is_periodic_i3min { false }, is_periodic_i3max { false };
    bool         is_axis_i2min { false }, is_axis_i2max { false };
    // gca parameters
    const real_t gca_larmor, gca_EovrB_sqr;
    // synchrotron cooling parameters
    const real_t coeff_sync;

  public:
    Pusher_kernel(const ndfield_t<D, 6>&      EB,
                  unsigned short              sp,
                  array_t<int*>&              i1,
                  array_t<int*>&              i2,
                  array_t<int*>&              i3,
                  array_t<int*>&              i1_prev,
                  array_t<int*>&              i2_prev,
                  array_t<int*>&              i3_prev,
                  array_t<prtldx_t*>&         dx1,
                  array_t<prtldx_t*>&         dx2,
                  array_t<prtldx_t*>&         dx3,
                  array_t<prtldx_t*>&         dx1_prev,
                  array_t<prtldx_t*>&         dx2_prev,
                  array_t<prtldx_t*>&         dx3_prev,
                  array_t<real_t*>&           ux1,
                  array_t<real_t*>&           ux2,
                  array_t<real_t*>&           ux3,
                  array_t<real_t*>&           phi,
                  array_t<short*>&            tag,
                  const M&                    metric,
                  // const PG&                   pgen,
                  real_t                      time,
                  real_t                      coeff,
                  real_t                      dt,
                  int                         ni1,
                  int                         ni2,
                  int                         ni3,
                  const boundaries_t<PrtlBC>& boundaries,
                  real_t                      gca_larmor_max,
                  real_t                      gca_eovrb_max,
                  real_t                      coeff_sync) :
      EB { EB },
      sp { sp },
      i1 { i1 },
      i2 { i2 },
      i3 { i3 },
      i1_prev { i1_prev },
      i2_prev { i2_prev },
      i3_prev { i3_prev },
      dx1 { dx1 },
      dx2 { dx2 },
      dx3 { dx3 },
      dx1_prev { dx1_prev },
      dx2_prev { dx2_prev },
      dx3_prev { dx3_prev },
      ux1 { ux1 },
      ux2 { ux2 },
      ux3 { ux3 },
      phi { phi },
      tag { tag },
      metric { metric },
      // pgen { pgen },
      time { time },
      coeff { coeff },
      dt { dt },
      ni1 { ni1 },
      ni2 { ni2 },
      ni3 { ni3 },
      gca_larmor { gca_larmor_max },
      gca_EovrB_sqr { SQR(gca_eovrb_max) },
      coeff_sync { coeff_sync } {

      raise::ErrorIf(boundaries.size() < 1, "boundaries defined incorrectly", HERE);
      is_absorb_i1min = (boundaries[0].first == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].first == PrtlBC::ABSORB);
      is_absorb_i1max = (boundaries[0].second == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].second == PrtlBC::ABSORB);
      is_periodic_i1min = (boundaries[0].first == PrtlBC::PERIODIC);
      is_periodic_i1max = (boundaries[0].second == PrtlBC::PERIODIC);
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_absorb_i2min = (boundaries[1].first == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].first == PrtlBC::ABSORB);
        is_absorb_i2max = (boundaries[1].second == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].second == PrtlBC::ABSORB);
        is_periodic_i2min = (boundaries[1].first == PrtlBC::PERIODIC);
        is_periodic_i2max = (boundaries[1].second == PrtlBC::PERIODIC);
        is_axis_i2min     = (boundaries[1].first == PrtlBC::AXIS);
        is_axis_i2max     = (boundaries[1].second == PrtlBC::AXIS);
      }
      if constexpr (D == Dim::_3D) {
        raise::ErrorIf(boundaries.size() < 3, "boundaries defined incorrectly", HERE);
        is_absorb_i3min = (boundaries[2].first == PrtlBC::ATMOSPHERE) ||
                          (boundaries[2].first == PrtlBC::ABSORB);
        is_absorb_i3max = (boundaries[2].second == PrtlBC::ATMOSPHERE) ||
                          (boundaries[2].second == PrtlBC::ABSORB);
        is_periodic_i3min = (boundaries[2].first == PrtlBC::PERIODIC);
        is_periodic_i3max = (boundaries[2].second == PrtlBC::PERIODIC);
      }
    }

    Inline void synchrotronDrag(index_t&               p,
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
      ux1(p) += coeff_sync * (kappaR[0] - gamma_prime_sqr * u_prime[0] * chiR_sqr);
      ux2(p) += coeff_sync * (kappaR[1] - gamma_prime_sqr * u_prime[1] * chiR_sqr);
      ux3(p) += coeff_sync * (kappaR[2] - gamma_prime_sqr * u_prime[2] * chiR_sqr);
    }

    /* photon --------------------------------------------------------------- */
    Inline void operator()(Photon_t, index_t p) const {
      if (tag(p) != ParticleTag::alive) {
        return;
      }
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      getPrtlPos(p, xp_Cd);
      posUpd<Massless_t>(p, xp_Cd);
    }

    // /* Vay/Boris, no gca, no ext force, no cooling -------------------------- */
    // template <typename P>
    // Inline void operator()(P, index_t p) const {
    //   if (tag(p) != ParticleTag::alive) {
    //     return;
    //   }
    //   coord_t<M::PrtlDim> xp_Cd { ZERO };
    //   getPrtlPos(p, xp_Cd);
    //   vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
    //   vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };

    //   getInterpFlds(p, ei, bi);
    //   metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
    //   metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);
    //   velUpd(P {}, p, ei_Cart, bi_Cart);
    //   posUpd<Massive_t>(p, xp_Cd);
    // }

    /* general case --------------------------------------------------------- */
    template <typename P, typename... Args>
    Inline void operator()(Union_t<P, Args...>, index_t p) const {
      if (tag(p) != ParticleTag::alive) {
        return;
      }
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      getPrtlPos(p, xp_Cd);
      // update cartesian velocity
      vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
      vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
      vec_t<Dim::_3D> force_Cart { ZERO };
      vec_t<Dim::_3D> u_prime { ZERO };
      vec_t<Dim::_3D> ei_Cart_rad { ZERO }, bi_Cart_rad { ZERO };
      bool            is_gca { false };

      getInterpFlds(p, ei, bi);
      metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
      metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);
      if constexpr (is_contained<Synchrotron_t, Union_t<Args...>>::value) {
        // backup fields & velocities to use later in cooling
        ei_Cart_rad[0] = ei_Cart[0];
        ei_Cart_rad[1] = ei_Cart[1];
        ei_Cart_rad[2] = ei_Cart[2];
        bi_Cart_rad[0] = bi_Cart[0];
        bi_Cart_rad[1] = bi_Cart[1];
        bi_Cart_rad[2] = bi_Cart[2];
        u_prime[0]     = ux1(p);
        u_prime[1]     = ux2(p);
        u_prime[2]     = ux3(p);
      }
      if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
        // coord_t<M::PrtlDim> xp_Ph { ZERO };
        // xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
        // if constexpr (M::PrtlDim != Dim::_1D) {
        //   xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
        // }
        // if constexpr (M::PrtlDim == Dim::_3D) {
        //   xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
        // }
        // real_t fx1 { ZERO }, fx2 { ZERO }, fx3 { ZERO };
        // metric.template transform_xyz<Idx::T, Idx::XYZ>(
        //   xp_Cd,
        //   { pgen.ext_force.fx1(sp, time, xp_Ph),
        //     pgen.ext_force.fx2(sp, time, xp_Ph),
        //     pgen.ext_force.fx3(sp, time, xp_Ph) },
        //   force_Cart);
      }
      if constexpr (is_contained<GCA_t, Union_t<Args...>>::value) {
        /* hybrid GCA/conventional mode --------------------------------- */
        const auto E2 { NORM_SQR(ei_Cart[0], ei_Cart[1], ei_Cart[2]) };
        const auto B2 { NORM_SQR(bi_Cart[0], bi_Cart[1], bi_Cart[2]) };
        const auto rL { math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p))) *
                        dt / (TWO * math::abs(coeff) * math::sqrt(B2)) };
        if (B2 > ZERO && rL < gca_larmor && (E2 / B2) < gca_EovrB_sqr) {
          is_gca = true;
          // update with GCA
          if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
            velUpd(Union_t<GCA_t, Extforce_t> {}, p, force_Cart, ei_Cart, bi_Cart);
          } else {
            velUpd(GCA_t {}, p, ei_Cart, bi_Cart);
          }
        } else {
          // update with conventional pusher
          if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
            ux1(p) += HALF * dt * force_Cart[0];
            ux2(p) += HALF * dt * force_Cart[1];
            ux3(p) += HALF * dt * force_Cart[2];
          }
          velUpd(P {}, p, ei_Cart, bi_Cart);
          if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
            ux1(p) += HALF * dt * force_Cart[0];
            ux2(p) += HALF * dt * force_Cart[1];
            ux3(p) += HALF * dt * force_Cart[2];
          }
        }
      } else {
        /* conventional pusher mode ------------------------------------- */
        // update with conventional pusher
        if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
          ux1(p) += HALF * dt * force_Cart[0];
          ux2(p) += HALF * dt * force_Cart[1];
          ux3(p) += HALF * dt * force_Cart[2];
        }
        velUpd(P {}, p, ei_Cart, bi_Cart);
        if constexpr (is_contained<Extforce_t, Union_t<Args...>>::value) {
          ux1(p) += HALF * dt * force_Cart[0];
          ux2(p) += HALF * dt * force_Cart[1];
          ux3(p) += HALF * dt * force_Cart[2];
        }
      }
      // cooling
      if constexpr (is_contained<Synchrotron_t, Union_t<Args...>>::value) {
        if (!is_gca) {
          u_prime[0] = HALF * (u_prime[0] + ux1(p));
          u_prime[1] = HALF * (u_prime[1] + ux2(p));
          u_prime[2] = HALF * (u_prime[2] + ux3(p));
          synchrotronDrag(p, u_prime, ei_Cart_rad, bi_Cart_rad);
        }
      }
      // update position
      posUpd<Massive_t>(p, xp_Cd);
    }

    template <typename T>
    Inline void posUpd(index_t& p, coord_t<M::PrtlDim>& xp) const {

      // get cartesian velocity
      const real_t        inv_energy { ONE / getEnergy(T {}, p) };
      vec_t<Dim::_3D>     vp_Cart { ux1(p) * inv_energy,
                                ux2(p) * inv_energy,
                                ux3(p) * inv_energy };
      // get cartesian position
      coord_t<M::PrtlDim> xp_Cart { ZERO };
      metric.template convert_xyz<Crd::Cd, Crd::XYZ>(xp, xp_Cart);
      // update cartesian position
      for (auto d = 0u; d < M::PrtlDim; ++d) {
        xp_Cart[d] += vp_Cart[d] * dt;
      }
      // transform back to code
      metric.template convert_xyz<Crd::XYZ, Crd::Cd>(xp_Cart, xp);

      // update x1
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        i1_prev(p)  = i1(p);
        dx1_prev(p) = dx1(p);
        from_Xi_to_i_di(xp[0], i1(p), dx1(p));
      }

      // update x2 & phi
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        i2_prev(p)  = i2(p);
        dx2_prev(p) = dx2(p);
        from_Xi_to_i_di(xp[1], i2(p), dx2(p));
        if constexpr (D == Dim::_2D && M::PrtlDim == Dim::_3D) {
          phi(p) = xp[2];
        }
      }

      // update x3
      if constexpr (D == Dim::_3D) {
        i3_prev(p)  = i3(p);
        dx3_prev(p) = dx3(p);
        from_Xi_to_i_di(xp[2], i3(p), dx3(p));
      }
      boundaryConditions(p);
    }

    /**
     * @brief update particle velocities
     * @param P pusher algorithm
     * @param p, e0, b0 index & interpolated fields
     */
    Inline void velUpd(Boris_t,
                       index_t&         p,
                       vec_t<Dim::_3D>& e0,
                       vec_t<Dim::_3D>& b0) const {

      real_t COEFF { coeff };

      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;
      vec_t<Dim::_3D> u0 { ux1(p) + e0[0], ux2(p) + e0[1], ux3(p) + e0[2] };

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

      ux1(p) = u0[0];
      ux2(p) = u0[1];
      ux3(p) = u0[2];
    }

    Inline void velUpd(Vay_t, index_t& p, vec_t<Dim::_3D>& e0, vec_t<Dim::_3D>& b0) const {
      auto COEFF { coeff };
      e0[0] *= COEFF;
      e0[1] *= COEFF;
      e0[2] *= COEFF;

      b0[0] *= COEFF;
      b0[1] *= COEFF;
      b0[2] *= COEFF;

      COEFF = ONE / math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p)));

      vec_t<Dim::_3D> u1 {
        (ux1(p) + TWO * e0[0] +
         CROSS_x1(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF),
        (ux2(p) + TWO * e0[1] +
         CROSS_x2(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF),
        (ux3(p) + TWO * e0[2] +
         CROSS_x3(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) * COEFF)
      };
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

      ux1(p) = COEFF2 * (u1[0] +
                         COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                           (b0[0] * COEFF) +
                         u1[1] * b0[2] * COEFF - u1[2] * b0[1] * COEFF);
      ux2(p) = COEFF2 * (u1[1] +
                         COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                           (b0[1] * COEFF) +
                         u1[2] * b0[0] * COEFF - u1[0] * b0[2] * COEFF);
      ux3(p) = COEFF2 * (u1[2] +
                         COEFF * DOT(u1[0], u1[1], u1[2], b0[0], b0[1], b0[2]) *
                           (b0[2] * COEFF) +
                         u1[0] * b0[1] * COEFF - u1[1] * b0[0] * COEFF);
    }

    Inline void velUpd(GCA_t, index_t& p, vec_t<Dim::_3D>& e0, vec_t<Dim::_3D>& b0) const {
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
      auto upar { DOT(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) +
                  coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) };

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
      ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    Inline void velUpd(Union_t<GCA_t, Extforce_t>,
                       index_t&         p,
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
      auto upar { DOT(ux1(p), ux2(p), ux3(p), b0[0], b0[1], b0[2]) +
                  coeff * TWO * DOT(e0[0], e0[1], e0[2], b0[0], b0[1], b0[2]) +
                  dt * DOT(f0[0], f0[1], f0[2], b0[0], b0[1], b0[2]) };

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
      ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
      ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
      ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
    }

    // Getters
    Inline void getPrtlPos(index_t& p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xp[0] = i_di_to_Xi(i1(p), dx1(p));
      }
      if constexpr (D == Dim::_2D) {
        xp[1] = i_di_to_Xi(i2(p), dx2(p));
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp[2] = phi(p);
        }
      }
      if constexpr (D == Dim::_3D) {
        xp[1] = i_di_to_Xi(i2(p), dx2(p));
        xp[2] = i_di_to_Xi(i3(p), dx3(p));
      }
    }

    Inline auto getEnergy(Massive_t, index_t& p) const -> real_t {
      return math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)));
    }

    Inline auto getEnergy(Massless_t, index_t& p) const -> real_t {
      return math::sqrt(SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)));
    }

    Inline void getInterpFlds(index_t&         p,
                              vec_t<Dim::_3D>& e0,
                              vec_t<Dim::_3D>& b0) const {
      if constexpr (D == Dim::_1D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };

        // first order
        real_t c0, c1;

        // Ex1
        // interpolate to nodes
        c0    = HALF * (EB(i, em::ex1) + EB(i - 1, em::ex1));
        c1    = HALF * (EB(i, em::ex1) + EB(i + 1, em::ex1));
        // interpolate from nodes to the particle position
        e0[0] = c0 * (ONE - dx1_) + c1 * dx1_;
        // Ex2
        c0    = EB(i, em::ex2);
        c1    = EB(i + 1, em::ex2);
        e0[1] = c0 * (ONE - dx1_) + c1 * dx1_;
        // Ex3
        c0    = EB(i, em::ex3);
        c1    = EB(i + 1, em::ex3);
        e0[2] = c0 * (ONE - dx1_) + c1 * dx1_;

        // Bx1
        c0    = EB(i, em::bx1);
        c1    = EB(i + 1, em::bx1);
        b0[0] = c0 * (ONE - dx1_) + c1 * dx1_;
        // Bx2
        c0    = HALF * (EB(i - 1, em::bx2) + EB(i, em::bx2));
        c1    = HALF * (EB(i, em::bx2) + EB(i + 1, em::bx2));
        b0[1] = c0 * (ONE - dx1_) + c1 * dx1_;
        // Bx3
        c0    = HALF * (EB(i - 1, em::bx3) + EB(i, em::bx3));
        c1    = HALF * (EB(i, em::bx3) + EB(i + 1, em::bx3));
        b0[2] = c0 * (ONE - dx1_) + c1 * dx1_;
      } else if constexpr (D == Dim::_2D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };

        // first order
        real_t c000, c100, c010, c110, c00, c10;

        // Ex1
        // interpolate to nodes
        c000  = HALF * (EB(i, j, em::ex1) + EB(i - 1, j, em::ex1));
        c100  = HALF * (EB(i, j, em::ex1) + EB(i + 1, j, em::ex1));
        c010  = HALF * (EB(i, j + 1, em::ex1) + EB(i - 1, j + 1, em::ex1));
        c110  = HALF * (EB(i, j + 1, em::ex1) + EB(i + 1, j + 1, em::ex1));
        // interpolate from nodes to the particle position
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        e0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
        // Ex2
        c000  = HALF * (EB(i, j, em::ex2) + EB(i, j - 1, em::ex2));
        c100  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j - 1, em::ex2));
        c010  = HALF * (EB(i, j, em::ex2) + EB(i, j + 1, em::ex2));
        c110  = HALF * (EB(i + 1, j, em::ex2) + EB(i + 1, j + 1, em::ex2));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        e0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
        // Ex3
        c000  = EB(i, j, em::ex3);
        c100  = EB(i + 1, j, em::ex3);
        c010  = EB(i, j + 1, em::ex3);
        c110  = EB(i + 1, j + 1, em::ex3);
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        e0[2] = c00 * (ONE - dx2_) + c10 * dx2_;

        // Bx1
        c000  = HALF * (EB(i, j, em::bx1) + EB(i, j - 1, em::bx1));
        c100  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j - 1, em::bx1));
        c010  = HALF * (EB(i, j, em::bx1) + EB(i, j + 1, em::bx1));
        c110  = HALF * (EB(i + 1, j, em::bx1) + EB(i + 1, j + 1, em::bx1));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        b0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
        // Bx2
        c000  = HALF * (EB(i - 1, j, em::bx2) + EB(i, j, em::bx2));
        c100  = HALF * (EB(i, j, em::bx2) + EB(i + 1, j, em::bx2));
        c010  = HALF * (EB(i - 1, j + 1, em::bx2) + EB(i, j + 1, em::bx2));
        c110  = HALF * (EB(i, j + 1, em::bx2) + EB(i + 1, j + 1, em::bx2));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        b0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
        // Bx3
        c000  = INV_4 * (EB(i - 1, j - 1, em::bx3) + EB(i - 1, j, em::bx3) +
                        EB(i, j - 1, em::bx3) + EB(i, j, em::bx3));
        c100  = INV_4 * (EB(i, j - 1, em::bx3) + EB(i, j, em::bx3) +
                        EB(i + 1, j - 1, em::bx3) + EB(i + 1, j, em::bx3));
        c010  = INV_4 * (EB(i - 1, j, em::bx3) + EB(i - 1, j + 1, em::bx3) +
                        EB(i, j, em::bx3) + EB(i, j + 1, em::bx3));
        c110  = INV_4 * (EB(i, j, em::bx3) + EB(i, j + 1, em::bx3) +
                        EB(i + 1, j, em::bx3) + EB(i + 1, j + 1, em::bx3));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        b0[2] = c00 * (ONE - dx2_) + c10 * dx2_;
      } else if constexpr (D == Dim::_3D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const int  k { i3(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };
        const auto dx3_ { static_cast<real_t>(dx3(p)) };

        // first order
        real_t c000, c100, c010, c110, c001, c101, c011, c111, c00, c10, c01,
          c11, c0, c1;

        // Ex1
        // interpolate to nodes
        c000 = HALF * (EB(i, j, k, em::ex1) + EB(i - 1, j, k, em::ex1));
        c100 = HALF * (EB(i, j, k, em::ex1) + EB(i + 1, j, k, em::ex1));
        c010 = HALF * (EB(i, j + 1, k, em::ex1) + EB(i - 1, j + 1, k, em::ex1));
        c110 = HALF * (EB(i, j + 1, k, em::ex1) + EB(i + 1, j + 1, k, em::ex1));
        // interpolate from nodes to the particle position
        c00  = c000 * (ONE - dx1_) + c100 * dx1_;
        c10  = c010 * (ONE - dx1_) + c110 * dx1_;
        c0   = c00 * (ONE - dx2_) + c10 * dx2_;
        // interpolate to nodes
        c001 = HALF * (EB(i, j, k + 1, em::ex1) + EB(i - 1, j, k + 1, em::ex1));
        c101 = HALF * (EB(i, j, k + 1, em::ex1) + EB(i + 1, j, k + 1, em::ex1));
        c011 = HALF *
               (EB(i, j + 1, k + 1, em::ex1) + EB(i - 1, j + 1, k + 1, em::ex1));
        c111 = HALF *
               (EB(i, j + 1, k + 1, em::ex1) + EB(i + 1, j + 1, k + 1, em::ex1));
        // interpolate from nodes to the particle position
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        e0[0] = c0 * (ONE - dx3_) + c1 * dx3_;

        // Ex2
        c000 = HALF * (EB(i, j, k, em::ex2) + EB(i, j - 1, k, em::ex2));
        c100 = HALF * (EB(i + 1, j, k, em::ex2) + EB(i + 1, j - 1, k, em::ex2));
        c010 = HALF * (EB(i, j, k, em::ex2) + EB(i, j + 1, k, em::ex2));
        c110 = HALF * (EB(i + 1, j, k, em::ex2) + EB(i + 1, j + 1, k, em::ex2));
        c00  = c000 * (ONE - dx1_) + c100 * dx1_;
        c10  = c010 * (ONE - dx1_) + c110 * dx1_;
        c0   = c00 * (ONE - dx2_) + c10 * dx2_;
        c001 = HALF * (EB(i, j, k + 1, em::ex2) + EB(i, j - 1, k + 1, em::ex2));
        c101 = HALF *
               (EB(i + 1, j, k + 1, em::ex2) + EB(i + 1, j - 1, k + 1, em::ex2));
        c011 = HALF * (EB(i, j, k + 1, em::ex2) + EB(i, j + 1, k + 1, em::ex2));
        c111 = HALF *
               (EB(i + 1, j, k + 1, em::ex2) + EB(i + 1, j + 1, k + 1, em::ex2));
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        e0[1] = c0 * (ONE - dx3_) + c1 * dx3_;

        // Ex3
        c000 = HALF * (EB(i, j, k, em::ex3) + EB(i, j, k - 1, em::ex3));
        c100 = HALF * (EB(i + 1, j, k, em::ex3) + EB(i + 1, j, k - 1, em::ex3));
        c010 = HALF * (EB(i, j + 1, k, em::ex3) + EB(i, j + 1, k - 1, em::ex3));
        c110 = HALF *
               (EB(i + 1, j + 1, k, em::ex3) + EB(i + 1, j + 1, k - 1, em::ex3));
        c001 = HALF * (EB(i, j, k, em::ex3) + EB(i, j, k + 1, em::ex3));
        c101 = HALF * (EB(i + 1, j, k, em::ex3) + EB(i + 1, j, k + 1, em::ex3));
        c011 = HALF * (EB(i, j + 1, k, em::ex3) + EB(i, j + 1, k + 1, em::ex3));
        c111 = HALF *
               (EB(i + 1, j + 1, k, em::ex3) + EB(i + 1, j + 1, k + 1, em::ex3));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c0    = c00 * (ONE - dx2_) + c10 * dx2_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        e0[2] = c0 * (ONE - dx3_) + c1 * dx3_;

        // Bx1
        c000 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j - 1, k, em::bx1) +
                        EB(i, j, k - 1, em::bx1) + EB(i, j - 1, k - 1, em::bx1));
        c100 = INV_4 *
               (EB(i + 1, j, k, em::bx1) + EB(i + 1, j - 1, k, em::bx1) +
                EB(i + 1, j, k - 1, em::bx1) + EB(i + 1, j - 1, k - 1, em::bx1));
        c001 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j, k + 1, em::bx1) +
                        EB(i, j - 1, k, em::bx1) + EB(i, j - 1, k + 1, em::bx1));
        c101 = INV_4 *
               (EB(i + 1, j, k, em::bx1) + EB(i + 1, j, k + 1, em::bx1) +
                EB(i + 1, j - 1, k, em::bx1) + EB(i + 1, j - 1, k + 1, em::bx1));
        c010 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j + 1, k, em::bx1) +
                        EB(i, j, k - 1, em::bx1) + EB(i, j + 1, k - 1, em::bx1));
        c110 = INV_4 *
               (EB(i + 1, j, k, em::bx1) + EB(i + 1, j, k - 1, em::bx1) +
                EB(i + 1, j + 1, k - 1, em::bx1) + EB(i + 1, j + 1, k, em::bx1));
        c011 = INV_4 * (EB(i, j, k, em::bx1) + EB(i, j + 1, k, em::bx1) +
                        EB(i, j + 1, k + 1, em::bx1) + EB(i, j, k + 1, em::bx1));
        c111 = INV_4 *
               (EB(i + 1, j, k, em::bx1) + EB(i + 1, j + 1, k, em::bx1) +
                EB(i + 1, j + 1, k + 1, em::bx1) + EB(i + 1, j, k + 1, em::bx1));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c0    = c00 * (ONE - dx2_) + c10 * dx2_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        b0[0] = c0 * (ONE - dx3_) + c1 * dx3_;

        // Bx2
        c000 = INV_4 * (EB(i - 1, j, k - 1, em::bx2) + EB(i - 1, j, k, em::bx2) +
                        EB(i, j, k - 1, em::bx2) + EB(i, j, k, em::bx2));
        c100 = INV_4 * (EB(i, j, k - 1, em::bx2) + EB(i, j, k, em::bx2) +
                        EB(i + 1, j, k - 1, em::bx2) + EB(i + 1, j, k, em::bx2));
        c001 = INV_4 * (EB(i - 1, j, k, em::bx2) + EB(i - 1, j, k + 1, em::bx2) +
                        EB(i, j, k, em::bx2) + EB(i, j, k + 1, em::bx2));
        c101 = INV_4 * (EB(i, j, k, em::bx2) + EB(i, j, k + 1, em::bx2) +
                        EB(i + 1, j, k, em::bx2) + EB(i + 1, j, k + 1, em::bx2));
        c010 = INV_4 *
               (EB(i - 1, j + 1, k - 1, em::bx2) + EB(i - 1, j + 1, k, em::bx2) +
                EB(i, j + 1, k - 1, em::bx2) + EB(i, j + 1, k, em::bx2));
        c110 = INV_4 *
               (EB(i, j + 1, k - 1, em::bx2) + EB(i, j + 1, k, em::bx2) +
                EB(i + 1, j + 1, k - 1, em::bx2) + EB(i + 1, j + 1, k, em::bx2));
        c011 = INV_4 *
               (EB(i - 1, j + 1, k, em::bx2) + EB(i - 1, j + 1, k + 1, em::bx2) +
                EB(i, j + 1, k, em::bx2) + EB(i, j + 1, k + 1, em::bx2));
        c111 = INV_4 *
               (EB(i, j + 1, k, em::bx2) + EB(i, j + 1, k + 1, em::bx2) +
                EB(i + 1, j + 1, k, em::bx2) + EB(i + 1, j + 1, k + 1, em::bx2));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c0    = c00 * (ONE - dx2_) + c10 * dx2_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        b0[1] = c0 * (ONE - dx3_) + c1 * dx3_;

        // Bx3
        c000 = INV_4 * (EB(i - 1, j - 1, k, em::bx3) + EB(i - 1, j, k, em::bx3) +
                        EB(i, j - 1, k, em::bx3) + EB(i, j, k, em::bx3));
        c100 = INV_4 * (EB(i, j - 1, k, em::bx3) + EB(i, j, k, em::bx3) +
                        EB(i + 1, j - 1, k, em::bx3) + EB(i + 1, j, k, em::bx3));
        c001 = INV_4 *
               (EB(i - 1, j - 1, k + 1, em::bx3) + EB(i - 1, j, k + 1, em::bx3) +
                EB(i, j - 1, k + 1, em::bx3) + EB(i, j, k + 1, em::bx3));
        c101 = INV_4 *
               (EB(i, j - 1, k + 1, em::bx3) + EB(i, j, k + 1, em::bx3) +
                EB(i + 1, j - 1, k + 1, em::bx3) + EB(i + 1, j, k + 1, em::bx3));
        c010 = INV_4 * (EB(i - 1, j, k, em::bx3) + EB(i - 1, j + 1, k, em::bx3) +
                        EB(i, j, k, em::bx3) + EB(i, j + 1, k, em::bx3));
        c110 = INV_4 * (EB(i, j, k, em::bx3) + EB(i, j + 1, k, em::bx3) +
                        EB(i + 1, j, k, em::bx3) + EB(i + 1, j + 1, k, em::bx3));
        c011 = INV_4 *
               (EB(i - 1, j, k + 1, em::bx3) + EB(i - 1, j + 1, k + 1, em::bx3) +
                EB(i, j, k + 1, em::bx3) + EB(i, j + 1, k + 1, em::bx3));
        c111 = INV_4 *
               (EB(i, j, k + 1, em::bx3) + EB(i, j + 1, k + 1, em::bx3) +
                EB(i + 1, j, k + 1, em::bx3) + EB(i + 1, j + 1, k + 1, em::bx3));
        c00   = c000 * (ONE - dx1_) + c100 * dx1_;
        c01   = c001 * (ONE - dx1_) + c101 * dx1_;
        c10   = c010 * (ONE - dx1_) + c110 * dx1_;
        c11   = c011 * (ONE - dx1_) + c111 * dx1_;
        c0    = c00 * (ONE - dx2_) + c10 * dx2_;
        c1    = c01 * (ONE - dx2_) + c11 * dx2_;
        b0[2] = c0 * (ONE - dx3_) + c1 * dx3_;
      }
    }

    // Extra
    Inline void boundaryConditions(index_t& p) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        if (i1(p) < 0) {
          if (is_periodic_i1min) {
            i1(p)      += ni1;
            i1_prev(p) += ni1;
          } else if (is_absorb_i1min) {
            tag(p) = ParticleTag::dead;
          }
        } else if (i1(p) >= ni1) {
          if (is_periodic_i1max) {
            i1(p)      -= ni1;
            i1_prev(p) -= ni1;
          } else if (is_absorb_i1max) {
            tag(p) = ParticleTag::dead;
          }
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        if (i2(p) < 0) {
          if (is_periodic_i2min) {
            i2(p)      += ni2;
            i2_prev(p) += ni2;
          } else if (is_absorb_i2min) {
            tag(p) = ParticleTag::dead;
          } else if (is_axis_i2min) {
            i2(p)  = 0;
            dx2(p) = ONE - dx2(p);
          }
        } else if (i2(p) >= ni2) {
          if (is_periodic_i2max) {
            i2(p)      -= ni2;
            i2_prev(p) -= ni2;
          } else if (is_absorb_i2max) {
            tag(p) = ParticleTag::dead;
          } else if (is_axis_i2max) {
            i2(p)  = ni2 - 1;
            dx2(p) = ONE - dx2(p);
          }
        }
      }
      if constexpr (D == Dim::_3D) {
        if (i3(p) < 0) {
          if (is_periodic_i3min) {
            i3(p)      += ni3;
            i3_prev(p) += ni3;
          } else if (is_absorb_i3min) {
            tag(p) = ParticleTag::dead;
          }
        } else if (i3(p) >= ni3) {
          if (is_periodic_i3max) {
            i3(p)      -= ni3;
            i3_prev(p) -= ni3;
          } else if (is_absorb_i3max) {
            tag(p) = ParticleTag::dead;
          }
        }
      }
    }
  };

} // namespace kernel::sr

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // KERNELS_PARTICLE_PUSHER_SR_HPP

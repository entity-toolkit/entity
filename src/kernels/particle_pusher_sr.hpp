/**
 * @file kernels/particle_pusher_sr.h
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

#ifndef KERNELS_PARTICLE_PUSHER_SR_HPP
#define KERNELS_PARTICLE_PUSHER_SR_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

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

  namespace Cooling {
    enum CoolingTags_ {
      None        = 0,
      Synchrotron = 1 << 0,
    };
  } // namespace Cooling

  typedef int CoolingTags;

  struct NoForce_t {
    NoForce_t() {}
  };

  /**
   * @brief
   * A helper struct which combines the atmospheric gravity
   * with (optionally) custom user-defined force
   * @tparam D Dimension
   * @tparam C Coordinate system
   * @tparam F Additional force
   * @tparam Atm Toggle for atmospheric gravity
   * @note when `Atm` is true, `g` contains a vector of gravity acceleration
   * @note when `Atm` is true, sign of `ds` indicates the direction of the boundary
   * !TODO: compensate for the species mass when applying atmospheric force
   */
  template <Dimension D, Coord::type C, class F = NoForce_t, bool Atm = false>
  struct Force {
    static constexpr auto ExtForce = not std::is_same<F, NoForce_t>::value;
    static_assert(ExtForce or Atm,
                  "Force initialized with neither PGen force nor gravity");

    const F      pgen_force;
    const real_t gx1, gx2, gx3, x_surf, ds;

    Force(const F& pgen_force, const vec_t<Dim::_3D>& g, real_t x_surf, real_t ds)
      : pgen_force { pgen_force }
      , gx1 { g[0] }
      , gx2 { g[1] }
      , gx3 { g[2] }
      , x_surf { x_surf }
      , ds { ds } {}

    Force(const F& pgen_force)
      : Force {
        pgen_force,
        { ZERO, ZERO, ZERO },
        ZERO,
        ZERO
    } {
      raise::ErrorIf(Atm, "Atmospheric gravity not provided", HERE);
    }

    Force(const vec_t<Dim::_3D>& g, real_t x_surf, real_t ds)
      : Force { NoForce_t {}, g, x_surf, ds } {
      raise::ErrorIf(ExtForce, "External force not provided", HERE);
    }

    Inline auto fx1(const spidx_t&    sp,
                    const simtime_t&  time,
                    bool              ext_force,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t f_x1 = ZERO;
      if constexpr (ExtForce) {
        if (ext_force) {
          f_x1 += pgen_force.fx1(sp, time, x_Ph);
        }
      }
      if constexpr (Atm) {
        if (gx1 != ZERO) {
          if ((ds > ZERO and x_Ph[0] >= x_surf + ds) or
              (ds < ZERO and x_Ph[0] <= x_surf + ds)) {
            return f_x1;
          }
          if constexpr (C == Coord::Cart) {
            return f_x1 + gx1;
          } else {
            return f_x1 + gx1 * SQR(x_surf / x_Ph[0]);
          }
        }
      }
      return f_x1;
    }

    Inline auto fx2(const spidx_t&    sp,
                    const simtime_t&  time,
                    bool              ext_force,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t f_x2 = ZERO;
      if constexpr (ExtForce) {
        if (ext_force) {
          f_x2 += pgen_force.fx2(sp, time, x_Ph);
        }
      }
      if constexpr (Atm and (D == Dim::_2D or D == Dim::_3D)) {
        if (gx2 != ZERO) {
          if ((ds > ZERO and x_Ph[1] >= x_surf + ds) or
              (ds < ZERO and x_Ph[1] <= x_surf + ds)) {
            return f_x2;
          }
          if constexpr (C == Coord::Cart) {
            return f_x2 + gx2;
          } else {
            raise::KernelError(HERE, "Invalid force for coordinate system");
          }
        }
      }
      return f_x2;
    }

    Inline auto fx3(const spidx_t&    sp,
                    const simtime_t&  time,
                    bool              ext_force,
                    const coord_t<D>& x_Ph) const -> real_t {
      real_t f_x3 = ZERO;
      if constexpr (ExtForce) {
        if (ext_force) {
          f_x3 += pgen_force.fx3(sp, time, x_Ph);
        }
      }
      if constexpr (Atm and D == Dim::_3D) {
        if (gx3 != ZERO) {
          if ((ds > ZERO and x_Ph[2] >= x_surf + ds) or
              (ds < ZERO and x_Ph[2] <= x_surf + ds)) {
            return f_x3;
          }
          if constexpr (C == Coord::Cart) {
            return f_x3 + gx3;
          } else {
            raise::KernelError(HERE, "Invalid force for coordinate system");
          }
        }
      }
      return f_x3;
    }
  };

  /**
   * @tparam M Metric
   * @tparam F Additional force
   */
  template <class M, class F = NoForce_t>
  struct Pusher_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static constexpr auto D        = M::Dim;
    static constexpr auto ExtForce = not std::is_same<F, NoForce_t>::value;

  private:
    const PrtlPusher::type pusher;
    const bool             GCA;
    const bool             ext_force;
    const CoolingTags      cooling;

    const randacc_ndfield_t<D, 6> EB;
    const spidx_t                 sp;
    array_t<int*>                 i1, i2, i3;
    array_t<int*>                 i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*>            dx1, dx2, dx3;
    array_t<prtldx_t*>            dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>              ux1, ux2, ux3;
    array_t<real_t*>              phi;
    array_t<short*>               tag;
    const M                       metric;
    const F                       force;

    const real_t time, coeff, dt;
    const int    ni1, ni2, ni3;
    bool         is_absorb_i1min { false }, is_absorb_i1max { false };
    bool         is_absorb_i2min { false }, is_absorb_i2max { false };
    bool         is_absorb_i3min { false }, is_absorb_i3max { false };
    bool         is_periodic_i1min { false }, is_periodic_i1max { false };
    bool         is_periodic_i2min { false }, is_periodic_i2max { false };
    bool         is_periodic_i3min { false }, is_periodic_i3max { false };
    bool         is_reflect_i1min { false }, is_reflect_i1max { false };
    bool         is_reflect_i2min { false }, is_reflect_i2max { false };
    bool         is_reflect_i3min { false }, is_reflect_i3max { false };
    bool         is_axis_i2min { false }, is_axis_i2max { false };
    // gca parameters
    const real_t gca_larmor, gca_EovrB_sqr;
    // synchrotron cooling parameters
    const real_t coeff_sync;

  public:
    Pusher_kernel(const PrtlPusher::type&        pusher,
                  bool                           GCA,
                  bool                           ext_force,
                  CoolingTags                    cooling,
                  const randacc_ndfield_t<D, 6>& EB,
                  spidx_t                        sp,
                  array_t<int*>&                 i1,
                  array_t<int*>&                 i2,
                  array_t<int*>&                 i3,
                  array_t<int*>&                 i1_prev,
                  array_t<int*>&                 i2_prev,
                  array_t<int*>&                 i3_prev,
                  array_t<prtldx_t*>&            dx1,
                  array_t<prtldx_t*>&            dx2,
                  array_t<prtldx_t*>&            dx3,
                  array_t<prtldx_t*>&            dx1_prev,
                  array_t<prtldx_t*>&            dx2_prev,
                  array_t<prtldx_t*>&            dx3_prev,
                  array_t<real_t*>&              ux1,
                  array_t<real_t*>&              ux2,
                  array_t<real_t*>&              ux3,
                  array_t<real_t*>&              phi,
                  array_t<short*>&               tag,
                  const M&                       metric,
                  const F&                       force,
                  real_t                         time,
                  real_t                         coeff,
                  real_t                         dt,
                  int                            ni1,
                  int                            ni2,
                  int                            ni3,
                  const boundaries_t<PrtlBC>&    boundaries,
                  real_t                         gca_larmor_max,
                  real_t                         gca_eovrb_max,
                  real_t                         coeff_sync)
      : pusher { pusher }
      , GCA { GCA }
      , ext_force { ext_force }
      , cooling { cooling }
      , EB { EB }
      , sp { sp }
      , i1 { i1 }
      , i2 { i2 }
      , i3 { i3 }
      , i1_prev { i1_prev }
      , i2_prev { i2_prev }
      , i3_prev { i3_prev }
      , dx1 { dx1 }
      , dx2 { dx2 }
      , dx3 { dx3 }
      , dx1_prev { dx1_prev }
      , dx2_prev { dx2_prev }
      , dx3_prev { dx3_prev }
      , ux1 { ux1 }
      , ux2 { ux2 }
      , ux3 { ux3 }
      , phi { phi }
      , tag { tag }
      , metric { metric }
      , force { force }
      , time { time }
      , coeff { coeff }
      , dt { dt }
      , ni1 { ni1 }
      , ni2 { ni2 }
      , ni3 { ni3 }
      , gca_larmor { gca_larmor_max }
      , gca_EovrB_sqr { SQR(gca_eovrb_max) }
      , coeff_sync { coeff_sync } {
      raise::ErrorIf(boundaries.size() < 1, "boundaries defined incorrectly", HERE);
      is_absorb_i1min = (boundaries[0].first == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].first == PrtlBC::ABSORB);
      is_absorb_i1max = (boundaries[0].second == PrtlBC::ATMOSPHERE) ||
                        (boundaries[0].second == PrtlBC::ABSORB);
      is_periodic_i1min = (boundaries[0].first == PrtlBC::PERIODIC);
      is_periodic_i1max = (boundaries[0].second == PrtlBC::PERIODIC);
      is_reflect_i1min  = (boundaries[0].first == PrtlBC::REFLECT);
      is_reflect_i1max  = (boundaries[0].second == PrtlBC::REFLECT);
      if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_absorb_i2min = (boundaries[1].first == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].first == PrtlBC::ABSORB);
        is_absorb_i2max = (boundaries[1].second == PrtlBC::ATMOSPHERE) ||
                          (boundaries[1].second == PrtlBC::ABSORB);
        is_periodic_i2min = (boundaries[1].first == PrtlBC::PERIODIC);
        is_periodic_i2max = (boundaries[1].second == PrtlBC::PERIODIC);
        is_reflect_i2min  = (boundaries[1].first == PrtlBC::REFLECT);
        is_reflect_i2max  = (boundaries[1].second == PrtlBC::REFLECT);
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
        is_reflect_i3min  = (boundaries[2].first == PrtlBC::REFLECT);
        is_reflect_i3max  = (boundaries[2].second == PrtlBC::REFLECT);
      }
    }

    Pusher_kernel(const PrtlPusher::type&     pusher,
                  bool                        GCA,
                  bool                        ext_force,
                  CoolingTags                 cooling,
                  const ndfield_t<D, 6>&      EB,
                  spidx_t                     sp,
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
                  simtime_t                   time,
                  real_t                      coeff,
                  real_t                      dt,
                  int                         ni1,
                  int                         ni2,
                  int                         ni3,
                  const boundaries_t<PrtlBC>& boundaries,
                  real_t                      gca_larmor_max,
                  real_t                      gca_eovrb_max,
                  real_t                      coeff_sync)
      : Pusher_kernel(pusher,
                      GCA,
                      ext_force,
                      cooling,
                      EB,
                      sp,
                      i1,
                      i2,
                      i3,
                      i1_prev,
                      i2_prev,
                      i3_prev,
                      dx1,
                      dx2,
                      dx3,
                      dx1_prev,
                      dx2_prev,
                      dx3_prev,
                      ux1,
                      ux2,
                      ux3,
                      phi,
                      tag,
                      metric,
                      NoForce_t {},
                      time,
                      coeff,
                      dt,
                      ni1,
                      ni2,
                      ni3,
                      boundaries,
                      gca_larmor_max,
                      gca_eovrb_max,
                      coeff_sync) {}

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

    Inline void operator()(index_t p) const {
      if (tag(p) != ParticleTag::alive) {
        if (tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      coord_t<M::PrtlDim> xp_Cd { ZERO };
      getPrtlPos(p, xp_Cd);
      if (pusher == PrtlPusher::PHOTON) {
        posUpd(false, p, xp_Cd);
        return;
      }
      // update cartesian velocity
      vec_t<Dim::_3D> ei { ZERO }, bi { ZERO };
      vec_t<Dim::_3D> ei_Cart { ZERO }, bi_Cart { ZERO };
      vec_t<Dim::_3D> force_Cart { ZERO };
      vec_t<Dim::_3D> u_prime { ZERO };
      vec_t<Dim::_3D> ei_Cart_rad { ZERO }, bi_Cart_rad { ZERO };
      bool            is_gca { false };

      //getInterpFlds(p, ei, bi);
      //  ToDo: Better way to call this
      getInterpFlds2nd(p, ei, bi);

      metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, ei, ei_Cart);
      metric.template transform_xyz<Idx::U, Idx::XYZ>(xp_Cd, bi, bi_Cart);
      if (cooling != 0) {
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
      if constexpr (ExtForce) {
        coord_t<M::PrtlDim> xp_Ph { ZERO };
        xp_Ph[0] = metric.template convert<1, Crd::Cd, Crd::Ph>(xp_Cd[0]);
        if constexpr (M::PrtlDim == Dim::_2D or M::PrtlDim == Dim::_3D) {
          xp_Ph[1] = metric.template convert<2, Crd::Cd, Crd::Ph>(xp_Cd[1]);
        }
        if constexpr (M::PrtlDim == Dim::_3D) {
          xp_Ph[2] = metric.template convert<3, Crd::Cd, Crd::Ph>(xp_Cd[2]);
        }
        metric.template transform_xyz<Idx::T, Idx::XYZ>(
          xp_Cd,
          { force.fx1(sp, time, ext_force, xp_Ph),
            force.fx2(sp, time, ext_force, xp_Ph),
            force.fx3(sp, time, ext_force, xp_Ph) },
          force_Cart);
      }
      if (GCA) {
        /* hybrid GCA/conventional mode --------------------------------- */
        const auto E2 { NORM_SQR(ei_Cart[0], ei_Cart[1], ei_Cart[2]) };
        const auto B2 { NORM_SQR(bi_Cart[0], bi_Cart[1], bi_Cart[2]) };
        const auto rL { math::sqrt(ONE + NORM_SQR(ux1(p), ux2(p), ux3(p))) *
                        dt / (TWO * math::abs(coeff) * math::sqrt(B2)) };
        if (B2 > ZERO && rL < gca_larmor && (E2 / B2) < gca_EovrB_sqr) {
          is_gca = true;
          // update with GCA
          if constexpr (ExtForce) {
            velUpd(true, p, force_Cart, ei_Cart, bi_Cart);
          } else {
            velUpd(true, p, ei_Cart, bi_Cart);
          }
        } else {
          // update with conventional pusher
          if constexpr (ExtForce) {
            ux1(p) += HALF * dt * force_Cart[0];
            ux2(p) += HALF * dt * force_Cart[1];
            ux3(p) += HALF * dt * force_Cart[2];
          }
          velUpd(false, p, ei_Cart, bi_Cart);
          if constexpr (ExtForce) {
            ux1(p) += HALF * dt * force_Cart[0];
            ux2(p) += HALF * dt * force_Cart[1];
            ux3(p) += HALF * dt * force_Cart[2];
          }
        }
      } else {
        /* conventional pusher mode ------------------------------------- */
        // update with conventional pusher
        if constexpr (ExtForce) {
          ux1(p) += HALF * dt * force_Cart[0];
          ux2(p) += HALF * dt * force_Cart[1];
          ux3(p) += HALF * dt * force_Cart[2];
        }
        velUpd(false, p, ei_Cart, bi_Cart);
        if constexpr (ExtForce) {
          ux1(p) += HALF * dt * force_Cart[0];
          ux2(p) += HALF * dt * force_Cart[1];
          ux3(p) += HALF * dt * force_Cart[2];
        }
      }
      // cooling
      if (cooling & Cooling::Synchrotron) {
        if (!is_gca) {
          u_prime[0] = HALF * (u_prime[0] + ux1(p));
          u_prime[1] = HALF * (u_prime[1] + ux2(p));
          u_prime[2] = HALF * (u_prime[2] + ux3(p));
          synchrotronDrag(p, u_prime, ei_Cart_rad, bi_Cart_rad);
        }
      }
      // update position
      posUpd(true, p, xp_Cd);
    }

    Inline void posUpd(bool massive, index_t& p, coord_t<M::PrtlDim>& xp) const {
      // get cartesian velocity
      if constexpr (M::CoordType == Coord::Cart) {
        // i+di push for Cartesian basis
        const real_t dt_inv_energy {
          massive
            ? (dt / math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p))))
            : (dt / math::sqrt(SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p))))
        };
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          i1_prev(p)  = i1(p);
          dx1_prev(p) = dx1(p);
          dx1(p) += metric.template transform<1, Idx::XYZ, Idx::U>(xp, ux1(p)) *
                    dt_inv_energy;
          i1(p) += static_cast<int>(dx1(p) >= ONE) -
                   static_cast<int>(dx1(p) < ZERO);
          dx1(p) -= (dx1(p) >= ONE);
          dx1(p) += (dx1(p) < ZERO);
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          i2_prev(p)  = i2(p);
          dx2_prev(p) = dx2(p);
          dx2(p) += metric.template transform<2, Idx::XYZ, Idx::U>(xp, ux2(p)) *
                    dt_inv_energy;
          i2(p) += static_cast<int>(dx2(p) >= ONE) -
                   static_cast<int>(dx2(p) < ZERO);
          dx2(p) -= (dx2(p) >= ONE);
          dx2(p) += (dx2(p) < ZERO);
        }
        if constexpr (D == Dim::_3D) {
          i3_prev(p)  = i3(p);
          dx3_prev(p) = dx3(p);
          dx3(p) += metric.template transform<3, Idx::XYZ, Idx::U>(xp, ux3(p)) *
                    dt_inv_energy;
          i3(p) += static_cast<int>(dx3(p) >= ONE) -
                   static_cast<int>(dx3(p) < ZERO);
          dx3(p) -= (dx3(p) >= ONE);
          dx3(p) += (dx3(p) < ZERO);
        }
      } else {
        // full Cartesian coordinate push in non-Cartesian basis
        const real_t inv_energy {
          massive ? ONE / math::sqrt(ONE + SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)))
                  : ONE / math::sqrt(SQR(ux1(p)) + SQR(ux2(p)) + SQR(ux3(p)))
        };
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
      }
      boundaryConditions(p, xp);
    }

    /**
     * @brief update particle velocities
     * @param P pusher algorithm
     * @param p, e0, b0 index & interpolated fields
     */
    Inline void velUpd(bool             with_gca,
                       index_t&         p,
                       vec_t<Dim::_3D>& e0,
                       vec_t<Dim::_3D>& b0) const {
      if (with_gca) {
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
        const vec_t<Dim::_3D> vE_Cart { wE[0] * factor,
                                        wE[1] * factor,
                                        wE[2] * factor };
        const auto            Gamma { math::sqrt(ONE + SQR(upar)) /
                           math::sqrt(
                             ONE - NORM_SQR(vE_Cart[0], vE_Cart[1], vE_Cart[2])) };
        ux1(p) = upar * b0[0] + vE_Cart[0] * Gamma;
        ux2(p) = upar * b0[1] + vE_Cart[1] * Gamma;
        ux3(p) = upar * b0[2] + vE_Cart[2] * Gamma;
      } else if (pusher == PrtlPusher::BORIS) {
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
      } else if (pusher == PrtlPusher::VAY) {
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

        COEFF = ONE / math::sqrt(
                        INV_2 *
                        (COEFF2 + math::sqrt(SQR(COEFF2) +
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
    }

    Inline void velUpd(bool,
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

    Inline void getInterpFlds(index_t&         p,
                              vec_t<Dim::_3D>& e0,
                              vec_t<Dim::_3D>& b0) const {
      if constexpr (D == Dim::_1D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };

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
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };

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
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const int  k { i3(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };
        const auto dx3_ { static_cast<real_t>(dx3(p)) };

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
    }

    Inline void getInterpFlds2nd(index_t&         p,
                                 vec_t<Dim::_3D>& e0,
                                 vec_t<Dim::_3D>& b0) const {
      if constexpr (D == Dim::_1D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };

        // direct interpolation of staggered grid
        // primal = i+ind, dual = i
        const int indx = static_cast<int>(static_cast<real_t>(dx1_ + HALF));

        // Compute weights for second-order interpolation
        // primal
        const auto w0p = HALF * SQR(HALF - dx1_ + static_cast<real_t>(indx));
        const auto w1p = static_cast<real_t>(0.75) -
                          SQR(dx1_ - static_cast<real_t>(indx));
        const auto w2p = ONE - w0p - w1p;

        // dual
        const auto w0d = HALF * SQR(ONE - dx1_);
        const auto w2d = HALF * SQR(dx1_);
        const auto w1d = ONE - w0d - w2d;

        // Ex1 (dual grid)
        const auto ex1_0 = EB(i - 1, em::ex1);
        const auto ex1_1 = EB(i, em::ex1);
        const auto ex1_2 = EB(i + 1, em::ex1);
        e0[0]            = ex1_0 * w0d + ex1_1 * w1d + ex1_2 * w2d;

        // Ex2 (primal grid)
        const auto ex2_0 = EB(indx + i - 1, em::ex2);
        const auto ex2_1 = EB(indx + i, em::ex2);
        const auto ex2_2 = EB(indx + i + 1, em::ex2);
        e0[1]            = ex2_0 * w0p + ex2_1 * w1p + ex2_2 * w2p;

        // Ex3 (primal grid)
        const auto ex3_0 = EB(indx + i - 1, em::ex3);
        const auto ex3_1 = EB(indx + i, em::ex3);
        const auto ex3_2 = EB(indx + i + 1, em::ex3);
        e0[2]            = ex3_0 * w0p + ex3_1 * w1p + ex3_2 * w2p;

        // Bx1 (primal grid)
        const auto bx1_0 = EB(indx + i - 1, em::bx1);
        const auto bx1_1 = EB(indx + i, em::bx1);
        const auto bx1_2 = EB(indx + i + 1, em::bx1);
        b0[0]            = bx1_0 * w0p + bx1_1 * w1p + bx1_2 * w2p;

        // Bx2 (dual grid)
        const auto bx2_0 = EB(i - 1, em::bx2);
        const auto bx2_1 = EB(i, em::bx2);
        const auto bx2_2 = EB(i + 1, em::bx2);
        b0[1]            = bx2_0 * w0d + bx2_1 * w1d + bx2_2 * w2d;

        // Bx3 (dual grid)
        const auto bx3_0 = EB(i - 1, em::bx3);
        const auto bx3_1 = EB(i, em::bx3);
        const auto bx3_2 = EB(i + 1, em::bx3);
        b0[2]            = bx3_0 * w0d + bx3_1 * w1d + bx3_2 * w2d;

      } else if constexpr (D == Dim::_2D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };

        // direct interpolation of staggered grid
        // primal = i+ind, dual = i
        const int indx = static_cast<int>(static_cast<real_t>(dx1_ + HALF));
        const int indy = static_cast<int>(static_cast<real_t>(dx2_ + HALF));

        // Compute weights for second-order interpolation
        // primal
        const auto w0px = HALF * SQR(HALF - dx1_ + static_cast<real_t>(indx));
        const auto w1px = static_cast<real_t>(0.75) -
                          SQR(dx1_ - static_cast<real_t>(indx));
        const auto w2px = ONE - w0px - w1px;
        const auto w0py = HALF * SQR(HALF - dx2_ + static_cast<real_t>(indy));
        const auto w1py = static_cast<real_t>(0.75) -
                          SQR(dx2_ - static_cast<real_t>(indy));
        const auto w2py = ONE - w0py - w1py;

        // dual
        const auto w0dx = HALF * SQR(ONE - dx1_);
        const auto w2dx = HALF * SQR(dx1_);
        const auto w1dx = ONE - w0dx - w2dx;
        const auto w0dy = HALF * SQR(ONE - dx2_);
        const auto w2dy = HALF * SQR(dx2_);
        const auto w1dy = ONE - w0dy - w2dy;

        // Ex1
        // Interpolate --- (dual, primal)
        // clang-format off
        const auto ex1_000 = EB(i - 1, indy + j - 1, em::ex1);
        const auto ex1_100 = EB(i,     indy + j - 1, em::ex1);
        const auto ex1_200 = EB(i + 1, indy + j - 1, em::ex1);
        const auto ex1_010 = EB(i - 1, indy + j,     em::ex1);
        const auto ex1_110 = EB(i,     indy + j,     em::ex1);
        const auto ex1_210 = EB(i + 1, indy + j,     em::ex1);
        const auto ex1_020 = EB(i - 1, indy + j + 1, em::ex1);
        const auto ex1_120 = EB(i,     indy + j + 1, em::ex1);
        const auto ex1_220 = EB(i + 1, indy + j + 1, em::ex1);
        // clang-format on

        const auto ex1_0 = ex1_000 * w0dx + ex1_100 * w1dx + ex1_200 * w2dx;
        const auto ex1_1 = ex1_010 * w0dx + ex1_110 * w1dx + ex1_210 * w2dx;
        const auto ex1_2 = ex1_020 * w0dx + ex1_120 * w1dx + ex1_220 * w2dx;
        e0[0]            = ex1_0 * w0py + ex1_1 * w1py + ex1_2 * w2py;

        // Ex2
        // Interpolate --- (primal, dual)
        // clang-format off
        const auto ex2_000 = EB(indx + i - 1, j - 1, em::ex2);
        const auto ex2_100 = EB(indx + i,     j - 1, em::ex2);
        const auto ex2_200 = EB(indx + i + 1, j - 1, em::ex2);
        const auto ex2_010 = EB(indx + i - 1, j,     em::ex2);
        const auto ex2_110 = EB(indx + i,     j,     em::ex2);
        const auto ex2_210 = EB(indx + i + 1, j,     em::ex2);
        const auto ex2_020 = EB(indx + i - 1, j + 1, em::ex2);
        const auto ex2_120 = EB(indx + i,     j + 1, em::ex2);
        const auto ex2_220 = EB(indx + i + 1, j + 1, em::ex2);
        // clang-format on

        const auto ex2_0 = ex2_000 * w0px + ex2_100 * w1px + ex2_200 * w2px;
        const auto ex2_1 = ex2_010 * w0px + ex2_110 * w1px + ex2_210 * w2px;
        const auto ex2_2 = ex2_020 * w0px + ex2_120 * w1px + ex2_220 * w2px;
        e0[1]            = ex2_0 * w0dy + ex2_1 * w1dy + ex2_2 * w2dy;

        // Ex3
        // Interpolate --- (primal, primal)
        // clang-format off
        const auto ex3_000 = EB(indx + i - 1, indy + j - 1, em::ex3);
        const auto ex3_100 = EB(indx + i,     indy + j - 1, em::ex3);
        const auto ex3_200 = EB(indx + i + 1, indy + j - 1, em::ex3);
        const auto ex3_010 = EB(indx + i - 1, indy + j,     em::ex3);
        const auto ex3_110 = EB(indx + i,     indy + j,     em::ex3);
        const auto ex3_210 = EB(indx + i + 1, indy + j,     em::ex3);
        const auto ex3_020 = EB(indx + i - 1, indy + j + 1, em::ex3);
        const auto ex3_120 = EB(indx + i,     indy + j + 1, em::ex3);
        const auto ex3_220 = EB(indx + i + 1, indy + j + 1, em::ex3);
        // clang-format on

        const auto ex3_0 = ex3_000 * w0px + ex3_100 * w1px + ex3_200 * w2px;
        const auto ex3_1 = ex3_010 * w0px + ex3_110 * w1px + ex3_210 * w2px;
        const auto ex3_2 = ex3_020 * w0px + ex3_120 * w1px + ex3_220 * w2px;
        e0[2]            = ex3_0 * w0py + ex3_1 * w1py + ex3_2 * w2py;

        // Bx1
        // Interpolate --- (primal, dual)
        // clang-format off
        const auto bx1_000 = EB(indx + i - 1, j - 1, em::bx1);
        const auto bx1_100 = EB(indx + i,     j - 1, em::bx1);
        const auto bx1_200 = EB(indx + i + 1, j - 1, em::bx1);
        const auto bx1_010 = EB(indx + i - 1, j,     em::bx1);
        const auto bx1_110 = EB(indx + i,     j,     em::bx1);
        const auto bx1_210 = EB(indx + i + 1, j,     em::bx1);
        const auto bx1_020 = EB(indx + i - 1, j + 1, em::bx1);
        const auto bx1_120 = EB(indx + i,     j + 1, em::bx1);
        const auto bx1_220 = EB(indx + i + 1, j + 1, em::bx1);
        // clang-format on

        const auto bx1_0 = bx1_000 * w0px + bx1_100 * w1px + bx1_200 * w2px;
        const auto bx1_1 = bx1_010 * w0px + bx1_110 * w1px + bx1_210 * w2px;
        const auto bx1_2 = bx1_020 * w0px + bx1_120 * w1px + bx1_220 * w2px;
        b0[0]            = bx1_0 * w0dy + bx1_1 * w1dy + bx1_2 * w2dy;

        // Bx2
        // Interpolate --- (dual, primal)
        // clang-format off
        const auto bx2_000 = EB(i - 1, indy + j - 1, em::bx2);
        const auto bx2_100 = EB(i,     indy + j - 1, em::bx2);
        const auto bx2_200 = EB(i + 1, indy + j - 1, em::bx2);
        const auto bx2_010 = EB(i - 1, indy + j,     em::bx2);
        const auto bx2_110 = EB(i,     indy + j,     em::bx2);
        const auto bx2_210 = EB(i + 1, indy + j,     em::bx2);
        const auto bx2_020 = EB(i - 1, indy + j + 1, em::bx2);
        const auto bx2_120 = EB(i,     indy + j + 1, em::bx2);
        const auto bx2_220 = EB(i + 1, indy + j + 1, em::bx2);
        // clang-format on

        const auto bx2_0 = bx2_000 * w0dx + bx2_100 * w1dx + bx2_200 * w2dx;
        const auto bx2_1 = bx2_010 * w0dx + bx2_110 * w1dx + bx2_210 * w2dx;
        const auto bx2_2 = bx2_020 * w0dx + bx2_120 * w1dx + bx2_220 * w2dx;
        b0[1]            = bx2_0 * w0py + bx2_1 * w1py + bx2_2 * w2py;

        // Bx3
        // Interpolate --- (dual, dual)
        // clang-format off
        const auto bx3_000 = EB(i - 1, j - 1, em::bx3);
        const auto bx3_100 = EB(i,     j - 1, em::bx3);
        const auto bx3_200 = EB(i + 1, j - 1, em::bx3);
        const auto bx3_010 = EB(i - 1, j,     em::bx3);
        const auto bx3_110 = EB(i,     j,     em::bx3);
        const auto bx3_210 = EB(i + 1, j,     em::bx3);
        const auto bx3_020 = EB(i - 1, j + 1, em::bx3);
        const auto bx3_120 = EB(i,     j + 1, em::bx3);
        const auto bx3_220 = EB(i + 1, j + 1, em::bx3);
        // clang-format on

        const auto bx3_0 = bx3_000 * w0dx + bx3_100 * w1dx + bx3_200 * w2dx;
        const auto bx3_1 = bx3_010 * w0dx + bx3_110 * w1dx + bx3_210 * w2dx;
        const auto bx3_2 = bx3_020 * w0dx + bx3_120 * w1dx + bx3_220 * w2dx;
        b0[2]            = bx3_0 * w0dy + bx3_1 * w1dy + bx3_2 * w2dy;

      } else if constexpr (D == Dim::_3D) {
        const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
        const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
        const int  k { i3(p) + static_cast<int>(N_GHOSTS) };
        const auto dx1_ { static_cast<real_t>(dx1(p)) };
        const auto dx2_ { static_cast<real_t>(dx2(p)) };
        const auto dx3_ { static_cast<real_t>(dx3(p)) };

        // direct interpolation of staggered grid
        // primal = i+ind, dual = i
        const int indx = static_cast<int>(static_cast<real_t>(dx1_ + HALF));
        const int indy = static_cast<int>(static_cast<real_t>(dx2_ + HALF));
        const int indz = static_cast<int>(static_cast<real_t>(dx3_ + HALF));

        // Compute weights for second-order interpolation
        // primal
        const auto w0px = HALF * SQR(HALF - dx1_ + static_cast<real_t>(indx));
        const auto w1px = static_cast<real_t>(0.75) -
                          SQR(dx1_ - static_cast<real_t>(indx));
        const auto w2px = ONE - w0px - w1px;
        const auto w0py = HALF * SQR(HALF - dx2_ + static_cast<real_t>(indy));
        const auto w1py = static_cast<real_t>(0.75) -
                          SQR(dx2_ - static_cast<real_t>(indy));
        const auto w2py = ONE - w0py - w1py;
        const auto w0pz = HALF * SQR(HALF - dx3_ + static_cast<real_t>(indz));
        const auto w1pz = static_cast<real_t>(0.75) -
                          SQR(dx3_ - static_cast<real_t>(indz));
        const auto w2pz = ONE - w0pz - w1pz;

        // dual
        const auto w0dx = HALF * SQR(ONE - dx1_);
        const auto w2dx = HALF * SQR(dx1_);
        const auto w1dx = ONE - w0dx - w2dx;
        const auto w0dy = HALF * SQR(ONE - dx2_);
        const auto w2dy = HALF * SQR(dx2_);
        const auto w1dy = ONE - w0dy - w2dy;
        const auto w0dz = HALF * SQR(ONE - dx3_);
        const auto w2dz = HALF * SQR(dx3_);
        const auto w1dz = ONE - w0dz - w2dz;

        // Ex1
        // Interpolate --- (dual, primal, primal)
        // clang-format off
        const auto ex1_000 = EB(i - 1, indy + j - 1, indz + k - 1, em::ex1);
        const auto ex1_100 = EB(i,     indy + j - 1, indz + k - 1, em::ex1);
        const auto ex1_200 = EB(i + 1, indy + j - 1, indz + k - 1, em::ex1);
        const auto ex1_010 = EB(i - 1, indy + j,     indz + k - 1, em::ex1);
        const auto ex1_110 = EB(i,     indy + j,     indz + k - 1, em::ex1);
        const auto ex1_210 = EB(i + 1, indy + j,     indz + k - 1, em::ex1);
        const auto ex1_020 = EB(i - 1, indy + j + 1, indz + k - 1, em::ex1);
        const auto ex1_120 = EB(i,     indy + j + 1, indz + k - 1, em::ex1);
        const auto ex1_220 = EB(i + 1, indy + j + 1, indz + k - 1, em::ex1);
        const auto ex1_001 = EB(i - 1, indy + j - 1, indz + k,     em::ex1);
        const auto ex1_101 = EB(i,     indy + j - 1, indz + k,     em::ex1);
        const auto ex1_201 = EB(i + 1, indy + j - 1, indz + k,     em::ex1);
        const auto ex1_011 = EB(i - 1, indy + j,     indz + k,     em::ex1);
        const auto ex1_111 = EB(i,     indy + j,     indz + k,     em::ex1);
        const auto ex1_211 = EB(i + 1, indy + j,     indz + k,     em::ex1);
        const auto ex1_021 = EB(i - 1, indy + j + 1, indz + k,     em::ex1);
        const auto ex1_121 = EB(i,     indy + j + 1, indz + k,     em::ex1);
        const auto ex1_221 = EB(i + 1, indy + j + 1, indz + k,     em::ex1);
        const auto ex1_002 = EB(i - 1, indy + j - 1, indz + k + 1, em::ex1);
        const auto ex1_102 = EB(i,     indy + j - 1, indz + k + 1, em::ex1);
        const auto ex1_202 = EB(i + 1, indy + j - 1, indz + k + 1, em::ex1);
        const auto ex1_012 = EB(i - 1, indy + j,     indz + k + 1, em::ex1);
        const auto ex1_112 = EB(i,     indy + j,     indz + k + 1, em::ex1);
        const auto ex1_212 = EB(i + 1, indy + j,     indz + k + 1, em::ex1);
        const auto ex1_022 = EB(i - 1, indy + j + 1, indz + k + 1, em::ex1);
        const auto ex1_122 = EB(i,     indy + j + 1, indz + k + 1, em::ex1);
        const auto ex1_222 = EB(i + 1, indy + j + 1, indz + k + 1, em::ex1);
        // clang-format on

        const auto ex1_0_0 = ex1_000 * w0dx + ex1_100 * w1dx + ex1_200 * w2dx;
        const auto ex1_1_0 = ex1_010 * w0dx + ex1_110 * w1dx + ex1_210 * w2dx;
        const auto ex1_2_0 = ex1_020 * w0dx + ex1_120 * w1dx + ex1_220 * w2dx;
        const auto ex1_0_1 = ex1_001 * w0dx + ex1_101 * w1dx + ex1_201 * w2dx;
        const auto ex1_1_1 = ex1_011 * w0dx + ex1_111 * w1dx + ex1_211 * w2dx;
        const auto ex1_2_1 = ex1_021 * w0dx + ex1_121 * w1dx + ex1_221 * w2dx;
        const auto ex1_0_2 = ex1_002 * w0dx + ex1_102 * w1dx + ex1_202 * w2dx;
        const auto ex1_1_2 = ex1_012 * w0dx + ex1_112 * w1dx + ex1_212 * w2dx;
        const auto ex1_2_2 = ex1_022 * w0dx + ex1_122 * w1dx + ex1_222 * w2dx;

        const auto ex1_00 = ex1_0_0 * w0py + ex1_1_0 * w1py + ex1_2_0 * w2py;
        const auto ex1_01 = ex1_0_1 * w0py + ex1_1_1 * w1py + ex1_2_1 * w2py;
        const auto ex1_02 = ex1_0_2 * w0py + ex1_1_2 * w1py + ex1_2_2 * w2py;

        e0[0] = ex1_00 * w0pz + ex1_01 * w1pz + ex1_02 * w2pz;

        // Ex2
        // Interpolate -- (primal, dual, primal)
        // clang-format off
        const auto ex2_000 = EB(indx + i - 1, j - 1, indz + k - 1, em::ex2);
        const auto ex2_100 = EB(indx + i,     j - 1, indz + k - 1, em::ex2);
        const auto ex2_200 = EB(indx + i + 1, j - 1, indz + k - 1, em::ex2);
        const auto ex2_010 = EB(indx + i - 1, j,     indz + k - 1, em::ex2);
        const auto ex2_110 = EB(indx + i,     j,     indz + k - 1, em::ex2);
        const auto ex2_210 = EB(indx + i + 1, j,     indz + k - 1, em::ex2);
        const auto ex2_020 = EB(indx + i - 1, j + 1, indz + k - 1, em::ex2);
        const auto ex2_120 = EB(indx + i,     j + 1, indz + k - 1, em::ex2);
        const auto ex2_220 = EB(indx + i + 1, j + 1, indz + k - 1, em::ex2);
        const auto ex2_001 = EB(indx + i - 1, j - 1, indz + k,     em::ex2);
        const auto ex2_101 = EB(indx + i,     j - 1, indz + k,     em::ex2);
        const auto ex2_201 = EB(indx + i + 1, j - 1, indz + k,     em::ex2);
        const auto ex2_011 = EB(indx + i - 1, j,     indz + k,     em::ex2);
        const auto ex2_111 = EB(indx + i,     j,     indz + k,     em::ex2);
        const auto ex2_211 = EB(indx + i + 1, j,     indz + k,     em::ex2);
        const auto ex2_021 = EB(indx + i - 1, j + 1, indz + k,     em::ex2);
        const auto ex2_121 = EB(indx + i,     j + 1, indz + k,     em::ex2);
        const auto ex2_221 = EB(indx + i + 1, j + 1, indz + k,     em::ex2);
        const auto ex2_002 = EB(indx + i - 1, j - 1, indz + k + 1, em::ex2);
        const auto ex2_102 = EB(indx + i,     j - 1, indz + k + 1, em::ex2);
        const auto ex2_202 = EB(indx + i + 1, j - 1, indz + k + 1, em::ex2);
        const auto ex2_012 = EB(indx + i - 1, j,     indz + k + 1, em::ex2);
        const auto ex2_112 = EB(indx + i,     j,     indz + k + 1, em::ex2);
        const auto ex2_212 = EB(indx + i + 1, j,     indz + k + 1, em::ex2);
        const auto ex2_022 = EB(indx + i - 1, j + 1, indz + k + 1, em::ex2);
        const auto ex2_122 = EB(indx + i,     j + 1, indz + k + 1, em::ex2);
        const auto ex2_222 = EB(indx + i + 1, j + 1, indz + k + 1, em::ex2);
        // clang-format on

        const auto ex2_0_0 = ex2_000 * w0px + ex2_100 * w1px + ex1_200 * w2px;
        const auto ex2_1_0 = ex2_010 * w0px + ex2_110 * w1px + ex1_210 * w2px;
        const auto ex2_2_0 = ex2_020 * w0px + ex2_120 * w1px + ex1_220 * w2px;
        const auto ex2_0_1 = ex2_001 * w0px + ex2_101 * w1px + ex2_201 * w2px;
        const auto ex2_1_1 = ex2_011 * w0px + ex2_111 * w1px + ex2_211 * w2px;
        const auto ex2_2_1 = ex2_021 * w0px + ex2_121 * w1px + ex2_221 * w2px;
        const auto ex2_0_2 = ex2_002 * w0px + ex2_102 * w1px + ex2_202 * w2px;
        const auto ex2_1_2 = ex2_012 * w0px + ex2_112 * w1px + ex2_212 * w2px;
        const auto ex2_2_2 = ex2_022 * w0px + ex2_122 * w1px + ex2_222 * w2px;

        const auto ex2_00 = ex2_0_0 * w0dy + ex2_1_0 * w1dy + ex2_2_0 * w2dy;
        const auto ex2_01 = ex2_0_1 * w0dy + ex2_1_1 * w1dy + ex2_2_1 * w2dy;
        const auto ex2_02 = ex2_0_2 * w0dy + ex2_1_2 * w1dy + ex2_2_2 * w2dy;

        e0[1] = ex2_00 * w0pz + ex2_01 * w1pz + ex2_02 * w2pz;

        // Ex3
        // Interpolate -- (primal, primal, dual)
        // clang-format off
        const auto ex3_000 = EB(indx + i - 1, indy + j - 1, k - 1, em::ex3);
        const auto ex3_100 = EB(indx + i,     indy + j - 1, k - 1, em::ex3);
        const auto ex3_200 = EB(indx + i + 1, indy + j - 1, k - 1, em::ex3);
        const auto ex3_010 = EB(indx + i - 1, indy + j,     k - 1, em::ex3);
        const auto ex3_110 = EB(indx + i,     indy + j,     k - 1, em::ex3);
        const auto ex3_210 = EB(indx + i + 1, indy + j,     k - 1, em::ex3);
        const auto ex3_020 = EB(indx + i - 1, indy + j + 1, k - 1, em::ex3);
        const auto ex3_120 = EB(indx + i,     indy + j + 1, k - 1, em::ex3);
        const auto ex3_220 = EB(indx + i + 1, indy + j + 1, k - 1, em::ex3);
        const auto ex3_001 = EB(indx + i - 1, indy + j - 1, k,     em::ex3);
        const auto ex3_101 = EB(indx + i,     indy + j - 1, k,     em::ex3);
        const auto ex3_201 = EB(indx + i + 1, indy + j - 1, k,     em::ex3);
        const auto ex3_011 = EB(indx + i - 1, indy + j,     k,     em::ex3);
        const auto ex3_111 = EB(indx + i,     indy + j,     k,     em::ex3);
        const auto ex3_211 = EB(indx + i + 1, indy + j,     k,     em::ex3);
        const auto ex3_021 = EB(indx + i - 1, indy + j + 1, k,     em::ex3);
        const auto ex3_121 = EB(indx + i,     indy + j + 1, k,     em::ex3);
        const auto ex3_221 = EB(indx + i + 1, indy + j + 1, k,     em::ex3);
        const auto ex3_002 = EB(indx + i - 1, indy + j - 1, k + 1, em::ex3);
        const auto ex3_102 = EB(indx + i,     indy + j - 1, k + 1, em::ex3);
        const auto ex3_202 = EB(indx + i + 1, indy + j - 1, k + 1, em::ex3);
        const auto ex3_012 = EB(indx + i - 1, indy + j,     k + 1, em::ex3);
        const auto ex3_112 = EB(indx + i,     indy + j,     k + 1, em::ex3);
        const auto ex3_212 = EB(indx + i + 1, indy + j,     k + 1, em::ex3);
        const auto ex3_022 = EB(indx + i - 1, indy + j + 1, k + 1, em::ex3);
        const auto ex3_122 = EB(indx + i,     indy + j + 1, k + 1, em::ex3);
        const auto ex3_222 = EB(indx + i + 1, indy + j + 1, k + 1, em::ex3);
        // clang-format on

        const auto ex3_0_0 = ex3_000 * w0px + ex3_100 * w1px + ex3_200 * w2px;
        const auto ex3_1_0 = ex3_010 * w0px + ex3_110 * w1px + ex3_210 * w2px;
        const auto ex3_2_0 = ex3_020 * w0px + ex3_120 * w1px + ex3_220 * w2px;
        const auto ex3_0_1 = ex3_001 * w0px + ex3_101 * w1px + ex3_201 * w2px;
        const auto ex3_1_1 = ex3_011 * w0px + ex3_111 * w1px + ex3_211 * w2px;
        const auto ex3_2_1 = ex3_021 * w0px + ex3_121 * w1px + ex3_221 * w2px;
        const auto ex3_0_2 = ex3_002 * w0px + ex3_102 * w1px + ex3_202 * w2px;
        const auto ex3_1_2 = ex3_012 * w0px + ex3_112 * w1px + ex3_212 * w2px;
        const auto ex3_2_2 = ex3_022 * w0px + ex3_122 * w1px + ex3_222 * w2px;

        const auto ex3_00 = ex3_0_0 * w0py + ex3_1_0 * w1py + ex3_2_0 * w2py;
        const auto ex3_01 = ex3_0_1 * w0py + ex3_1_1 * w1py + ex3_2_1 * w2py;
        const auto ex3_02 = ex3_0_2 * w0py + ex3_1_2 * w1py + ex3_2_2 * w2py;

        e0[2] = ex3_00 * w0dz + ex3_01 * w1dz + ex3_02 * w2dz;

        // Bx1
        // Interpolate -- (primal, dual, dual)
        // clang-format off
        const auto bx1_000 = EB(indx + i - 1, j - 1, k - 1, em::bx1);
        const auto bx1_100 = EB(indx + i,     j - 1, k - 1, em::bx1);
        const auto bx1_200 = EB(indx + i + 1, j - 1, k - 1, em::bx1);
        const auto bx1_010 = EB(indx + i - 1, j,     k - 1, em::bx1);
        const auto bx1_110 = EB(indx + i,     j,     k - 1, em::bx1);
        const auto bx1_210 = EB(indx + i + 1, j,     k - 1, em::bx1);
        const auto bx1_020 = EB(indx + i - 1, j + 1, k - 1, em::bx1);
        const auto bx1_120 = EB(indx + i,     j + 1, k - 1, em::bx1);
        const auto bx1_220 = EB(indx + i + 1, j + 1, k - 1, em::bx1);
        const auto bx1_001 = EB(indx + i - 1, j - 1, k,     em::bx1);
        const auto bx1_101 = EB(indx + i,     j - 1, k,     em::bx1);
        const auto bx1_201 = EB(indx + i + 1, j - 1, k,     em::bx1);
        const auto bx1_011 = EB(indx + i - 1, j,     k,     em::bx1);
        const auto bx1_111 = EB(indx + i,     j,     k,     em::bx1);
        const auto bx1_211 = EB(indx + i + 1, j,     k,     em::bx1);
        const auto bx1_021 = EB(indx + i - 1, j + 1, k,     em::bx1);
        const auto bx1_121 = EB(indx + i,     j + 1, k,     em::bx1);
        const auto bx1_221 = EB(indx + i + 1, j + 1, k,     em::bx1);
        const auto bx1_002 = EB(indx + i - 1, j - 1, k + 1, em::bx1);
        const auto bx1_102 = EB(indx + i,     j - 1, k + 1, em::bx1);
        const auto bx1_202 = EB(indx + i + 1, j - 1, k + 1, em::bx1);
        const auto bx1_012 = EB(indx + i - 1, j,     k + 1, em::bx1);
        const auto bx1_112 = EB(indx + i,     j,     k + 1, em::bx1);
        const auto bx1_212 = EB(indx + i + 1, j,     k + 1, em::bx1);
        const auto bx1_022 = EB(indx + i - 1, j + 1, k + 1, em::bx1);
        const auto bx1_122 = EB(indx + i,     j + 1, k + 1, em::bx1);
        const auto bx1_222 = EB(indx + i + 1, j + 1, k + 1, em::bx1);
        // clang-format on

        const auto bx1_0_0 = bx1_000 * w0px + bx1_100 * w1px + bx1_200 * w2px;
        const auto bx1_1_0 = bx1_010 * w0px + bx1_110 * w1px + bx1_210 * w2px;
        const auto bx1_2_0 = bx1_020 * w0px + bx1_120 * w1px + bx1_220 * w2px;
        const auto bx1_0_1 = bx1_001 * w0px + bx1_101 * w1px + bx1_201 * w2px;
        const auto bx1_1_1 = bx1_011 * w0px + bx1_111 * w1px + bx1_211 * w2px;
        const auto bx1_2_1 = bx1_021 * w0px + bx1_121 * w1px + bx1_221 * w2px;
        const auto bx1_0_2 = bx1_002 * w0px + bx1_102 * w1px + bx1_202 * w2px;
        const auto bx1_1_2 = bx1_012 * w0px + bx1_112 * w1px + bx1_212 * w2px;
        const auto bx1_2_2 = bx1_022 * w0px + bx1_122 * w1px + bx1_222 * w2px;

        const auto bx1_00 = bx1_0_0 * w0dy + bx1_1_0 * w1dy + bx1_2_0 * w2dy;
        const auto bx1_01 = bx1_0_1 * w0dy + bx1_1_1 * w1dy + bx1_2_1 * w2dy;
        const auto bx1_02 = bx1_0_2 * w0dy + bx1_1_2 * w1dy + bx1_2_2 * w2dy;

        b0[0] = bx1_00 * w0dz + bx1_01 * w1dz + bx1_02 * w2dz;

        // Bx2
        // Interpolate -- (dual, primal, dual)
        // clang-format off
        const auto bx2_000 = EB(i - 1, indy + j - 1, k - 1, em::bx2);
        const auto bx2_100 = EB(i,     indy + j - 1, k - 1, em::bx2);
        const auto bx2_200 = EB(i + 1, indy + j - 1, k - 1, em::bx2);
        const auto bx2_010 = EB(i - 1, indy + j,     k - 1, em::bx2);
        const auto bx2_110 = EB(i,     indy + j,     k - 1, em::bx2);
        const auto bx2_210 = EB(i + 1, indy + j,     k - 1, em::bx2);
        const auto bx2_020 = EB(i - 1, indy + j + 1, k - 1, em::bx2);
        const auto bx2_120 = EB(i,     indy + j + 1, k - 1, em::bx2);
        const auto bx2_220 = EB(i + 1, indy + j + 1, k - 1, em::bx2);
        const auto bx2_001 = EB(i - 1, indy + j - 1, k,     em::bx2);
        const auto bx2_101 = EB(i,     indy + j - 1, k,     em::bx2);
        const auto bx2_201 = EB(i + 1, indy + j - 1, k,     em::bx2);
        const auto bx2_011 = EB(i - 1, indy + j,     k,     em::bx2);
        const auto bx2_111 = EB(i,     indy + j,     k,     em::bx2);
        const auto bx2_211 = EB(i + 1, indy + j,     k,     em::bx2);
        const auto bx2_021 = EB(i - 1, indy + j + 1, k,     em::bx2);
        const auto bx2_121 = EB(i,     indy + j + 1, k,     em::bx2);
        const auto bx2_221 = EB(i + 1, indy + j + 1, k,     em::bx2);
        const auto bx2_002 = EB(i - 1, indy + j - 1, k + 1, em::bx2);
        const auto bx2_102 = EB(i,     indy + j - 1, k + 1, em::bx2);
        const auto bx2_202 = EB(i + 1, indy + j - 1, k + 1, em::bx2);
        const auto bx2_012 = EB(i - 1, indy + j,     k + 1, em::bx2);
        const auto bx2_112 = EB(i,     indy + j,     k + 1, em::bx2);
        const auto bx2_212 = EB(i + 1, indy + j,     k + 1, em::bx2);
        const auto bx2_022 = EB(i - 1, indy + j + 1, k + 1, em::bx2);
        const auto bx2_122 = EB(i,     indy + j + 1, k + 1, em::bx2);
        const auto bx2_222 = EB(i + 1, indy + j + 1, k + 1, em::bx2);
        // clang-format on

        const auto bx2_0_0 = bx2_000 * w0dx + bx2_100 * w1dx + bx2_200 * w2dx;
        const auto bx2_1_0 = bx2_010 * w0dx + bx2_110 * w1dx + bx2_210 * w2dx;
        const auto bx2_2_0 = bx2_020 * w0dx + bx2_120 * w1dx + bx2_220 * w2dx;
        const auto bx2_0_1 = bx2_001 * w0dx + bx2_101 * w1dx + bx2_201 * w2dx;
        const auto bx2_1_1 = bx2_011 * w0dx + bx2_111 * w1dx + bx2_211 * w2dx;
        const auto bx2_2_1 = bx2_021 * w0dx + bx2_121 * w1dx + bx2_221 * w2dx;
        const auto bx2_0_2 = bx2_002 * w0dx + bx2_102 * w1dx + bx2_202 * w2dx;
        const auto bx2_1_2 = bx2_012 * w0dx + bx2_112 * w1dx + bx2_212 * w2dx;
        const auto bx2_2_2 = bx2_022 * w0dx + bx2_122 * w1dx + bx2_222 * w2dx;

        const auto bx2_00 = bx2_0_0 * w0py + bx2_1_0 * w1py + bx2_2_0 * w2py;
        const auto bx2_01 = bx2_0_1 * w0py + bx2_1_1 * w1py + bx2_2_1 * w2py;
        const auto bx2_02 = bx2_0_2 * w0py + bx2_1_2 * w1py + bx2_2_2 * w2py;

        b0[1] = bx2_00 * w0dz + bx2_01 * w1dz + bx2_02 * w2dz;

        // Bx3
        // Interpolate -- (dual, dual, primal)
        // clang-format off
        const auto bx3_000 = EB(i - 1, j - 1, indz + k - 1, em::bx3);
        const auto bx3_100 = EB(i,     j - 1, indz + k - 1, em::bx3);
        const auto bx3_200 = EB(i + 1, j - 1, indz + k - 1, em::bx3);
        const auto bx3_010 = EB(i - 1, j,     indz + k - 1, em::bx3);
        const auto bx3_110 = EB(i,     j,     indz + k - 1, em::bx3);
        const auto bx3_210 = EB(i + 1, j,     indz + k - 1, em::bx3);
        const auto bx3_020 = EB(i - 1, j + 1, indz + k - 1, em::bx3);
        const auto bx3_120 = EB(i,     j + 1, indz + k - 1, em::bx3);
        const auto bx3_220 = EB(i + 1, j + 1, indz + k - 1, em::bx3);
        const auto bx3_001 = EB(i - 1, j - 1, indz + k,     em::bx3);
        const auto bx3_101 = EB(i,     j - 1, indz + k,     em::bx3);
        const auto bx3_201 = EB(i + 1, j - 1, indz + k,     em::bx3);
        const auto bx3_011 = EB(i - 1, j,     indz + k,     em::bx3);
        const auto bx3_111 = EB(i,     j,     indz + k,     em::bx3);
        const auto bx3_211 = EB(i + 1, j,     indz + k,     em::bx3);
        const auto bx3_021 = EB(i - 1, j + 1, indz + k,     em::bx3);
        const auto bx3_121 = EB(i,     j + 1, indz + k,     em::bx3);
        const auto bx3_221 = EB(i + 1, j + 1, indz + k,     em::bx3);
        const auto bx3_002 = EB(i - 1, j - 1, indz + k + 1, em::bx3);
        const auto bx3_102 = EB(i,     j - 1, indz + k + 1, em::bx3);
        const auto bx3_202 = EB(i + 1, j - 1, indz + k + 1, em::bx3);
        const auto bx3_012 = EB(i - 1, j,     indz + k + 1, em::bx3);
        const auto bx3_112 = EB(i,     j,     indz + k + 1, em::bx3);
        const auto bx3_212 = EB(i + 1, j,     indz + k + 1, em::bx3);
        const auto bx3_022 = EB(i - 1, j + 1, indz + k + 1, em::bx3);
        const auto bx3_122 = EB(i,     j + 1, indz + k + 1, em::bx3);
        const auto bx3_222 = EB(i + 1, j + 1, indz + k + 1, em::bx3);
        // clang-format on

        const auto bx3_0_0 = bx3_000 * w0dx + bx3_100 * w1dx + bx3_200 * w2dx;
        const auto bx3_1_0 = bx3_010 * w0dx + bx3_110 * w1dx + bx3_210 * w2dx;
        const auto bx3_2_0 = bx3_020 * w0dx + bx3_120 * w1dx + bx3_220 * w2dx;
        const auto bx3_0_1 = bx3_001 * w0dx + bx3_101 * w1dx + bx3_201 * w2dx;
        const auto bx3_1_1 = bx3_011 * w0dx + bx3_111 * w1dx + bx3_211 * w2dx;
        const auto bx3_2_1 = bx3_021 * w0dx + bx3_121 * w1dx + bx3_221 * w2dx;
        const auto bx3_0_2 = bx3_002 * w0dx + bx3_102 * w1dx + bx3_202 * w2dx;
        const auto bx3_1_2 = bx3_012 * w0dx + bx3_112 * w1dx + bx3_212 * w2dx;
        const auto bx3_2_2 = bx3_022 * w0dx + bx3_122 * w1dx + bx3_222 * w2dx;

        const auto bx3_00 = bx3_0_0 * w0dy + bx3_1_0 * w1dy + bx3_2_0 * w2dy;
        const auto bx3_01 = bx3_0_1 * w0dy + bx3_1_1 * w1dy + bx3_2_1 * w2dy;
        const auto bx3_02 = bx3_0_2 * w0dy + bx3_1_2 * w1dy + bx3_2_2 * w2dy;

        b0[2] = bx3_00 * w0pz + bx3_01 * w1pz + bx3_02 * w2pz;
      }
    }

    // Extra
    Inline void boundaryConditions(index_t& p, coord_t<M::PrtlDim>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (i1(p) < 0) {
          if (is_periodic_i1min) {
            i1(p)      += ni1;
            i1_prev(p) += ni1;
          } else if (is_absorb_i1min) {
            tag(p) = ParticleTag::dead;
          } else if (is_reflect_i1min) {
            i1(p)      = 0;
            dx1(p)     = ONE - dx1(p);
            invert_vel = true;
          }
        } else if (i1(p) >= ni1) {
          if (is_periodic_i1max) {
            i1(p)      -= ni1;
            i1_prev(p) -= ni1;
          } else if (is_absorb_i1max) {
            tag(p) = ParticleTag::dead;
          } else if (is_reflect_i1max) {
            i1(p)      = ni1 - 1;
            dx1(p)     = ONE - dx1(p);
            invert_vel = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            ux1(p) = -ux1(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { ux1(p), ux2(p), ux3(p) },
              v);
            v[0] = -v[0];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            ux1(p) = vXYZ[0];
            ux2(p) = vXYZ[1];
            ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        auto invert_vel = false;
        if (i2(p) < 0) {
          if (is_periodic_i2min) {
            i2(p)      += ni2;
            i2_prev(p) += ni2;
          } else if (is_absorb_i2min) {
            tag(p) = ParticleTag::dead;
          } else if (is_reflect_i2min) {
            i2(p)      = 0;
            dx2(p)     = ONE - dx2(p);
            invert_vel = true;
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
          } else if (is_reflect_i2max) {
            i2(p)      = ni2 - 1;
            dx2(p)     = ONE - dx2(p);
            invert_vel = true;
          } else if (is_axis_i2max) {
            i2(p)  = ni2 - 1;
            dx2(p) = ONE - dx2(p);
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            ux2(p) = -ux2(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { ux1(p), ux2(p), ux3(p) },
              v);
            v[1] = -v[1];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            ux1(p) = vXYZ[0];
            ux2(p) = vXYZ[1];
            ux3(p) = vXYZ[2];
          }
        }
      }
      if constexpr (D == Dim::_3D) {
        auto invert_vel = false;
        if (i3(p) < 0) {
          if (is_periodic_i3min) {
            i3(p)      += ni3;
            i3_prev(p) += ni3;
          } else if (is_absorb_i3min) {
            tag(p) = ParticleTag::dead;
          } else if (is_reflect_i3min) {
            i3(p)      = 0;
            dx3(p)     = ONE - dx3(p);
            invert_vel = true;
          }
        } else if (i3(p) >= ni3) {
          if (is_periodic_i3max) {
            i3(p)      -= ni3;
            i3_prev(p) -= ni3;
          } else if (is_absorb_i3max) {
            tag(p) = ParticleTag::dead;
          } else if (is_reflect_i3max) {
            i3(p)      = ni3 - 1;
            dx3(p)     = ONE - dx3(p);
            invert_vel = true;
          }
        }
        if (invert_vel) {
          if constexpr (M::CoordType == Coord::Cart) {
            ux3(p) = -ux3(p);
          } else {
            vec_t<Dim::_3D> v { ZERO }, vXYZ { ZERO };
            metric.template transform_xyz<Idx::XYZ, Idx::U>(
              xp,
              { ux1(p), ux2(p), ux3(p) },
              v);
            v[2] = -v[2];
            metric.template transform_xyz<Idx::U, Idx::XYZ>(xp, v, vXYZ);
            ux1(p) = vXYZ[0];
            ux2(p) = vXYZ[1];
            ux3(p) = vXYZ[2];
          }
        }
      }
#if defined(MPI_ENABLED)
      if constexpr (D == Dim::_1D) {
        tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= ni1);
      } else if constexpr (D == Dim::_2D) {
        tag(p) = mpi::SendTag(tag(p), i1(p) < 0, i1(p) >= ni1, i2(p) < 0, i2(p) >= ni2);
      } else if constexpr (D == Dim::_3D) {
        tag(p) = mpi::SendTag(tag(p),
                              i1(p) < 0,
                              i1(p) >= ni1,
                              i2(p) < 0,
                              i2(p) >= ni2,
                              i3(p) < 0,
                              i3(p) >= ni3);
      }
#endif
    }
  };

} // namespace kernel::sr

#undef from_Xi_to_i_di
#undef from_Xi_to_i
#undef i_di_to_Xi

#endif // KERNELS_PARTICLE_PUSHER_SR_HPP

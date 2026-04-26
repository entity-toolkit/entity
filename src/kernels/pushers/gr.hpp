/**
 * @file kernels/pushers/gr.hpp
 * @brief Implementation of the particle pusher for GR
 * @implements
 *   - kernel::gr::Pusher_kernel<>
 * @namespaces:
 *   - kernel::gr::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - 3D implementation
 */

#ifndef KERNELS_PUSHERS_GR_HPP
#define KERNELS_PUSHERS_GR_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

/* -------------------------------------------------------------------------- */
/* Local macros                                                               */
/* -------------------------------------------------------------------------- */
#define from_Xi_to_i(XI, I)                                                    \
  {                                                                            \
    (I) = static_cast<int>(((XI) + 1)) - 1;                                    \
  }

#define from_Xi_to_i_di(XI, I, DI)                                             \
  {                                                                            \
    from_Xi_to_i((XI), (I));                                                   \
    (DI) = static_cast<prtldx_t>((XI)) - static_cast<prtldx_t>(I);             \
  }

#define i_di_to_Xi(I, DI) (static_cast<real_t>((I)) + static_cast<real_t>((DI)))

#define DERIVATIVE_IN_R(func, x)                                               \
  ((func({ (x)[0] + ctx.epsilon, (x)[1] }) -                                   \
    func({ (x)[0] - ctx.epsilon, (x)[1] })) /                                  \
   (TWO * ctx.epsilon))

#define DERIVATIVE_IN_TH(func, x)                                              \
  ((func({ (x)[0], (x)[1] + ctx.epsilon }) -                                   \
    func({ (x)[0], (x)[1] - ctx.epsilon })) /                                  \
   (TWO * ctx.epsilon))

/* -------------------------------------------------------------------------- */

namespace kernel::gr {
  using namespace ntt;

  struct Massive_t {};

  struct Massless_t {};

  /**
   * @brief Algorithm for the Particle pusher
   * @tparam M Metric
   */
  template <GRMetricClass M>
  class Pusher_kernel {
    static constexpr auto D = M::Dim;

  private:
    const PusherContext       ctx;
    const PusherBoundaries<D> bc;
    PusherArrays              particles;

    const randacc_ndfield_t<D, 6> DB;
    const randacc_ndfield_t<D, 6> DB0;
    const M                       metric;

    const real_t normalized_dt_half;

    bool is_axis_i2min { false }, is_axis_i2max { false };
    bool is_absorb_i1min { false }, is_absorb_i1max { false };

  public:
    Pusher_kernel(const PusherContext&       pusher_ctx,
                  const PusherBoundaries<D>& pusher_boundaries,
                  PusherArrays&              pusher_arrays,
                  const ndfield_t<D, 6>&     DB,
                  const ndfield_t<D, 6>&     DB0,
                  const M&                   metric)
      : ctx { pusher_ctx }
      , bc { pusher_boundaries }
      , particles { pusher_arrays }
      , DB { DB }
      , DB0 { DB0 }
      , metric { metric }
      , normalized_dt_half { HALF * (ctx.charge / ctx.mass) * ctx.omegaB0 *
                             ctx.dt } {}

    /**
     * @brief Main pusher subroutine for photon particles.
     */
    Inline void operator()(Massless_t, index_t) const;

    /**
     * @brief Main pusher subroutine for massive particles.
     */
    Inline void operator()(Massive_t, index_t) const;

    /**
     * @brief Iterative geodesic pusher substep for momentum only.
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicMomentumPush(T,
                                     const coord_t<D>&      xp,
                                     const vec_t<Dim::_3D>& vp,
                                     vec_t<Dim::_3D>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep for coordinate only.
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    template <typename T>
    Inline void GeodesicCoordinatePush(T,
                                       const coord_t<D>&      xp,
                                       const vec_t<Dim::_3D>& vp,
                                       coord_t<D>&            xp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     * @param vp_upd updated particle velocity [return].
     */
    template <typename T>
    Inline void GeodesicFullPush(T,
                                 const coord_t<D>&      xp,
                                 const vec_t<Dim::_3D>& vp,
                                 coord_t<D>&            xp_upd,
                                 vec_t<Dim::_3D>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate (at n + 1/2 for leapfrog, at n for old scheme).
     * @param vp particle velocity (at n + 1/2 for leapfrog, at n for old scheme).
     * @param phi updated phi [return].
     */
    template <typename T>
    Inline void UpdatePhi(T,
                          const coord_t<D>&      xp,
                          const vec_t<Dim::_3D>& vp,
                          real_t                 phi) const;

    /**
     * @brief EM pusher (Boris) substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param Dp_hat hatted electric field at the particle position.
     * @param Bp_hat hatted magnetic field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EMHalfPush(const coord_t<D>&      xp,
                           const vec_t<Dim::_3D>& vp,
                           const vec_t<Dim::_3D>& Dp_hat,
                           const vec_t<Dim::_3D>& Bp_hat,
                           vec_t<Dim::_3D>&       vp_upd) const {
      vec_t<Dim::_3D> D0 { Dp_hat[0], Dp_hat[1], Dp_hat[2] };
      vec_t<Dim::_3D> B0 { Bp_hat[0], Bp_hat[1], Bp_hat[2] };
      vec_t<Dim::_3D> vp_hat { ZERO }, vp_upd_hat { ZERO };
      metric.template transform<Idx::D, Idx::T>(xp, vp, vp_upd_hat);

      // this is a half-push
      real_t COEFF { normalized_dt_half * HALF * metric.alpha(xp) };

      D0[0] *= COEFF;
      D0[1] *= COEFF;
      D0[2] *= COEFF;

      vp_upd_hat[0] += D0[0];
      vp_upd_hat[1] += D0[1];
      vp_upd_hat[2] += D0[2];

      COEFF *= ONE / math::sqrt(ONE + SQR(vp_upd_hat[0]) + SQR(vp_upd_hat[1]) +
                                SQR(vp_upd_hat[2]));
      B0[0] *= COEFF;
      B0[1] *= COEFF;
      B0[2] *= COEFF;
      COEFF  = TWO / (ONE + SQR(B0[0]) + SQR(B0[1]) + SQR(B0[2]));

      vp_hat[0] = (vp_upd_hat[0] + vp_upd_hat[1] * B0[2] - vp_upd_hat[2] * B0[1]) *
                  COEFF;
      vp_hat[1] = (vp_upd_hat[1] + vp_upd_hat[2] * B0[0] - vp_upd_hat[0] * B0[2]) *
                  COEFF;
      vp_hat[2] = (vp_upd_hat[2] + vp_upd_hat[0] * B0[1] - vp_upd_hat[1] * B0[0]) *
                  COEFF;

      vp_upd_hat[0] += vp_hat[1] * B0[2] - vp_hat[2] * B0[1] + D0[0];
      vp_upd_hat[1] += vp_hat[2] * B0[0] - vp_hat[0] * B0[2] + D0[1];
      vp_upd_hat[2] += vp_hat[0] * B0[1] - vp_hat[1] * B0[0] + D0[2];

      metric.template transform<Idx::T, Idx::D>(xp, vp_upd_hat, vp_upd);
    }

    // Helper functions

    /**
     * @brief Direct field interpolation to particle position with arbitrary shape function.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    template <unsigned short O>
    Inline void interpolateFields(index_t          p,
                                  vec_t<Dim::_3D>& e,
                                  vec_t<Dim::_3D>& b) const;

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(u_i u_j h^ij) for massless particles.
     */
    Inline auto computeGamma(const Massless_t&,
                             const vec_t<Dim::_3D>& u_cov,
                             const vec_t<Dim::_3D>& u_cntrv) const -> real_t {
      return math::sqrt(
        u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] + u_cov[2] * u_cntrv[2]);
    }

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(1 + u_i u_j h^ij) for massive particles.
     */
    Inline auto computeGamma(const Massive_t&,
                             const vec_t<Dim::_3D>& u_cov,
                             const vec_t<Dim::_3D>& u_cntrv) const -> real_t {
      return math::sqrt(ONE + u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] +
                        u_cov[2] * u_cntrv[2]);
    }

    // Extra
    Inline void boundaryConditions(index_t) const;
  };

  /* -------------------------------------------------------------------------- */
  /*                               Geodesic pusher */
  /* -------------------------------------------------------------------------- */

  template <GRMetricClass M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicMomentumPush(T,
                                                     const coord_t<D>&      xp,
                                                     const vec_t<Dim::_3D>& vp,
                                                     vec_t<Dim::_3D>& vp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "1D not applicable");
    } else if constexpr (D == Dim::_2D) {
      // initialize midpoint values & updated values
      vec_t<Dim::_3D> vp_mid { ZERO };
      vec_t<Dim::_3D> vp_mid_cntrv { ZERO };
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < ctx.niter; ++i) {
        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpointы
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) / metric.alpha(xp) };

        // find updated velocity
        // vp_upd[0] =
        //   vp[0] +
        //   dt *
        //     (-metric.alpha(xp) * u0 * DERIVATIVE_IN_R(metric.alpha, xp) +
        //      vp_mid[0] * DERIVATIVE_IN_R(metric.beta1, xp) -
        //      (HALF / u0) *
        //        (DERIVATIVE_IN_R((metric.template h<1, 1>), xp) * SQR(vp_mid[0]) +
        //         DERIVATIVE_IN_R((metric.template h<2, 2>), xp) * SQR(vp_mid[1]) +
        //         DERIVATIVE_IN_R((metric.template h<3, 3>), xp) * SQR(vp_mid[2]) +
        //         TWO * DERIVATIVE_IN_R((metric.template h<1, 3>), xp) *
        //           vp_mid[0] * vp_mid[2]));
        // vp_upd[1] =
        //   vp[1] +
        //   dt *
        //     (-metric.alpha(xp) * u0 * DERIVATIVE_IN_TH(metric.alpha, xp) +
        //      vp_mid[0] * DERIVATIVE_IN_TH(metric.beta1, xp) -
        //      (HALF / u0) *
        //        (DERIVATIVE_IN_TH((metric.template h<1, 1>), xp) * SQR(vp_mid[0]) +
        //         DERIVATIVE_IN_TH((metric.template h<2, 2>), xp) * SQR(vp_mid[1]) +
        //         DERIVATIVE_IN_TH((metric.template h<3, 3>), xp) * SQR(vp_mid[2]) +
        //         TWO * DERIVATIVE_IN_TH((metric.template h<1, 3>), xp) *
        //           vp_mid[0] * vp_mid[2]));
        vp_upd[0] = vp[0] +
                    ctx.dt * (-metric.alpha(xp) * u0 * metric.dr_alpha(xp) +
                              vp_mid[0] * metric.dr_beta1(xp) -
                              (HALF / u0) *
                                (metric.dr_h11(xp) * SQR(vp_mid[0]) +
                                 metric.dr_h22(xp) * SQR(vp_mid[1]) +
                                 metric.dr_h33(xp) * SQR(vp_mid[2]) +
                                 TWO * metric.dr_h13(xp) * vp_mid[0] * vp_mid[2]));
        vp_upd[1] = vp[1] +
                    ctx.dt * (-metric.alpha(xp) * u0 * metric.dt_alpha(xp) +
                              vp_mid[0] * metric.dt_beta1(xp) -
                              (HALF / u0) *
                                (metric.dt_h11(xp) * SQR(vp_mid[0]) +
                                 metric.dt_h22(xp) * SQR(vp_mid[1]) +
                                 metric.dt_h33(xp) * SQR(vp_mid[2]) +
                                 TWO * metric.dt_h13(xp) * vp_mid[0] * vp_mid[2]));
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <GRMetricClass M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicCoordinatePush(T,
                                                       const coord_t<D>& xp,
                                                       const vec_t<Dim::_3D>& vp,
                                                       coord_t<D>& xp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "GeodesicCoordinatePush: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      vec_t<Dim::_3D>   vp_cntrv { ZERO };
      coord_t<Dim::_2D> xp_mid { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];

      for (auto i { 0 }; i < ctx.niter; ++i) {
        // find midpoint values
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp_mid, vp, vp_cntrv);
        real_t gamma = computeGamma(T {}, vp, vp_cntrv);

        // find midpoint coefficients
        real_t u0 { gamma / metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + ctx.dt * (vp_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + ctx.dt * (vp_cntrv[1] / u0);
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <GRMetricClass M>
  template <typename T>
  Inline void Pusher_kernel<M>::GeodesicFullPush(T,
                                                 const coord_t<D>&      xp,
                                                 const vec_t<Dim::_3D>& vp,
                                                 coord_t<D>&            xp_upd,
                                                 vec_t<Dim::_3D>& vp_upd) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "GeodesicFullPush: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      // initialize midpoint values & updated values
      vec_t<Dim::_2D> xp_mid { ZERO };
      vec_t<Dim::_3D> vp_mid { ZERO }, vp_mid_cntrv { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < ctx.niter; ++i) {
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.template transform<Idx::D, Idx::U>(xp_mid, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpoint
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) /
                    metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + ctx.dt * (vp_mid_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + ctx.dt * (vp_mid_cntrv[1] / u0);

        // find updated velocity
        vp_upd[0] =
          vp[0] +
          ctx.dt *
            (-metric.alpha(xp_mid) * u0 * DERIVATIVE_IN_R(metric.alpha, xp_mid) +
             vp_mid[0] * DERIVATIVE_IN_R(metric.beta1, xp_mid) -
             (HALF / u0) *
               (DERIVATIVE_IN_R((metric.template h<1, 1>), xp_mid) * SQR(vp_mid[0]) +
                DERIVATIVE_IN_R((metric.template h<2, 2>), xp_mid) * SQR(vp_mid[1]) +
                DERIVATIVE_IN_R((metric.template h<3, 3>), xp_mid) * SQR(vp_mid[2]) +
                TWO * DERIVATIVE_IN_R((metric.template h<1, 3>), xp_mid) *
                  vp_mid[0] * vp_mid[2]));
        vp_upd[1] = vp[1] +
                    ctx.dt *
                      (-metric.alpha(xp_mid) * u0 *
                         DERIVATIVE_IN_TH(metric.alpha, xp_mid) +
                       vp_mid[0] * DERIVATIVE_IN_TH(metric.beta1, xp_mid) -
                       (HALF / u0) *
                         (DERIVATIVE_IN_TH((metric.template h<1, 1>), xp_mid) *
                            SQR(vp_mid[0]) +
                          DERIVATIVE_IN_TH((metric.template h<2, 2>), xp_mid) *
                            SQR(vp_mid[1]) +
                          DERIVATIVE_IN_TH((metric.template h<3, 3>), xp_mid) *
                            SQR(vp_mid[2]) +
                          TWO * DERIVATIVE_IN_TH((metric.template h<1, 3>), xp_mid) *
                            vp_mid[0] * vp_mid[2]));
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  /* -------------------------------------------------------------------------- */
  /*                                 Phi pusher */
  /* -------------------------------------------------------------------------- */
  template <GRMetricClass M>
  template <typename T>
  Inline void Pusher_kernel<M>::UpdatePhi(T,
                                          const coord_t<D>&      xp,
                                          const vec_t<Dim::_3D>& vp,
                                          real_t                 phi) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "UpdatePhi: 1D implementation called");
    } else if constexpr (D == Dim::_2D) {
      vec_t<Dim::_3D> vp_cntrv { ZERO };
      metric.template transform<Idx::D, Idx::U>(xp, vp, vp_cntrv);
      real_t u0 { computeGamma(T {}, vp, vp_cntrv) / metric.alpha(xp) };
      phi += ctx.dt * vp_cntrv[2] / u0;
      if (phi >= constant::TWO_PI) {
        phi -= constant::TWO_PI;
      } else if (phi < ZERO) {
        phi += constant::TWO_PI;
      }
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  template <GRMetricClass M>
  template <unsigned short O>
  Inline void Pusher_kernel<M>::interpolateFields(index_t          p,
                                                  vec_t<Dim::_3D>& e0,
                                                  vec_t<Dim::_3D>& b0) const {

    // Zig-zag interpolation
    if constexpr (O == 0u) {

      if constexpr (D == Dim::_1D) {
        raise::KernelError(HERE, "1D not applicable");
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

        real_t pondmx = static_cast<real_t>(indx + 1) - (dx1_ + HALF);
        real_t pondpx = ONE - pondmx;
        real_t pondmy = static_cast<real_t>(indy + 1) - (dx2_ + HALF);
        real_t pondpy = ONE - pondmy;

        // Ex1
        // Interpolate --- (dual, primal)
        c000  = DB(i - 1 + indx, j, em::dx1);
        c100  = DB(i + indx, j, em::dx1);
        c010  = DB(i - 1 + indx, j + 1, em::dx1);
        c110  = DB(i + indx, j + 1, em::dx1);
        c00   = c000 * pondmx + c100 * pondpx;
        c10   = c010 * pondmx + c110 * pondpx;
        e0[0] = c00 * ponpmy + c10 * ponppy;
        // Ex2
        // Interpolate -- (primal, dual)
        c000  = DB(i, j - 1 + indy, em::dx2);
        c100  = DB(i + 1, j - 1 + indy, em::dx2);
        c010  = DB(i, j + indy, em::dx2);
        c110  = DB(i + 1, j + indy, em::dx2);
        c00   = c000 * ponpmx + c100 * ponppx;
        c10   = c010 * ponpmx + c110 * ponppx;
        e0[1] = c00 * pondmy + c10 * pondpy;
        // Ex3
        // Interpolate -- (primal, primal)
        c000  = DB(i, j, em::dx3);
        c100  = DB(i + 1, j, em::dx3);
        c010  = DB(i, j + 1, em::dx3);
        c110  = DB(i + 1, j + 1, em::dx3);
        c00   = c000 * ponpmx + c100 * ponppx;
        c10   = c010 * ponpmx + c110 * ponppx;
        e0[2] = c00 * ponpmy + c10 * ponppy;

        // Bx1
        // Interpolate -- (primal, dual)
        c000  = DB0(i, j - 1 + indy, em::bx1);
        c100  = DB0(i + 1, j - 1 + indy, em::bx1);
        c010  = DB0(i, j + indy, em::bx1);
        c110  = DB0(i + 1, j + indy, em::bx1);
        c00   = c000 * ponpmx + c100 * ponppx;
        c10   = c010 * ponpmx + c110 * ponppx;
        b0[0] = c00 * pondmy + c10 * pondpy;
        // Bx2
        // Interpolate -- (dual, primal)
        c000  = DB0(i - 1 + indx, j, em::bx2);
        c100  = DB0(i + indx, j, em::bx2);
        c010  = DB0(i - 1 + indx, j + 1, em::bx2);
        c110  = DB0(i + indx, j + 1, em::bx2);
        c00   = c000 * pondmx + c100 * pondpx;
        c10   = c010 * pondmx + c110 * pondpx;
        b0[1] = c00 * ponpmy + c10 * ponppy;
        // Bx3
        // Interpolate -- (dual, dual)
        c000  = DB0(i - 1 + indx, j - 1 + indy, em::bx3);
        c100  = DB0(i + indx, j - 1 + indy, em::bx3);
        c010  = DB0(i - 1 + indx, j + indy, em::bx3);
        c110  = DB0(i + indx, j + indy, em::bx3);
        c00   = c000 * pondmx + c100 * pondpx;
        c10   = c010 * pondmx + c110 * pondpx;
        b0[2] = c00 * pondmy + c10 * pondpy;
      } else if constexpr (D == Dim::_3D) {
        raise::KernelError(HERE, "3D not applicable");
      }
    } else if constexpr (O >= 1u) {

      if constexpr (D == Dim::_1D) {
        raise::KernelError(HERE, "1D not applicable");
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
            c0 += S1d[idx1] * DB(id_min + idx1, jp_min + idx2, em::dx1);
          }
          e0[0] += c0 * S2p[idx2];
        }

        // Ex2 -- primal, dual
        e0[1] = ZERO;
        for (int idx2 = 0; idx2 < O + 1; idx2++) {
          real_t c0 = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            c0 += S1p[idx1] * DB(ip_min + idx1, jd_min + idx2, em::dx2);
          }
          e0[1] += c0 * S2d[idx2];
        }

        // Ex3 -- primal, primal
        e0[2] = ZERO;
        for (int idx2 = 0; idx2 < O + 1; idx2++) {
          real_t c0 = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            c0 += S1p[idx1] * DB(ip_min + idx1, jp_min + idx2, em::dx3);
          }
          e0[2] += c0 * S2p[idx2];
        }

        // Bx1 -- primal, dual
        b0[0] = ZERO;
        for (int idx2 = 0; idx2 < O + 1; idx2++) {
          real_t c0 = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            c0 += S1p[idx1] * DB0(ip_min + idx1, jd_min + idx2, em::bx1);
          }
          b0[0] += c0 * S2d[idx2];
        }

        // Bx2 -- dual, primal
        b0[1] = ZERO;
        for (int idx2 = 0; idx2 < O + 1; idx2++) {
          real_t c0 = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            c0 += S1d[idx1] * DB0(id_min + idx1, jp_min + idx2, em::bx2);
          }
          b0[1] += c0 * S2p[idx2];
        }

        // Bx3 -- dual, dual
        b0[2] = ZERO;
        for (int idx2 = 0; idx2 < O + 1; idx2++) {
          real_t c0 = ZERO;
          for (int idx1 = 0; idx1 < O + 1; idx1++) {
            c0 += S1d[idx1] * DB0(id_min + idx1, jd_min + idx2, em::bx3);
          }
          b0[2] += c0 * S2d[idx2];
        }

      } else if constexpr (D == Dim::_3D) {
        raise::KernelError(HERE, "3D not applicable");
      }
    }
  }

  /* ------------------------------ Photon pusher ----------------------------- */

  template <GRMetricClass M>
  Inline void Pusher_kernel<M>::operator()(Massless_t, index_t p) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "Photon pusher not implemented for 1D");
    } else if constexpr (D == Dim::_2D) {
      if (particles.tag(p) != ParticleTag::alive) {
        if (particles.tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      // record previous coordinate
      particles.i1_prev(p)  = particles.i1(p);
      particles.i2_prev(p)  = particles.i2(p);
      particles.dx1_prev(p) = particles.dx1(p);
      particles.dx2_prev(p) = particles.dx2(p);

      coord_t<Dim::_2D> xp { ZERO };
      vec_t<Dim::_3D> vp { particles.ux1(p), particles.ux2(p), particles.ux3(p) };

      xp[0] = i_di_to_Xi(particles.i1(p), particles.dx1(p));
      xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));

      /* ----------------------------- Leapfrog pusher ---------------------------- */
      // u_i(n - 1/2) -> u_i(n + 1/2)
      vec_t<Dim::_3D> vp_upd { ZERO };
      GeodesicMomentumPush<Massless_t>(Massless_t {}, xp, vp, vp_upd);
      // x^i(n) -> x^i(n + 1)
      coord_t<Dim::_2D> xp_upd { ZERO };
      GeodesicCoordinatePush<Massless_t>(Massless_t {}, xp, vp_upd, xp_upd);
      // update phi
      UpdatePhi<Massless_t>(
        Massless_t {},
        { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
        vp_upd,
        particles.phi(p));

      // update coordinate
      int      i1_, i2_;
      prtldx_t dx1_, dx2_;
      from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
      from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
      particles.i1(p)  = i1_;
      particles.dx1(p) = dx1_;
      particles.i2(p)  = i2_;
      particles.dx2(p) = dx2_;

      // update velocity
      particles.ux1(p) = vp_upd[0];
      particles.ux2(p) = vp_upd[1];
      particles.ux3(p) = vp_upd[2];

      boundaryConditions(p);
    } else if constexpr (D == Dim::_3D) {
      raise::KernelError(HERE, "3D not implemented");
    }
  }

  /* ------------------------- Massive particle pusher ------------------------ */

  template <GRMetricClass M>
  Inline void Pusher_kernel<M>::operator()(Massive_t, index_t p) const {
    if constexpr (D == Dim::_1D) {
      raise::KernelError(HERE, "Massive pusher not implemented for 1D");
    } else if constexpr (D == Dim::_2D) {
      if (particles.tag(p) != ParticleTag::alive) {
        if (particles.tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "Invalid particle tag in pusher");
        }
        return;
      }
      // record previous coordinate
      particles.i1_prev(p)  = particles.i1(p);
      particles.i2_prev(p)  = particles.i2(p);
      particles.dx1_prev(p) = particles.dx1(p);
      particles.dx2_prev(p) = particles.dx2(p);

      coord_t<Dim::_2D> xp { ZERO };

      xp[0] = i_di_to_Xi(particles.i1(p), particles.dx1(p));
      xp[1] = i_di_to_Xi(particles.i2(p), particles.dx2(p));

      coord_t<Dim::_2D> xp_ { ZERO };
      xp_[0] = xp[0];
      real_t       theta_Cd { xp[1] };
      const real_t theta_Ph { metric.template convert<2, Crd::Cd, Crd::Ph>(
        theta_Cd) };
      const real_t small_angle { constant::SMALL_ANGLE_GR };
      const auto   large_angle { constant::PI - small_angle };
      if (theta_Ph < small_angle) {
        theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(small_angle);
      } else if (theta_Ph >= large_angle) {
        theta_Cd = metric.template convert<2, Crd::Ph, Crd::Cd>(large_angle);
      }
      xp_[1] = theta_Cd;

      vec_t<Dim::_3D> Dp_cntrv { ZERO }, Bp_cntrv { ZERO }, Dp_hat { ZERO },
        Bp_hat { ZERO };
      interpolateFields<SHAPE_ORDER>(p, Dp_cntrv, Bp_cntrv);
      metric.template transform<Idx::U, Idx::T>(xp, Dp_cntrv, Dp_hat);
      metric.template transform<Idx::U, Idx::T>(xp, Bp_cntrv, Bp_hat);

      vec_t<Dim::_3D> vp { particles.ux1(p), particles.ux2(p), particles.ux3(p) };

      /* -------------------------------- Leapfrog -------------------------------- */
      /* u_i(n - 1/2) -> u*_i(n) */
      vec_t<Dim::_3D> vp_upd { ZERO };
      EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      /* u*_i(n) -> u**_i(n) */
      vp[0] = vp_upd[0];
      vp[1] = vp_upd[1];
      vp[2] = vp_upd[2];
      GeodesicMomentumPush<Massive_t>(Massive_t {}, xp_, vp, vp_upd);
      /* u**_i(n) -> u_i(n + 1/2) */
      vp[0] = vp_upd[0];
      vp[1] = vp_upd[1];
      vp[2] = vp_upd[2];
      EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
      /* x^i(n) -> x^i(n + 1) */
      coord_t<Dim::_2D> xp_upd { ZERO };
      GeodesicCoordinatePush<Massive_t>(Massive_t {}, xp, vp_upd, xp_upd);

      // update phi
      UpdatePhi<Massive_t>(
        Massive_t {},
        { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
        vp_upd,
        particles.phi(p));

      // update coordinate
      int      i1_, i2_;
      prtldx_t dx1_, dx2_;
      from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
      from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
      particles.i1(p)  = i1_;
      particles.dx1(p) = dx1_;
      particles.i2(p)  = i2_;
      particles.dx2(p) = dx2_;

      // update velocity
      particles.ux1(p) = vp_upd[0];
      particles.ux2(p) = vp_upd[1];
      particles.ux3(p) = vp_upd[2];

      boundaryConditions(p);
    } else if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
    }
  }

  // Boundary conditions

  template <GRMetricClass M>
  Inline void Pusher_kernel<M>::boundaryConditions(index_t p) const {
    if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
      if ((particles.i1(p) < 0 && bc.is_absorb_i1min) or
          (particles.i1(p) >= ctx.ni1 && bc.is_absorb_i1max)) {
        particles.tag(p) = ParticleTag::dead;
      }
    }
    if constexpr (D == Dim::_2D || D == Dim::_3D) {
      if (particles.i2(p) < 0) {
        if (bc.is_axis_i2min) {
          particles.i2(p)  = 0;
          particles.dx2(p) = ONE - particles.dx2(p);
          particles.ux2(p) = -particles.ux2(p);
        }
      } else if (particles.i2(p) >= ctx.ni2) {
        if (bc.is_axis_i2max) {
          particles.i2(p)  = ctx.ni2 - 1;
          particles.dx2(p) = ONE - particles.dx2(p);
          particles.ux2(p) = -particles.ux2(p);
        }
      }
    }
    if constexpr (D == Dim::_3D) {
      raise::KernelNotImplementedError(HERE);
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

} // namespace kernel::gr

#undef DERIVATIVE_IN_TH
#undef DERIVATIVE_IN_R

#undef i_di_to_Xi
#undef from_Xi_to_i_di
#undef from_Xi_to_i

#endif // KERNELS_PARTICLE_PUSHER_GR_HPP

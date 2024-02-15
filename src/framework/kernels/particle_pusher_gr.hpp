#ifndef FRAMEWORK_KERNELS_PARTICLE_PUSHER_GR_H
#define FRAMEWORK_KERNELS_PARTICLE_PUSHER_GR_H

#include "wrapper.h"

#include "particle_macros.h"

#include "meshblock/fields.h"
#include "meshblock/particles.h"
#include "utils/qmath.h"

namespace ntt {
  struct Massive_t {};

  struct Massless_t {};

  /**
   * @brief Algorithm for the Particle pusher.
   * @tparam D Dimension.
   * @tparam M Metric.
   */
  template <Dimension D, class M>
  class Pusher_kernel {
    ndfield_t<D, 6>    DB;
    ndfield_t<D, 6>    DB0;
    array_t<int*>      i1, i2, i3;
    array_t<int*>      i1_prev, i2_prev, i3_prev;
    array_t<prtldx_t*> dx1, dx2, dx3;
    array_t<prtldx_t*> dx1_prev, dx2_prev, dx3_prev;
    array_t<real_t*>   ux1, ux2, ux3;
    array_t<real_t*>   phi;
    array_t<short*>    tag;
    const M            metric;

    const real_t coeff, dt;
    const int    ni1, ni2, ni3;
    const real_t epsilon;
    const int    niter;
    const int    i1_absorb;

    bool is_axis_i2min { false }, is_axis_i2max { false };
    bool is_absorb_i1min { false }, is_absorb_i1max { false };
    bool is_absorb_i2min { false }, is_absorb_i2max { false };

  public:
    Pusher_kernel(const ndfield_t<D, 6>&    DB,
                  const ndfield_t<D, 6>&    DB0,
                  const array_t<int*>&      i1,
                  const array_t<int*>&      i2,
                  const array_t<int*>&      i3,
                  const array_t<int*>&      i1_prev,
                  const array_t<int*>&      i2_prev,
                  const array_t<int*>&      i3_prev,
                  const array_t<prtldx_t*>& dx1,
                  const array_t<prtldx_t*>& dx2,
                  const array_t<prtldx_t*>& dx3,
                  const array_t<prtldx_t*>& dx1_prev,
                  const array_t<prtldx_t*>& dx2_prev,
                  const array_t<prtldx_t*>& dx3_prev,
                  const array_t<real_t*>&   ux1,
                  const array_t<real_t*>&   ux2,
                  const array_t<real_t*>&   ux3,
                  const array_t<real_t*>&   phi,
                  const array_t<short*>&    tag,
                  const M&                  metric,
                  const real_t&             coeff,
                  const real_t&             dt,
                  const int&                ni1,
                  const int&                ni2,
                  const int&                ni3,
                  const real_t&             epsilon,
                  const int&                niter,
                  const std::vector<std::vector<BoundaryCondition>>& boundaries) :
      DB { DB },
      DB0 { DB0 },
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
      coeff { coeff },
      dt { dt },
      ni1 { ni1 },
      ni2 { ni2 },
      ni3 { ni3 },
      epsilon { epsilon },
      niter { niter },
      i1_absorb { static_cast<int>(metric.x1_Phys2Code(metric.rhorizon())) - 5 } {

      NTTHostErrorIf(boundaries.size() < 2, "boundaries defined incorrectly");
      is_absorb_i1min = (boundaries[0][0] == BoundaryCondition::OPEN) ||
                        (boundaries[0][0] == BoundaryCondition::CUSTOM) ||
                        (boundaries[0][0] == BoundaryCondition::ABSORB);
      is_absorb_i1max = (boundaries[0][1] == BoundaryCondition::OPEN) ||
                        (boundaries[0][1] == BoundaryCondition::CUSTOM) ||
                        (boundaries[0][1] == BoundaryCondition::ABSORB);
      is_absorb_i2min = (boundaries[1][0] == BoundaryCondition::OPEN) ||
                        (boundaries[1][0] == BoundaryCondition::CUSTOM) ||
                        (boundaries[1][0] == BoundaryCondition::ABSORB);
      is_absorb_i2max = (boundaries[1][1] == BoundaryCondition::OPEN) ||
                        (boundaries[1][1] == BoundaryCondition::CUSTOM) ||
                        (boundaries[1][1] == BoundaryCondition::ABSORB);
      is_axis_i2min = (boundaries[1][0] == BoundaryCondition::AXIS);
      is_axis_i2max = (boundaries[1][1] == BoundaryCondition::AXIS);
    }

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
                                     const coord_t<D>&  xp,
                                     const vec_t<Dim3>& vp,
                                     vec_t<Dim3>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep for coordinate only.
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate.
     * @param vp particle velocity.
     * @param xp_upd updated particle coordinate [return].
     */
    template <typename T>
    Inline void GeodesicCoordinatePush(T,
                                       const coord_t<D>&  xp,
                                       const vec_t<Dim3>& vp,
                                       coord_t<D>&        xp_upd) const;

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
                                 const coord_t<D>&  xp,
                                 const vec_t<Dim3>& vp,
                                 coord_t<D>&        xp_upd,
                                 vec_t<Dim3>&       vp_upd) const;

    /**
     * @brief Iterative geodesic pusher substep (old method).
     * @tparam T Push type (Massless_t or Massive_t)
     * @param xp particle coordinate (at n + 1/2 for leapfrog, at n for old scheme).
     * @param vp particle velocity (at n + 1/2 for leapfrog, at n for old scheme).
     * @param phi updated phi [return].
     */
    template <typename T>
    Inline void UpdatePhi(T,
                          const coord_t<D>&  xp,
                          const vec_t<Dim3>& vp,
                          real_t&            phi) const;

    /**
     * @brief EM pusher (Boris) substep.
     * @param xp coordinate of the particle.
     * @param vp covariant velocity of the particle.
     * @param Dp_hat hatted electric field at the particle position.
     * @param Bp_hat hatted magnetic field at the particle position.
     * @param v_upd updated covarient velocity of the particle [return].
     */
    Inline void EMHalfPush(const coord_t<D>&  xp,
                           const vec_t<Dim3>& vp,
                           const vec_t<Dim3>& Dp_hat,
                           const vec_t<Dim3>& Bp_hat,
                           vec_t<Dim3>&       vp_upd) const {
      vec_t<Dim3> D0 { Dp_hat[0], Dp_hat[1], Dp_hat[2] };
      vec_t<Dim3> B0 { Bp_hat[0], Bp_hat[1], Bp_hat[2] };
      vec_t<Dim3> vp_hat { ZERO }, vp_upd_hat { ZERO };
      metric.v3_Cov2Hat(xp, vp, vp_upd_hat);

      // this is a half-push
      real_t COEFF { coeff * HALF * metric.alpha(xp) };

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

      metric.v3_Hat2Cov(xp, vp_upd_hat, vp_upd);
    }

    // Helper functions

    /**
     * @brief First order Yee mesh field interpolation to particle position.
     * @param p index of the particle.
     * @param e interpolated e-field vector of size 3 [return].
     * @param b interpolated b-field vector of size 3 [return].
     */
    Inline void interpolateFields(index_t& p, vec_t<Dim3>& e, vec_t<Dim3>& b) const;

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(u_i u_j h^ij) for massless particles.
     */
    Inline auto computeGamma(const Massless_t&,
                             const vec_t<Dim3>& u_cov,
                             const vec_t<Dim3>& u_cntrv) const -> real_t {
      return math::sqrt(
        u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] + u_cov[2] * u_cntrv[2]);
    }

    /**
     * @brief Compute the gamma parameter Gamma = sqrt(1 + u_i u_j h^ij) for massive particles.
     */
    Inline auto computeGamma(const Massive_t&,
                             const vec_t<Dim3>& u_cov,
                             const vec_t<Dim3>& u_cntrv) const -> real_t {
      return math::sqrt(ONE + u_cov[0] * u_cntrv[0] + u_cov[1] * u_cntrv[1] +
                        u_cov[2] * u_cntrv[2]);
    }

    // Extra
    Inline void boundaryConditions(index_t&) const;
  };

  /* -------------------------------------------------------------------------- */
  /*                               Geodesic pusher */
  /* -------------------------------------------------------------------------- */

#define DERIVATIVE_IN_R(func, x)                                                     \
  ((metric.func({ x[0] + epsilon, x[1] }) - metric.func({ x[0] - epsilon, x[1] })) / \
   (TWO * epsilon))

#define DERIVATIVE_IN_TH(func, x)                                                    \
  ((metric.func({ x[0], x[1] + epsilon }) - metric.func({ x[0], x[1] - epsilon })) / \
   (TWO * epsilon))

  template <Dimension D, class M>
  template <typename T>
  Inline void Pusher_kernel<D, M>::GeodesicMomentumPush(T,
                                                        const coord_t<D>&  xp,
                                                        const vec_t<Dim3>& vp,
                                                        vec_t<Dim3>& vp_upd) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      // initialize midpoint values & updated values
      vec_t<Dim3> vp_mid { ZERO };
      vec_t<Dim3> vp_mid_cntrv { ZERO };
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < niter; ++i) {
        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.v3_Cov2Cntrv(xp, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpoint
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) / metric.alpha(xp) };

        // find updated velocity
        vp_upd[0] = vp[0] +
                    dt *
                      (-metric.alpha(xp) * u0 * DERIVATIVE_IN_R(alpha, xp) +
                       vp_mid[0] * DERIVATIVE_IN_R(beta1, xp) -
                       (HALF / u0) *
                         (DERIVATIVE_IN_R(h11, xp) * SQR(vp_mid[0]) +
                          DERIVATIVE_IN_R(h22, xp) * SQR(vp_mid[1]) +
                          DERIVATIVE_IN_R(h33, xp) * SQR(vp_mid[2]) +
                          TWO * DERIVATIVE_IN_R(h13, xp) * vp_mid[0] * vp_mid[2]));
        vp_upd[1] = vp[1] +
                    dt * (-metric.alpha(xp) * u0 * DERIVATIVE_IN_TH(alpha, xp) +
                          vp_mid[1] * DERIVATIVE_IN_TH(beta1, xp) -
                          (HALF / u0) *
                            (DERIVATIVE_IN_TH(h11, xp) * SQR(vp_mid[0]) +
                             DERIVATIVE_IN_TH(h22, xp) * SQR(vp_mid[1]) +
                             DERIVATIVE_IN_TH(h33, xp) * SQR(vp_mid[2]) +
                             TWO * DERIVATIVE_IN_TH(h13, xp) * vp_mid[0] *
                               vp_mid[2]));
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  template <Dimension D, class M>
  template <typename T>
  Inline void Pusher_kernel<D, M>::GeodesicCoordinatePush(T,
                                                          const coord_t<D>&  xp,
                                                          const vec_t<Dim3>& vp,
                                                          coord_t<D>& xp_upd) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      vec_t<Dim3>   vp_cntrv { ZERO };
      coord_t<Dim2> xp_mid { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];

      for (auto i { 0 }; i < niter; ++i) {
        // find midpoint values
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find contravariant midpoint velocity
        metric.v3_Cov2Cntrv(xp_mid, vp, vp_cntrv);
        real_t gamma = computeGamma(T {}, vp, vp_cntrv);

        // find midpoint coefficients
        real_t u0 { gamma / metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + dt * (vp_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + dt * (vp_cntrv[1] / u0);
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  template <Dimension D, class M>
  template <typename T>
  Inline void Pusher_kernel<D, M>::GeodesicFullPush(T,
                                                    const coord_t<D>&  xp,
                                                    const vec_t<Dim3>& vp,
                                                    coord_t<D>&        xp_upd,
                                                    vec_t<Dim3>& vp_upd) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      // initialize midpoint values & updated values
      vec_t<Dim2> xp_mid { ZERO };
      vec_t<Dim3> vp_mid { ZERO }, vp_mid_cntrv { ZERO };
      xp_upd[0] = xp[0];
      xp_upd[1] = xp[1];
      vp_upd[0] = vp[0];
      vp_upd[1] = vp[1];
      vp_upd[2] = vp[2];

      for (auto i { 0 }; i < niter; ++i) {
        xp_mid[0] = HALF * (xp[0] + xp_upd[0]);
        xp_mid[1] = HALF * (xp[1] + xp_upd[1]);

        // find midpoint values
        vp_mid[0] = HALF * (vp[0] + vp_upd[0]);
        vp_mid[1] = HALF * (vp[1] + vp_upd[1]);
        vp_mid[2] = vp[2];

        // find contravariant midpoint velocity
        metric.v3_Cov2Cntrv(xp_mid, vp_mid, vp_mid_cntrv);

        // find Gamma / alpha at midpoint
        real_t u0 { computeGamma(T {}, vp_mid, vp_mid_cntrv) /
                    metric.alpha(xp_mid) };

        // find updated coordinate shift
        xp_upd[0] = xp[0] + dt * (vp_mid_cntrv[0] / u0 - metric.beta1(xp_mid));
        xp_upd[1] = xp[1] + dt * (vp_mid_cntrv[1] / u0);

        // find updated velocity
        vp_upd[0] = vp[0] +
                    dt * (-metric.alpha(xp_mid) * u0 *
                            DERIVATIVE_IN_R(alpha, xp_mid) +
                          vp_mid[0] * DERIVATIVE_IN_R(beta1, xp_mid) -
                          (HALF / u0) *
                            (DERIVATIVE_IN_R(h11, xp_mid) * SQR(vp_mid[0]) +
                             DERIVATIVE_IN_R(h22, xp_mid) * SQR(vp_mid[1]) +
                             DERIVATIVE_IN_R(h33, xp_mid) * SQR(vp_mid[2]) +
                             TWO * DERIVATIVE_IN_R(h13, xp_mid) * vp_mid[0] *
                               vp_mid[2]));
        vp_upd[1] = vp[1] +
                    dt * (-metric.alpha(xp_mid) * u0 *
                            DERIVATIVE_IN_TH(alpha, xp_mid) +
                          vp_mid[1] * DERIVATIVE_IN_TH(beta1, xp_mid) -
                          (HALF / u0) *
                            (DERIVATIVE_IN_TH(h11, xp_mid) * SQR(vp_mid[0]) +
                             DERIVATIVE_IN_TH(h22, xp_mid) * SQR(vp_mid[1]) +
                             DERIVATIVE_IN_TH(h33, xp_mid) * SQR(vp_mid[2]) +
                             TWO * DERIVATIVE_IN_TH(h13, xp_mid) * vp_mid[0] *
                               vp_mid[2]));
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

#undef DERIVATIVE_IN_TH
#undef DERIVATIVE_IN_R

  /* -------------------------------------------------------------------------- */
  /*                                 Phi pusher */
  /* -------------------------------------------------------------------------- */
  template <Dimension D, class M>
  template <typename T>
  Inline void Pusher_kernel<D, M>::UpdatePhi(T,
                                             const coord_t<D>&  xp,
                                             const vec_t<Dim3>& vp,
                                             real_t&            phi) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      vec_t<Dim3> vp_cntrv { ZERO };
      metric.v3_Cov2Cntrv(xp, vp, vp_cntrv);
      real_t u0 { computeGamma(T {}, vp, vp_cntrv) / metric.alpha(xp) };
      phi += dt * vp_cntrv[2] / u0;
      if (phi >= constant::TWO_PI) {
        phi -= constant::TWO_PI;
      } else if (phi < ZERO) {
        phi += constant::TWO_PI;
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  template <Dimension D, class M>
  Inline void Pusher_kernel<D, M>::interpolateFields(index_t&     p,
                                                     vec_t<Dim3>& e0,
                                                     vec_t<Dim3>& b0) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      // first order interpolation
      // Using fields em::e = D(t = n), and em::b0 = B(t = n)

      const int  i { i1(p) + static_cast<int>(N_GHOSTS) };
      const int  j { i2(p) + static_cast<int>(N_GHOSTS) };
      const auto dx1_ { static_cast<real_t>(dx1(p)) };
      const auto dx2_ { static_cast<real_t>(dx2(p)) };

      // first order
      real_t c000, c100, c010, c110, c00, c10;

      // Ex1
      // interpolate to nodes
      c000  = HALF * (DB(i, j, em::dx1) + DB(i - 1, j, em::dx1));
      c100  = HALF * (DB(i, j, em::dx1) + DB(i + 1, j, em::dx1));
      c010  = HALF * (DB(i, j + 1, em::dx1) + DB(i - 1, j + 1, em::dx1));
      c110  = HALF * (DB(i, j + 1, em::dx1) + DB(i + 1, j + 1, em::dx1));
      // interpolate from nodes to the particle position
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex2
      c000  = HALF * (DB(i, j, em::dx2) + DB(i, j - 1, em::dx2));
      c100  = HALF * (DB(i + 1, j, em::dx2) + DB(i + 1, j - 1, em::dx2));
      c010  = HALF * (DB(i, j, em::dx2) + DB(i, j + 1, em::dx2));
      c110  = HALF * (DB(i + 1, j, em::dx2) + DB(i + 1, j + 1, em::dx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Ex3
      c000  = DB(i, j, em::dx3);
      c100  = DB(i + 1, j, em::dx3);
      c010  = DB(i, j + 1, em::dx3);
      c110  = DB(i + 1, j + 1, em::dx3);
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      e0[2] = c00 * (ONE - dx2_) + c10 * dx2_;

      // Bx1
      c000  = HALF * (DB0(i, j, em::bx1) + DB0(i, j - 1, em::bx1));
      c100  = HALF * (DB0(i + 1, j, em::bx1) + DB0(i + 1, j - 1, em::bx1));
      c010  = HALF * (DB0(i, j, em::bx1) + DB0(i, j + 1, em::bx1));
      c110  = HALF * (DB0(i + 1, j, em::bx1) + DB0(i + 1, j + 1, em::bx1));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[0] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx2
      c000  = HALF * (DB0(i - 1, j, em::bx2) + DB0(i, j, em::bx2));
      c100  = HALF * (DB0(i, j, em::bx2) + DB0(i + 1, j, em::bx2));
      c010  = HALF * (DB0(i - 1, j + 1, em::bx2) + DB0(i, j + 1, em::bx2));
      c110  = HALF * (DB0(i, j + 1, em::bx2) + DB0(i + 1, j + 1, em::bx2));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[1] = c00 * (ONE - dx2_) + c10 * dx2_;
      // Bx3
      c000  = INV_4 * (DB0(i - 1, j - 1, em::bx3) + DB0(i - 1, j, em::bx3) +
                      DB0(i, j - 1, em::bx3) + DB0(i, j, em::bx3));
      c100  = INV_4 * (DB0(i, j - 1, em::bx3) + DB0(i, j, em::bx3) +
                      DB0(i + 1, j - 1, em::bx3) + DB0(i + 1, j, em::bx3));
      c010  = INV_4 * (DB0(i - 1, j, em::bx3) + DB0(i - 1, j + 1, em::bx3) +
                      DB0(i, j, em::bx3) + DB0(i, j + 1, em::bx3));
      c110  = INV_4 * (DB0(i, j, em::bx3) + DB0(i, j + 1, em::bx3) +
                      DB0(i + 1, j, em::bx3) + DB0(i + 1, j + 1, em::bx3));
      c00   = c000 * (ONE - dx1_) + c100 * dx1_;
      c10   = c010 * (ONE - dx1_) + c110 * dx1_;
      b0[2] = c00 * (ONE - dx2_) + c10 * dx2_;
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  /* ------------------------------ Photon pusher ----------------------------- */

  template <Dimension D, class M>
  Inline void Pusher_kernel<D, M>::operator()(Massless_t, index_t p) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      if (tag(p) == ParticleTag::alive) {
        // record previous coordinate
        i1_prev(p)  = i1(p);
        i2_prev(p)  = i2(p);
        dx1_prev(p) = dx1(p);
        dx2_prev(p) = dx2(p);

        coord_t<Dim2> xp { ZERO };
        vec_t<Dim3>   vp { ux1(p), ux2(p), ux3(p) };

        xp[0] = i_di_to_Xi(i1(p), dx1(p));
        xp[1] = i_di_to_Xi(i2(p), dx2(p));

        /* ----------------------------- Leapfrog pusher ---------------------------- */
        // u_i(n - 1/2) -> u_i(n + 1/2)
        vec_t<Dim3> vp_upd { ZERO };
        GeodesicMomentumPush<Massless_t>(Massless_t {}, xp, vp, vp_upd);
        // x^i(n) -> x^i(n + 1)
        coord_t<Dim2> xp_upd { ZERO };
        GeodesicCoordinatePush<Massless_t>(Massless_t {}, xp, vp_upd, xp_upd);
        // update phi
        UpdatePhi<Massless_t>(
          Massless_t {},
          { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
          vp_upd,
          phi(p));

        // update coordinate
        int      i1_, i2_;
        prtldx_t dx1_, dx2_;
        from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
        from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
        i1(p)  = i1_;
        dx1(p) = dx1_;
        i2(p)  = i2_;
        dx2(p) = dx2_;

        // update velocity
        ux1(p) = vp_upd[0];
        ux2(p) = vp_upd[1];
        ux3(p) = vp_upd[2];

        boundaryConditions(p);
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  /* ------------------------- Massive particle pusher ------------------------ */

  template <Dimension D, class M>
  Inline void Pusher_kernel<D, M>::operator()(Massive_t, index_t p) const {
    if constexpr (D == Dim1) {
      NTTError("not applicable");
    } else if constexpr (D == Dim2) {
      if (tag(p) == ParticleTag::alive) {
        // record previous coordinate
        i1_prev(p)  = i1(p);
        i2_prev(p)  = i2(p);
        dx1_prev(p) = dx1(p);
        dx2_prev(p) = dx2(p);

        coord_t<Dim2> xp { ZERO };

        xp[0] = i_di_to_Xi(i1(p), dx1(p));
        xp[1] = i_di_to_Xi(i2(p), dx2(p));

        vec_t<Dim3> Dp_cntrv { ZERO }, Bp_cntrv { ZERO }, Dp_hat { ZERO },
          Bp_hat { ZERO };
        interpolateFields(p, Dp_cntrv, Bp_cntrv);
        metric.v3_Cntrv2Hat(xp, Dp_cntrv, Dp_hat);
        metric.v3_Cntrv2Hat(xp, Bp_cntrv, Bp_hat);

        vec_t<Dim3> vp { ux1(p), ux2(p), ux3(p) };

        /* -------------------------------- Leapfrog -------------------------------- */
        /* u_i(n - 1/2) -> u*_i(n) */
        vec_t<Dim3> vp_upd { ZERO };
        EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
        /* u*_i(n) -> u**_i(n) */
        vp[0] = vp_upd[0];
        vp[1] = vp_upd[1];
        vp[2] = vp_upd[2];
        GeodesicMomentumPush<Massive_t>(Massive_t {}, xp, vp, vp_upd);
        /* u**_i(n) -> u_i(n + 1/2) */
        vp[0] = vp_upd[0];
        vp[1] = vp_upd[1];
        vp[2] = vp_upd[2];
        EMHalfPush(xp, vp, Dp_hat, Bp_hat, vp_upd);
        /* x^i(n) -> x^i(n + 1) */
        coord_t<Dim2> xp_upd { ZERO };
        GeodesicCoordinatePush<Massive_t>(Massive_t {}, xp, vp_upd, xp_upd);

        // update phi
        UpdatePhi<Massive_t>(
          Massive_t {},
          { (xp[0] + xp_upd[0]) * HALF, (xp[1] + xp_upd[1]) * HALF },
          vp_upd,
          phi(p));

        // update coordinate
        int      i1_, i2_;
        prtldx_t dx1_, dx2_;
        from_Xi_to_i_di(xp_upd[0], i1_, dx1_);
        from_Xi_to_i_di(xp_upd[1], i2_, dx2_);
        i1(p)  = i1_;
        dx1(p) = dx1_;
        i2(p)  = i2_;
        dx2(p) = dx2_;

        // update velocity
        ux1(p) = vp_upd[0];
        ux2(p) = vp_upd[1];
        ux3(p) = vp_upd[2];

        boundaryConditions(p);
      }
    } else if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

  // Boundary conditions

  template <Dimension D, class M>
  Inline void Pusher_kernel<D, M>::boundaryConditions(index_t& p) const {
    if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
      if (i1(p) < i1_absorb && is_absorb_i1min) {
        tag(p) = ParticleTag::dead;
      } else if (i1(p) >= ni1 && is_absorb_i1max) {
        tag(p) = ParticleTag::dead;
      }
    }
    if constexpr (D == Dim2 || D == Dim3) {
      if (i2(p) < 1) {
        if (is_absorb_i2min) {
          tag(p) = ParticleTag::dead;
        } else if (is_axis_i2min) {
          ux2(p) = -ux2(p);
        }
      } else if (i2(p) >= ni2 - 1) {
        if (is_absorb_i2max) {
          tag(p) = ParticleTag::dead;
        } else if (is_axis_i2min) {
          ux2(p) = -ux2(p);
        }
      }
    }
    if constexpr (D == Dim3) {
      NTTError("not implemented");
    }
  }

} // namespace ntt

#endif

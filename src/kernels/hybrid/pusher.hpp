/**
 * @file kernels/hybrid/pusher.hpp
 * @brief Non-relativistic modified-Boris ion pusher (+ fused moment deposit)
 *        for the HYBRID engine
 * @implements
 *   - kernel::hybrid::PushMode
 *   - kernel::hybrid::PusherContext
 *   - kernel::hybrid::Pusher_kernel<>
 * @namespaces:
 *   - kernel::hybrid::
 *
 * Implements the Pegasus modified-Boris integrator (Kunz, Stone & Bai 2014,
 * arXiv:1311.4865, eq. 12) for the base case sigma = varpi = 0. One push advances
 * a particle from t^(n) to t^(n+1) using fields centered at t^(n+1/2):
 *
 *   (12a)  x*       = x^(n)    + (dt/2) v^(n)         // first half drift
 *   (12b)  v^-      = v^(n)    + (dt/2) c Ec          // first electric half-kick
 *   (12c)  v^+      = v^- + rotation(Bc)              // Boris rotation (gamma = 1)
 *   (12d)  v^(n+1)  = v^+      + (dt/2) c Ec          // second electric half-kick
 *   (12e)  x^(n+1)  = x*       + (dt/2) v^(n+1)       // second half drift
 *
 * with the coupling c = (q/m), folded with omegaB0*dt/2 into `normalized_dt_half`.
 * Ec, Bc are *cell-centered*, *time-centered at n+1/2* fields (the `bckp` buffer:
 * Ec in comps 0..2, Bc in comps 3..5), interpolated at the predicted position x*
 * with the SAME shape function used by the deposit (paper §3.2).
 *
 * FUSED DEPOSIT. Immediately after advancing a particle, the kernel deposits the
 * ion number density N -> aux comp 3 and momentum density V = sum(w*m*v) ->
 * aux comps 0..2, with NO Lorentz factor (non-relativistic). This avoids a second
 * pass through the particle table and lets the transient predictor produce its
 * predicted moments WITHOUT writing the particle arrays (so no save/restore of
 * x^(n), v^(n) is needed — both pushes start from the stored state).
 *
 * The template `PushMode Mode` selects the three uses (Pegasus Fig. 2):
 *   - MomentsOnly : deposit N,V from the stored x^(n), v^(n); no push, no store.
 *                   Used once at step 0 to seed aux with N^(0), V^(0).
 *   - Predictor   : push in registers (Fig. 2 steps 7+8), deposit predicted
 *                   N', V'; do NOT write the particle arrays, no particle BCs.
 *   - Corrector   : push (Fig. 2 step 12), deposit final N^(n+1), V^(n+1), AND
 *                   write back x^(n+1), v^(n+1) (+ i/dx_prev) and apply particle BCs.
 *
 * @note NON-RELATIVISTIC. ux1/ux2/ux3 store the 3-velocity v; gamma == 1 everywhere
 *       (no 1/sqrt(1+v^2) in the rotation, no dt/gamma in the drift).
 * @note Cartesian Minkowski only. 1D gather/deposit are implemented; 2D/3D are
 *       stubs (the hybrid EMF solver is itself 1D-only for now).
 * @see PIC/hybrid/pusher.md for the full plan and the correctness traps.
 */

#ifndef KERNELS_HYBRID_PUSHER_HPP
#define KERNELS_HYBRID_PUSHER_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/containers/particles.h"
#include "kernels/particle_shapes.hpp"
#include "kernels/pushers/context.h" // kernel::sr::PusherBoundaries<D> (BC flags)

#if defined(MPI_ENABLED)
  #include "arch/mpi_tags.h"
#endif

namespace kernel::hybrid {
  using namespace ntt;

  enum class PushMode {
    MomentsOnly, // deposit only (step-0 seed); no push, no store
    Predictor,   // push in registers + deposit; no store, no BC
    Corrector    // push + deposit + store-back + BC
  };

  /**
   * @brief Lean parameter bundle for the hybrid ion pusher / deposit.
   * @note mass/charge live on the species, not on `ParticleArrays`, so they are
   *       passed in here (cf. kernel::sr::PusherContext).
   */
  struct PusherContext {
    const float  mass, charge;
    const real_t dt;
    const real_t omegaB0;       // = 1 / larmor0
    const real_t inv_n0;        // = 1 / scales.n0  (number-density normalization)
    const bool   use_weights;
    const int    ni1, ni2, ni3; // # active cells per direction (periodic wrap)

    PusherContext(float  mass,
                  float  charge,
                  real_t dt,
                  real_t omegaB0,
                  real_t inv_n0,
                  bool   use_weights,
                  int    ni1,
                  int    ni2,
                  int    ni3)
      : mass { mass }
      , charge { charge }
      , dt { dt }
      , omegaB0 { omegaB0 }
      , inv_n0 { inv_n0 }
      , use_weights { use_weights }
      , ni1 { ni1 }
      , ni2 { ni2 }
      , ni3 { ni3 } {}
  };

  /**
   * @tparam M    Metric (Cartesian Minkowski for the hybrid engine).
   * @tparam Mode MomentsOnly / Predictor / Corrector — see file header.
   */
  template <class M, PushMode Mode>
  struct Pusher_kernel {
    static constexpr auto D = M::Dim;
    static_assert(M::CoordType == Coord::Cartesian,
                  "hybrid pusher: Cartesian (Minkowski) metric only");

    // shape order and half-window; gather and deposit MUST share the cell-centering
    static constexpr unsigned short O      = SHAPE_ORDER;
    static constexpr int            window = (SHAPE_ORDER + 1) / 2;

    const PusherContext                   ctx;
    const kernel::sr::PusherBoundaries<D> bc;
    ParticleArrays                        particles;

    // read-only, cell-centered, time-centered (n+1/2) fields:
    //   Ec -> comps 0..2,  Bc -> comps 3..5   (the `bckp` buffer); unused in MomentsOnly
    const randacc_ndfield_t<D, 6> EB;
    // scatter view over `aux`:  V -> comps 0..2,  N -> comp 3
    scatter_ndfield_t<D, 6>       moments;

    const M metric;

    // coefficients (precomputed once)
    const real_t normalized_dt_half; // 1/2 * (q/m) * omegaB0 * dt  (E-kick & rotation)
    const real_t dt_half;            // 1/2 * dt                    (each half drift)

    Pusher_kernel(const PusherContext&                   pusher_ctx,
                  const kernel::sr::PusherBoundaries<D>& pusher_boundaries,
                  ParticleArrays&                        pusher_arrays,
                  const randacc_ndfield_t<D, 6>&         EB,
                  const scatter_ndfield_t<D, 6>&         moments,
                  const M&                               metric)
      : ctx { pusher_ctx }
      , bc { pusher_boundaries }
      , particles { pusher_arrays }
      , EB { EB }
      , moments { moments }
      , metric { metric }
      , normalized_dt_half { HALF * (ctx.charge / ctx.mass) * ctx.omegaB0 * ctx.dt }
      , dt_half { HALF * ctx.dt } {}

    // ........................................................................
    // main per-particle update
    // ........................................................................
    Inline void operator()(prtlidx_t p) const {
      if (particles.tag(p) != ParticleTag::alive) {
        return;
      }

      // load start-of-step state x^(n), v^(n); ux holds the 3-velocity v
      const int      i1n { particles.i1(p) };
      const prtldx_t dx1n { particles.dx1(p) };
      int            i2n { 0 }, i3n { 0 };
      prtldx_t       dx2n { ZERO }, dx3n { ZERO };
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        i2n  = particles.i2(p);
        dx2n = particles.dx2(p);
      }
      if constexpr (D == Dim::_3D) {
        i3n  = particles.i3(p);
        dx3n = particles.dx3(p);
      }

      // working state in registers (real_t offset avoids float drift across substeps)
      int    i1 { i1n }, i2 { i2n }, i3 { i3n };
      real_t dx1 { static_cast<real_t>(dx1n) };
      real_t dx2 { static_cast<real_t>(dx2n) };
      real_t dx3 { static_cast<real_t>(dx3n) };
      vec_t<Dim::_3D> v { particles.ux1(p), particles.ux2(p), particles.ux3(p) };

      if constexpr (Mode != PushMode::MomentsOnly) {
        // (12a) first half drift to the predicted position x*
        halfDrift(i1, dx1, i2, dx2, i3, dx3, v);
        // interpolate cell-centered Ec, Bc at x* (Cartesian: already in XYZ basis)
        vec_t<Dim::_3D> e0 { ZERO }, b0 { ZERO };
        gather(i1, dx1, i2, dx2, i3, dx3, e0, b0);
        // (12b-d) electric half-kick / Boris rotation / electric half-kick
        velocityPush(e0, b0, v);
        // (12e) second half drift to x^(n+1)
        halfDrift(i1, dx1, i2, dx2, i3, dx3, v);
      }

      // fused moment deposit at the advanced (or, for MomentsOnly, the stored) state
      deposit(p, i1, dx1, i2, dx2, i3, dx3, v);

      // corrector commits the accepted move -> store back + particle boundaries
      if constexpr (Mode == PushMode::Corrector) {
        if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
          particles.i1_prev(p)  = i1n;
          particles.dx1_prev(p) = dx1n;
          particles.i1(p)       = i1;
          particles.dx1(p)      = static_cast<prtldx_t>(dx1);
        }
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          particles.i2_prev(p)  = i2n;
          particles.dx2_prev(p) = dx2n;
          particles.i2(p)       = i2;
          particles.dx2(p)      = static_cast<prtldx_t>(dx2);
        }
        if constexpr (D == Dim::_3D) {
          particles.i3_prev(p)  = i3n;
          particles.dx3_prev(p) = dx3n;
          particles.i3(p)       = i3;
          particles.dx3(p)      = static_cast<prtldx_t>(dx3);
        }
        particles.ux1(p) = v[0];
        particles.ux2(p) = v[1];
        particles.ux3(p) = v[2];
        boundaryConditions(p);
      }
    }

    // ........................................................................
    // velocity update — eq. (12b)-(12d), gamma == 1
    // e0,b0 enter as the raw interpolated fields and are scaled here.
    // ........................................................................
    Inline void velocityPush(vec_t<Dim::_3D>& e0,
                             vec_t<Dim::_3D>& b0,
                             vec_t<Dim::_3D>& v) const {
      e0[0] *= normalized_dt_half;
      e0[1] *= normalized_dt_half;
      e0[2] *= normalized_dt_half;
      b0[0] *= normalized_dt_half; // t = (dt/2) c B   (NO relativistic 1/gamma!)
      b0[1] *= normalized_dt_half;
      b0[2] *= normalized_dt_half;

      // (12b) v^- = v + (dt/2) c E
      v[0] += e0[0];
      v[1] += e0[1];
      v[2] += e0[2];

      // (12c) Boris rotation: v^+ = v^- + s (v^- + v^- x t) x t,  s = 2/(1+|t|^2)
      const real_t          s { TWO / (ONE + NORM_SQR(b0[0], b0[1], b0[2])) };
      const vec_t<Dim::_3D> vp {
        (v[0] + CROSS_x1(v[0], v[1], v[2], b0[0], b0[1], b0[2])) * s,
        (v[1] + CROSS_x2(v[0], v[1], v[2], b0[0], b0[1], b0[2])) * s,
        (v[2] + CROSS_x3(v[0], v[1], v[2], b0[0], b0[1], b0[2])) * s
      };
      v[0] += CROSS_x1(vp[0], vp[1], vp[2], b0[0], b0[1], b0[2]);
      v[1] += CROSS_x2(vp[0], vp[1], vp[2], b0[0], b0[1], b0[2]);
      v[2] += CROSS_x3(vp[0], vp[1], vp[2], b0[0], b0[1], b0[2]);

      // (12d) v^(n+1) = v^+ + (dt/2) c E
      v[0] += e0[0];
      v[1] += e0[1];
      v[2] += e0[2];
    }

    // ........................................................................
    // half-step drift: x += (dt/2) v, expressed as i/dx increments.
    // transform<i,XYZ,U> converts the Cartesian velocity to a coordinate
    // (cell-fraction) velocity by dividing by the cell size dx.
    // ........................................................................
    Inline void halfDrift(int&                   i1,
                          real_t&                dx1,
                          int&                   i2,
                          real_t&                dx2,
                          int&                   i3,
                          real_t&                dx3,
                          const vec_t<Dim::_3D>& v) const {
      coord_t<D> xp { ZERO };
      xCoord(i1, dx1, i2, dx2, i3, dx3, xp);
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        dx1 += metric.template transform<1, Idx::XYZ, Idx::U>(xp, v[0]) * dt_half;
        rebucket(i1, dx1);
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        dx2 += metric.template transform<2, Idx::XYZ, Idx::U>(xp, v[1]) * dt_half;
        rebucket(i2, dx2);
      }
      if constexpr (D == Dim::_3D) {
        dx3 += metric.template transform<3, Idx::XYZ, Idx::U>(xp, v[2]) * dt_half;
        rebucket(i3, dx3);
      }
    }

    // single-cell rebucketing of (i, dx); valid for |displacement| < 1 cell (CFL)
    static Inline void rebucket(int& i, real_t& dx) {
      const int shift { static_cast<int>(dx >= ONE) - static_cast<int>(dx < ZERO) };
      i  += shift;
      dx -= static_cast<real_t>(shift);
    }

    // fill the code-coordinate position xp from the working i/dx state
    // (only consumed by metric.transform, position-independent for Minkowski)
    Inline void xCoord(int         i1,
                       real_t      dx1,
                       int         i2,
                       real_t      dx2,
                       int         i3,
                       real_t      dx3,
                       coord_t<D>& xp) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        xp[0] = static_cast<real_t>(i1) + dx1;
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        xp[1] = static_cast<real_t>(i2) + dx2;
      }
      if constexpr (D == Dim::_3D) {
        xp[2] = static_cast<real_t>(i3) + dx3;
      }
    }

    // valid cell-index guards. The field arrays span ni + 2*N_GHOSTS in each
    // direction; a particle that overshoots a boundary by more than one ghost
    // layer in a single step (e.g. bulk-flow Courant > 1, or near a reflecting
    // wall before the BC is applied) would otherwise gather/scatter past the
    // array end — an out-of-bounds atomic that page-faults on GPU. Skipping the
    // out-of-range cell keeps the kernel safe; the lost edge contribution is
    // negligible and the particle is wrapped/reflected by the boundary handler.
    Inline auto inX1(int c) const -> bool {
      return (c >= 0) && (c < ctx.ni1 + 2 * static_cast<int>(N_GHOSTS));
    }
    Inline auto inX2(int c) const -> bool {
      return (c >= 0) && (c < ctx.ni2 + 2 * static_cast<int>(N_GHOSTS));
    }
    Inline auto inX3(int c) const -> bool {
      return (c >= 0) && (c < ctx.ni3 + 2 * static_cast<int>(N_GHOSTS));
    }

    // ........................................................................
    // cell-centered field gather at (i, dx) — transpose of the moment deposit.
    // ........................................................................
    Inline void gather(int              i1,
                       real_t           dx1,
                       int              i2,
                       real_t           dx2,
                       int              i3,
                       real_t           dx3,
                       vec_t<Dim::_3D>& e0,
                       vec_t<Dim::_3D>& b0) const {
      if constexpr (D == Dim::_1D) {
        for (int di1 { -window }; di1 <= window; ++di1) {
          const int c { i1 + di1 + static_cast<int>(N_GHOSTS) };
          if (not inX1(c)) {
            continue;
          }
          const real_t S { prtl_shape::particle_shape<O>(
            math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
          e0[0] += S * EB(c, 0);
          e0[1] += S * EB(c, 1);
          e0[2] += S * EB(c, 2);
          b0[0] += S * EB(c, 3);
          b0[1] += S * EB(c, 4);
          b0[2] += S * EB(c, 5);
        }
      } else if constexpr (D == Dim::_2D) {
        for (int di2 { -window }; di2 <= window; ++di2) {
          const int c2 { i2 + di2 + static_cast<int>(N_GHOSTS) };
          if (not inX2(c2)) {
            continue;
          }
          const real_t sx2 { prtl_shape::particle_shape<O>(
            math::abs(dx2 - (static_cast<real_t>(di2) + HALF))) };
          for (int di1 { -window }; di1 <= window; ++di1) {
            const int c1 { i1 + di1 + static_cast<int>(N_GHOSTS) };
            if (not inX1(c1)) {
              continue;
            }
            const real_t S { sx2 * prtl_shape::particle_shape<O>(
                                     math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
            e0[0] += S * EB(c1, c2, 0);
            e0[1] += S * EB(c1, c2, 1);
            e0[2] += S * EB(c1, c2, 2);
            b0[0] += S * EB(c1, c2, 3);
            b0[1] += S * EB(c1, c2, 4);
            b0[2] += S * EB(c1, c2, 5);
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (int di3 { -window }; di3 <= window; ++di3) {
          const int c3 { i3 + di3 + static_cast<int>(N_GHOSTS) };
          if (not inX3(c3)) {
            continue;
          }
          const real_t sx3 { prtl_shape::particle_shape<O>(
            math::abs(dx3 - (static_cast<real_t>(di3) + HALF))) };
          for (int di2 { -window }; di2 <= window; ++di2) {
            const int c2 { i2 + di2 + static_cast<int>(N_GHOSTS) };
            if (not inX2(c2)) {
              continue;
            }
            const real_t sx23 { sx3 * prtl_shape::particle_shape<O>(
                                        math::abs(dx2 - (static_cast<real_t>(di2) + HALF))) };
            for (int di1 { -window }; di1 <= window; ++di1) {
              const int c1 { i1 + di1 + static_cast<int>(N_GHOSTS) };
              if (not inX1(c1)) {
                continue;
              }
              const real_t S { sx23 * prtl_shape::particle_shape<O>(
                                        math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
              e0[0] += S * EB(c1, c2, c3, 0);
              e0[1] += S * EB(c1, c2, c3, 1);
              e0[2] += S * EB(c1, c2, c3, 2);
              b0[0] += S * EB(c1, c2, c3, 3);
              b0[1] += S * EB(c1, c2, c3, 4);
              b0[2] += S * EB(c1, c2, c3, 5);
            }
          }
        }
      }
    }

    // ........................................................................
    // fused moment deposit:  N -> aux::3,  V = m*v -> aux::0..2 (non-relativistic).
    // Cell-centered, shape-weighted, transpose of `gather`.
    // ........................................................................
    Inline void deposit(prtlidx_t              p,
                        int                    i1,
                        real_t                 dx1,
                        int                    i2,
                        real_t                 dx2,
                        int                    i3,
                        real_t                 dx3,
                        const vec_t<Dim::_3D>& v) const {
      real_t w { ctx.inv_n0 };
      if constexpr (D == Dim::_1D) {
        w /= metric.sqrt_det_h({ static_cast<real_t>(i1) + HALF });
      } else if constexpr (D == Dim::_2D) {
        w /= metric.sqrt_det_h(
          { static_cast<real_t>(i1) + HALF, static_cast<real_t>(i2) + HALF });
      } else if constexpr (D == Dim::_3D) {
        w /= metric.sqrt_det_h({ static_cast<real_t>(i1) + HALF,
                                 static_cast<real_t>(i2) + HALF,
                                 static_cast<real_t>(i3) + HALF });
      }
      if (ctx.use_weights) {
        w *= particles.weight(p);
      }
      const real_t cN { w };
      const real_t cV0 { w * ctx.mass * v[0] };
      const real_t cV1 { w * ctx.mass * v[1] };
      const real_t cV2 { w * ctx.mass * v[2] };

      auto buff = moments.access();
      if constexpr (D == Dim::_1D) {
        for (int di1 { -window }; di1 <= window; ++di1) {
          const int c { i1 + di1 + static_cast<int>(N_GHOSTS) };
          if (not inX1(c)) {
            continue;
          }
          const real_t S { prtl_shape::particle_shape<O>(
            math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
          buff(c, 0) += cV0 * S;
          buff(c, 1) += cV1 * S;
          buff(c, 2) += cV2 * S;
          buff(c, 3) += cN * S;
        }
      } else if constexpr (D == Dim::_2D) {
        for (int di2 { -window }; di2 <= window; ++di2) {
          const int c2 { i2 + di2 + static_cast<int>(N_GHOSTS) };
          if (not inX2(c2)) {
            continue;
          }
          const real_t sx2 { prtl_shape::particle_shape<O>(
            math::abs(dx2 - (static_cast<real_t>(di2) + HALF))) };
          for (int di1 { -window }; di1 <= window; ++di1) {
            const int c1 { i1 + di1 + static_cast<int>(N_GHOSTS) };
            if (not inX1(c1)) {
              continue;
            }
            const real_t S { sx2 * prtl_shape::particle_shape<O>(
                                     math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
            buff(c1, c2, 0) += cV0 * S;
            buff(c1, c2, 1) += cV1 * S;
            buff(c1, c2, 2) += cV2 * S;
            buff(c1, c2, 3) += cN * S;
          }
        }
      } else if constexpr (D == Dim::_3D) {
        for (int di3 { -window }; di3 <= window; ++di3) {
          const int c3 { i3 + di3 + static_cast<int>(N_GHOSTS) };
          if (not inX3(c3)) {
            continue;
          }
          const real_t sx3 { prtl_shape::particle_shape<O>(
            math::abs(dx3 - (static_cast<real_t>(di3) + HALF))) };
          for (int di2 { -window }; di2 <= window; ++di2) {
            const int c2 { i2 + di2 + static_cast<int>(N_GHOSTS) };
            if (not inX2(c2)) {
              continue;
            }
            const real_t sx23 { sx3 * prtl_shape::particle_shape<O>(
                                        math::abs(dx2 - (static_cast<real_t>(di2) + HALF))) };
            for (int di1 { -window }; di1 <= window; ++di1) {
              const int c1 { i1 + di1 + static_cast<int>(N_GHOSTS) };
              if (not inX1(c1)) {
                continue;
              }
              const real_t S { sx23 * prtl_shape::particle_shape<O>(
                                        math::abs(dx1 - (static_cast<real_t>(di1) + HALF))) };
              buff(c1, c2, c3, 0) += cV0 * S;
              buff(c1, c2, c3, 1) += cV1 * S;
              buff(c1, c2, c3, 2) += cV2 * S;
              buff(c1, c2, c3, 3) += cN * S;
            }
          }
        }
      }
    }

    // ........................................................................
    // particle boundaries — periodic / absorb / reflect (Cartesian Minkowski,
    // so a reflection just negates the corresponding 3-velocity component), plus
    // the MPI leaving-direction tagging. Modeled on kernels/pushers/sr.hpp:659-814.
    // ........................................................................
    Inline void boundaryConditions(prtlidx_t p) const {
      if constexpr (D == Dim::_1D || D == Dim::_2D || D == Dim::_3D) {
        if (particles.i1(p) < 0) {
          if (bc.is_periodic_i1min) {
            particles.i1(p)      += ctx.ni1;
            particles.i1_prev(p) += ctx.ni1;
          } else if (bc.is_absorb_i1min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i1min) {
            particles.i1(p)  = 0;
            particles.dx1(p) = ONE - particles.dx1(p);
            particles.ux1(p) = -particles.ux1(p);
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
            particles.ux1(p) = -particles.ux1(p);
          }
        }
      }
      if constexpr (D == Dim::_2D || D == Dim::_3D) {
        if (particles.i2(p) < 0) {
          if (bc.is_periodic_i2min) {
            particles.i2(p)      += ctx.ni2;
            particles.i2_prev(p) += ctx.ni2;
          } else if (bc.is_absorb_i2min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i2min) {
            particles.i2(p)  = 0;
            particles.dx2(p) = ONE - particles.dx2(p);
            particles.ux2(p) = -particles.ux2(p);
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
            particles.ux2(p) = -particles.ux2(p);
          }
        }
      }
      if constexpr (D == Dim::_3D) {
        if (particles.i3(p) < 0) {
          if (bc.is_periodic_i3min) {
            particles.i3(p)      += ctx.ni3;
            particles.i3_prev(p) += ctx.ni3;
          } else if (bc.is_absorb_i3min) {
            particles.tag(p) = ParticleTag::dead;
          } else if (bc.is_reflect_i3min) {
            particles.i3(p)  = 0;
            particles.dx3(p) = ONE - particles.dx3(p);
            particles.ux3(p) = -particles.ux3(p);
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
            particles.ux3(p) = -particles.ux3(p);
          }
        }
      }
#if defined(MPI_ENABLED)
      // tag the particle with the direction it leaves the local subdomain so the
      // metadomain particle exchange ships it to the right neighbor
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
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_PUSHER_HPP

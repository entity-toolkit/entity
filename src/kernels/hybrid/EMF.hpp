#ifndef KERNELS_HYBRID_EMF_HPP
#define KERNELS_HYBRID_EMF_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include <cstdint>

namespace kernel::hybrid {

  /**
   * Fields are stored in the contravariant (Idx::U) code basis of the Minkowski
   * metric (cell size dx): components with index i <= D are physical/dx
   * (h_ii = dx^2), out-of-plane components are physical (h_ii = 1). The
   * deposited moments (PP = V, NN = N in `aux`) are physical (Cartesian XYZ
   * velocity moments; N in units of n0).
   *
   * Inputs (Bf, Bfs, Ee_in) are U-basis. The outputs follow two conventions:
   *   - Ee_out (edge E, consumed by the Faraday curl and the trapezoidal
   *     average with Ee_in): U-basis, matching em/em0 storage.
   *   - Ec_out, Bc_out (cell-centered, read raw by the particle pusher and
   *     averaged with Ec = em0::012): physical (XYZ).
   *
   * Index-space differences of U-basis fields obey
   *   delta(b_inplane) = d(B_phys),  delta(b_outofplane) = dx * d(B_phys),
   * which is the origin of the per-term dx / dx_inv factors below.
   */
  // EEONLY (requires INIT): compute and write ONLY the raw edge-E (Ee_out) --
  // used by the sub-cycled field advance, which refreshes Ee from frozen
  // moments every sub-step and needs neither Ec nor Bc there. In this mode
  // Bfs is the 3-component scratch (cur = the sub-stepped B) and Bf is unused
  // (pass cur).
  template <Dimension D, bool INIT, bool EEONLY = false>
  class EMF_kernel {
    static_assert(INIT || not EEONLY, "EEONLY requires INIT (raw writes)");
    static constexpr uint8_t N1 = (INIT || EEONLY) ? 3 : 6;
    static constexpr uint8_t N2 = (INIT && not EEONLY) ? 6 : 3;

    ndfield_t<D, 6>  PP;
    ndfield_t<D, 6>  NN;
    ndfield_t<D, 6>  Ee_in;
    ndfield_t<D, N1> Bf;
    ndfield_t<D, 6>  Ec;
    ndfield_t<D, N2> Bfs;

    ndfield_t<D, 6> Ee_out;
    ndfield_t<D, 6> Ec_out;
    ndfield_t<D, 6> Bc_out;

    const uint8_t comp_PP;
    const uint8_t comp_NN;
    const uint8_t comp_Ee_in;
    const uint8_t comp_Bf;
    const uint8_t comp_Ec;
    const uint8_t comp_Bfs;

    const uint8_t comp_Ee_out;
    const uint8_t comp_Ec_out;
    const uint8_t comp_Bc_out;

    const real_t dt;
    const real_t gamma_ad;
    const real_t theta;
    const real_t d0;
    const real_t rho0;
    const real_t dens_min;
    const real_t hall_lim;
    const real_t resist_vac;
    const real_t resist_hyper;
    const real_t dx;
    const real_t dx_inv;

  public:
    EMF_kernel(const ndfield_t<D, 6>&  PP,
               const ndfield_t<D, 6>&  NN,
               const ndfield_t<D, 6>&  Ee_in,
               const ndfield_t<D, N1>& Bf,
               const ndfield_t<D, 6>&  Ec,
               const ndfield_t<D, N2>& Bfs,
               ndfield_t<D, 6>&        Ee_out,
               ndfield_t<D, 6>&        Ec_out,
               ndfield_t<D, 6>&        Bc_out,
               uint8_t                 comp_PP,
               uint8_t                 comp_NN,
               uint8_t                 comp_Ee_in,
               uint8_t                 comp_Bf,
               uint8_t                 comp_Ec,
               uint8_t                 comp_Bfs,
               uint8_t                 comp_Ee_out,
               uint8_t                 comp_Ec_out,
               uint8_t                 comp_Bc_out,
               real_t                  dt,
               real_t                  gamma_ad,
               real_t                  theta,
               real_t                  d0,
               real_t                  rho0,
               real_t                  dens_min,
               real_t                  hall_lim,
               real_t                  resist_vac,
               real_t                  resist_hyper,
               real_t                  dx)
      : PP { PP }
      , NN { NN }
      , Ee_in { Ee_in }
      , Bf { Bf }
      , Ec { Ec }
      , Bfs { Bfs }
      , Ee_out { Ee_out }
      , Ec_out { Ec_out }
      , Bc_out { Bc_out }
      , comp_PP { comp_PP }
      , comp_NN { comp_NN }
      , comp_Ee_in { comp_Ee_in }
      , comp_Bf { comp_Bf }
      , comp_Ec { comp_Ec }
      , comp_Bfs { comp_Bfs }
      , comp_Ee_out { comp_Ee_out }
      , comp_Ec_out { comp_Ec_out }
      , comp_Bc_out { comp_Bc_out }
      , dt { dt }
      , gamma_ad { gamma_ad }
      , theta { theta }
      , d0 { d0 }
      , rho0 { rho0 }
      , dens_min { dens_min }
      , hall_lim { hall_lim }
      , resist_vac { resist_vac }
      , resist_hyper { resist_hyper }
      , dx { dx }
      , dx_inv { ONE / dx } {}

    /**
     * @brief Effective vacuum resistivity, clamped to the explicit-diffusion
     *        stability limit eta dt / dx^2 <= 0.4 for this kernel's dt (the
     *        sub-step dt inside the sub-cycled advance), so the user knob can
     *        never destabilize the update.
     */
    Inline auto eta_vac() const -> real_t {
      return math::min(resist_vac, static_cast<real_t>(0.4) * SQR(dx) / dt);
    }

    /**
     * @brief Effective HYPER-resistivity (E -= eta_H grad^2 curl B, i.e.
     *        dB/dt -= eta_H grad^4 B), applied everywhere in compute_Ee.
     *
     * The moments filter creates a spectral gap: p binomial passes attenuate
     * the deposited N, V by cos^(2p)(k dx / 2), which is ~zero at wavelengths
     * of a few cells. In that band the field solver evolves B with no kinetic
     * feedback, so the ion damping that would physically kill those modes never
     * reaches Ohm's law, and sustained compression/shear in the frozen moments
     * pumps the orphaned band by induction. Hyper-resistivity (damping ~ eta_H
     * k^4) blankets the filtered band while leaving physical scales (k dx < 0.3)
     * essentially untouched; size eta_H so eta_H (pi/dx)^4 exceeds the drive.
     *
     * Clamped to the explicit grad^4 stability limit for this kernel's dt (the
     * sub-step dt inside the sub-cycled advance): the discrete grad^2 max
     * eigenvalue is 4 D / dx^2, so eta_H (4 D / dx^2)^2 dt <= 1.
     */
    Inline auto eta_hyper() const -> real_t {
      constexpr real_t ndim = static_cast<real_t>(static_cast<dim_t>(D));
      return math::min(resist_hyper,
                       SQR(SQR(dx)) / (SQR(static_cast<real_t>(4) * ndim) * dt));
    }

    /**
     * @brief curl B at the Ee component locations, prefactored into the OUTPUT
     *        basis of compute_Ee (1D/2D: in-plane comps E/dx, out-of-plane
     *        physical; 3D: all E/dx). The resistive-vacuum E is +eta times
     *        these; the hyper-resistive E is -eta_H dx_inv^2 times their
     *        discrete Laplacian.
     */
    Inline void res_curl(const tuple_t<ncells_t, D>& i,
                         real_t&                     c0,
                         real_t&                     c1,
                         real_t&                     c2) const {
      if constexpr (D == Dim::_1D) {
        const auto i1 = i[0];
        c0            = ZERO; // (curl B)_x = 0 in 1D
        c1 = -dx_inv * (Bfs(i1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2));
        c2 = dx_inv * (Bfs(i1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1));
      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];
        c0            = SQR(dx_inv) *
             (Bfs(i1, i2, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2));
        c1 = -SQR(dx_inv) *
             (Bfs(i1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2, comp_Bfs + 2));
        c2 = (Bfs(i1, i2, comp_Bfs + 1) - Bfs(i1 - 1, i2, comp_Bfs + 1)) -
             (Bfs(i1, i2, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0));
      } else {
        const auto i1 = i[0];
        const auto i2 = i[1];
        const auto i3 = i[2];
        c0            = dx_inv * ((Bfs(i1, i2, i3, comp_Bfs + 2) -
                        Bfs(i1, i2 - 1, i3, comp_Bfs + 2)) -
                       (Bfs(i1, i2, i3, comp_Bfs + 1) -
                        Bfs(i1, i2, i3 - 1, comp_Bfs + 1)));
        c1 = dx_inv * ((Bfs(i1, i2, i3, comp_Bfs + 0) -
                        Bfs(i1, i2, i3 - 1, comp_Bfs + 0)) -
                       (Bfs(i1, i2, i3, comp_Bfs + 2) -
                        Bfs(i1 - 1, i2, i3, comp_Bfs + 2)));
        c2 = dx_inv * ((Bfs(i1, i2, i3, comp_Bfs + 1) -
                        Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) -
                       (Bfs(i1, i2, i3, comp_Bfs + 0) -
                        Bfs(i1, i2 - 1, i3, comp_Bfs + 0)));
      }
    }

    /**
     * @brief Per-cell limiter on the Hall term. The whistler speed implied at
     *        the grid cutoff, v_w = d0^2 pi |B| / (dx N), scales with the local
     *        |B|/N and can exceed the resolvable dx/dt at shock overshoots or
     *        magnetic cavities (large |B|, small N), where an explicit advance
     *        is unstable. The limiter scales the Hall term so the implied
     *        whistler Courant never exceeds `hall_lim`; it is the identity in
     *        resolvable cells. hall_lim <= 0 disables it.
     */
    Inline auto hall_limiter(const tuple_t<ncells_t, D>& i) const -> real_t {
      if (hall_lim <= ZERO) {
        return ONE;
      }
      // Worst case (max |B|, min N) over the +/-1 neighborhood the Hall stencil
      // reads: a cell-local estimate misses a sharp front where the output
      // cell's own B is small but the curl pulls in a much larger neighbor.
      real_t bsq = ZERO;
      real_t nn;
      if constexpr (D == Dim::_1D) {
        const auto i1 = i[0];
        nn            = NN(i1, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          const auto ii = i1 + static_cast<ncells_t>(di);
          // U-basis: b1 = Bx/dx, b2/b3 physical
          const real_t b = SQR(dx * Bfs(ii, comp_Bfs + 0)) +
                           SQR(Bfs(ii, comp_Bfs + 1)) +
                           SQR(Bfs(ii, comp_Bfs + 2));
          bsq = math::max(bsq, b);
          nn  = math::min(nn, NN(ii, comp_NN));
        }
      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];
        nn            = NN(i1, i2, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            const auto ii = i1 + static_cast<ncells_t>(di);
            const auto jj = i2 + static_cast<ncells_t>(dj);
            // U-basis: b1/b2 = B/dx, b3 physical
            const real_t b = SQR(dx * Bfs(ii, jj, comp_Bfs + 0)) +
                             SQR(dx * Bfs(ii, jj, comp_Bfs + 1)) +
                             SQR(Bfs(ii, jj, comp_Bfs + 2));
            bsq = math::max(bsq, b);
            nn  = math::min(nn, NN(ii, jj, comp_NN));
          }
        }
      } else {
        const auto i1 = i[0];
        const auto i2 = i[1];
        const auto i3 = i[2];
        nn            = NN(i1, i2, i3, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
              const auto ii = i1 + static_cast<ncells_t>(di);
              const auto jj = i2 + static_cast<ncells_t>(dj);
              const auto kk = i3 + static_cast<ncells_t>(dk);
              // U-basis: all comps physical/dx
              const real_t b = SQR(dx * Bfs(ii, jj, kk, comp_Bfs + 0)) +
                               SQR(dx * Bfs(ii, jj, kk, comp_Bfs + 1)) +
                               SQR(dx * Bfs(ii, jj, kk, comp_Bfs + 2));
              bsq = math::max(bsq, b);
              nn  = math::min(nn, NN(ii, jj, kk, comp_NN));
            }
          }
        }
      }
      // v_w(B, N) = d0^2 pi |B| / (dx N); reduces to the hybrid-CFL formula
      // (d0^2/rho0) pi / dx at B = B0 = 1/larmor0, N = 1
      const real_t v_w = SQR(d0) * static_cast<real_t>(constant::PI) *
                         math::sqrt(bsq) / (dx * math::max(nn, dens_min));
      // branch instead of min(1, x/y): avoids the 0/0 when B = 0
      return (v_w * dt > hall_lim * dx) ? (hall_lim * dx / (dt * v_w)) : ONE;
    }

    /**
     * @brief Vacuum cutoff: scales E by min(1, N_raw / dens_min). Below the
     *        density threshold a cell has no ions to define an E field, so E
     *        ramps to zero and B is left frozen there, instead of running the
     *        Ohm's law with a 1/dens_min-amplified right-hand side. The ramp is
     *        continuous, so cells do not flicker at the threshold and plasma-side
     *        interface cells keep evolving from their own E.
     *
     *        A hard E = 0 vacuum is only safe when the interface is quiescent.
     *        When fast plasma sweeps ALONG a vacuum boundary, masking E to zero
     *        makes an artificial tangential-E discontinuity (a surface current)
     *        and Faraday pumps B at the interface at rate ~ u/dx. The optional
     *        resistive vacuum (resist_vac) crossfades E to eta * curl B below the
     *        threshold instead, so the vacuum diffuses toward a curl-free
     *        (potential) field and tangential E has a physical continuation.
     */
    Inline auto vac_factor(real_t n_raw) const -> real_t {
      // safe for dens_min <= 0: the branch is never taken
      return (n_raw < dens_min) ? (n_raw / dens_min) : ONE;
    }

    Inline void compute_Ee(const tuple_t<ncells_t, D>& i,
                           real_t&                     E0,
                           real_t&                     E1,
                           real_t&                     E2) const {
      // Hall/whistler coefficient d_i^2 * Omega_i = d0^2/rho0. Leading minus is
      // INTENTIONAL: the (curl B) x B block below is summed into the E bracket,
      // which is then multiplied by -1/N. A positive coeff would give
      // E_Hall = -(d0^2/rho0)(curl B)xB/N -- the WRONG sign vs the generalized
      // Ohm's law (Pegasus eq. 10 / standard Hall-MHD: +(d0^2/rho0)(curl B)xB/N).
      // The minus restores the correct sign. (motional & e-pressure unaffected.)
      // hall_limiter caps the implied local whistler Courant (see its doc).
      const real_t coeff { -SQR(d0) / rho0 * hall_limiter(i) };
      if constexpr (D == Dim::_1D) {
        const auto   i1 = i[0];
        // Ee* = EMF(N^(n), P^(n), Bf*)
        const real_t N0r { NN(i1, comp_NN) };
        const real_t N12r { INV_2 * (NN(i1, comp_NN) + NN(i1 - 1, comp_NN)) };
        const real_t N0 { math::max(N0r, dens_min) };
        const real_t N1 { math::max(N12r, dens_min) };
        const real_t N2 { math::max(N12r, dens_min) };

        // 1D basis: b1 = Bx/dx, b2/b3 physical; output e1 = Ex/dx, e2/e3 physical
        E0 = dx_inv * (-Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 2) +
                       Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 1));
        E1 = -INV_4 * (Bfs(i1, comp_Bfs + 2) + Bfs(i1 - 1, comp_Bfs + 2)) *
               (PP(i1, comp_PP + 0) + PP(i1 - 1, comp_PP + 0)) +
             dx * (INV_2) * (PP(i1, comp_PP + 2) + PP(i1 - 1, comp_PP + 2)) *
               Bfs(i1, comp_Bfs + 0);
        E2 = (INV_4) * (Bfs(i1, comp_Bfs + 1) + Bfs(i1 - 1, comp_Bfs + 1)) *
               (PP(i1, comp_PP + 0) + PP(i1 - 1, comp_PP + 0)) -
             dx * INV_2 * (PP(i1, comp_PP + 1) + PP(i1 - 1, comp_PP + 1)) *
               Bfs(i1, comp_Bfs + 0);

        E0 += coeff * SQR(dx_inv) *
              (-INV_2 * (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                 Bfs(i1, comp_Bfs + 1) -
               INV_2 * (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                 Bfs(i1, comp_Bfs + 2));
        E1 += coeff * ((Bfs(i1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                       Bfs(i1, comp_Bfs + 0));
        E2 += coeff * ((Bfs(i1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                       Bfs(i1, comp_Bfs + 0));

        {
          const real_t vac0 { vac_factor(N0r) };
          const real_t vac12 { vac_factor(N12r) };
          E0 *= -vac0 / N0;
          E1 *= -vac12 / N1;
          E2 *= -vac12 / N2;
          if (resist_vac > ZERO or resist_hyper > ZERO) {
            real_t c0, c1, c2;
            res_curl(i, c0, c1, c2);
            if (resist_vac > ZERO) {
              // resistive vacuum: crossfade to E = eta * curl B below the
              // threshold (see the note at vac_factor)
              const real_t eta = eta_vac();
              E1 += (ONE - vac12) * eta * c1;
              E2 += (ONE - vac12) * eta * c2;
            }
            if (resist_hyper > ZERO) {
              // hyper-resistivity, everywhere: E -= eta_H grad^2(curl B)
              // (see the note at eta_hyper)
              real_t l0, l1, l2, r0, r1, r2;
              res_curl({ i1 - 1 }, l0, l1, l2);
              res_curl({ i1 + 1 }, r0, r1, r2);
              const real_t etah = eta_hyper() * SQR(dx_inv);
              E1 -= etah * (l1 + r1 - TWO * c1);
              E2 -= etah * (l2 + r2 - TWO * c2);
            }
          }
        }
      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];

        const real_t N0r { INV_2 *
                           (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN)) };
        const real_t N1r { INV_2 *
                           (NN(i1, i2, comp_NN) + NN(i1 - 1, i2, comp_NN)) };
        const real_t N2r { INV_4 *
                           (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN) +
                            NN(i1 - 1, i2, comp_NN) +
                            NN(i1 - 1, i2 - 1, comp_NN)) };
        const real_t N0 { math::max(N0r, dens_min) };
        const real_t N1 { math::max(N1r, dens_min) };
        const real_t N2 { math::max(N2r, dens_min) };

        // 2D basis: b1/b2 = B/dx, b3 physical; output e1/e2 = E/dx, e3 physical
        E0 = dx_inv * INV_4 *
               (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2)) *
               (PP(i1, i2, comp_PP + 1) + PP(i1, i2 - 1, comp_PP + 1)) -
             INV_2 * (PP(i1, i2, comp_PP + 2) + PP(i1, i2 - 1, comp_PP + 2)) *
               Bfs(i1, i2, comp_Bfs + 1);
        E1 = -dx_inv * INV_4 *
               (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1 - 1, i2, comp_Bfs + 2)) *
               (PP(i1, i2, comp_PP + 0) + PP(i1 - 1, i2, comp_PP + 0)) +
             (INV_2) * (PP(i1, i2, comp_PP + 2) + PP(i1 - 1, i2, comp_PP + 2)) *
               Bfs(i1, i2, comp_Bfs + 0);
        E2 = dx *
             (-INV_8 *
                (Bfs(i1, i2, comp_Bfs + 0) + Bfs(i1, i2 - 1, comp_Bfs + 0)) *
                (PP(i1, i2, comp_PP + 1) + PP(i1, i2 - 1, comp_PP + 1) +
                 PP(i1 - 1, i2, comp_PP + 1) + PP(i1 - 1, i2 - 1, comp_PP + 1)) +
              (INV_8) *
                (Bfs(i1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
                (PP(i1, i2, comp_PP + 0) + PP(i1, i2 - 1, comp_PP + 0) +
                 PP(i1 - 1, i2, comp_PP + 0) + PP(i1 - 1, i2 - 1, comp_PP + 0)));

        E0 +=
          coeff *
          (-SQR(dx_inv) * INV_8 *
             (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2)) *
             (Bfs(i1 + 1, i2, comp_Bfs + 2) + Bfs(i1 + 1, i2 - 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) +
           (INV_2) *
             (Bfs(i1 + 1, i2, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
             Bfs(i1, i2, comp_Bfs + 1));
        E1 += coeff *
              (-SQR(dx_inv) * INV_8 *
                 (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1 - 1, i2, comp_Bfs + 2)) *
                 (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2) +
                  Bfs(i1 - 1, i2 + 1, comp_Bfs + 2) -
                  Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) -
               INV_2 *
                 (Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
                  Bfs(i1, i2 + 1, comp_Bfs + 1) - Bfs(i1, i2, comp_Bfs + 1) +
                  Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) +
                  Bfs(i1 - 1, i2, comp_Bfs + 1)) *
                 Bfs(i1, i2, comp_Bfs + 0));
        E2 +=
          coeff *
          ((INV_4) * (Bfs(i1, i2, comp_Bfs + 0) + Bfs(i1, i2 - 1, comp_Bfs + 0)) *
             (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) +
           (INV_4) * (Bfs(i1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
             (Bfs(i1, i2, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)));

        {
          const real_t vac0 { vac_factor(N0r) };
          const real_t vac1 { vac_factor(N1r) };
          const real_t vac2 { vac_factor(N2r) };
          E0 *= -vac0 / N0;
          E1 *= -vac1 / N1;
          E2 *= -vac2 / N2;
          if (resist_vac > ZERO or resist_hyper > ZERO) {
            real_t c0, c1, c2;
            res_curl(i, c0, c1, c2);
            if (resist_vac > ZERO) {
              // resistive vacuum: crossfade to E = eta * curl B below the
              // threshold (see the note at vac_factor)
              const real_t eta = eta_vac();
              E0 += (ONE - vac0) * eta * c0;
              E1 += (ONE - vac1) * eta * c1;
              E2 += (ONE - vac2) * eta * c2;
            }
            if (resist_hyper > ZERO) {
              // hyper-resistivity, everywhere: E -= eta_H grad^2(curl B)
              // (see the note at eta_hyper)
              real_t w0, w1, w2, e0, e1, e2, s0, s1, s2, n0, n1, n2;
              res_curl({ i1 - 1, i2 }, w0, w1, w2);
              res_curl({ i1 + 1, i2 }, e0, e1, e2);
              res_curl({ i1, i2 - 1 }, s0, s1, s2);
              res_curl({ i1, i2 + 1 }, n0, n1, n2);
              const real_t etah = eta_hyper() * SQR(dx_inv);
              E0 -= etah * (w0 + e0 + s0 + n0 - static_cast<real_t>(4) * c0);
              E1 -= etah * (w1 + e1 + s1 + n1 - static_cast<real_t>(4) * c1);
              E2 -= etah * (w2 + e2 + s2 + n2 - static_cast<real_t>(4) * c2);
            }
          }
        }

      } else if constexpr (D == Dim::_3D) {
        const auto   i1 = i[0];
        const auto   i2 = i[1];
        const auto   i3 = i[2];
        const real_t N0r { INV_4 * (NN(i1, i2, i3, comp_NN) +
                                    NN(i1, i2, i3 - 1, comp_NN) +
                                    NN(i1, i2 - 1, i3, comp_NN) +
                                    NN(i1, i2 - 1, i3 - 1, comp_NN)) };
        const real_t N1r { INV_4 * (NN(i1, i2, i3, comp_NN) +
                                    NN(i1, i2, i3 - 1, comp_NN) +
                                    NN(i1 - 1, i2, i3, comp_NN) +
                                    NN(i1 - 1, i2, i3 - 1, comp_NN)) };
        const real_t N2r { INV_4 * (NN(i1, i2, i3, comp_NN) +
                                    NN(i1, i2 - 1, i3, comp_NN) +
                                    NN(i1 - 1, i2, i3, comp_NN) +
                                    NN(i1 - 1, i2 - 1, i3, comp_NN)) };
        const real_t N0 { math::max(N0r, dens_min) };
        const real_t N1 { math::max(N1r, dens_min) };
        const real_t N2 { math::max(N2r, dens_min) };
        E0 = -INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1, i2, i3 - 1, comp_Bfs + 1)) *
               (PP(i1, i2, i3, comp_PP + 2) + PP(i1, i2, i3 - 1, comp_PP + 2) +
                PP(i1, i2 - 1, i3, comp_PP + 2) +
                PP(i1, i2 - 1, i3 - 1, comp_PP + 2)) +
             INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1, i2 - 1, i3, comp_Bfs + 2)) *
               (PP(i1, i2, i3, comp_PP + 1) + PP(i1, i2, i3 - 1, comp_PP + 1) +
                PP(i1, i2 - 1, i3, comp_PP + 1) +
                PP(i1, i2 - 1, i3 - 1, comp_PP + 1));
        E1 = INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2, i3 - 1, comp_Bfs + 0)) *
               (PP(i1, i2, i3, comp_PP + 2) + PP(i1, i2, i3 - 1, comp_PP + 2) +
                PP(i1 - 1, i2, i3, comp_PP + 2) +
                PP(i1 - 1, i2, i3 - 1, comp_PP + 2)) -
             INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1 - 1, i2, i3, comp_Bfs + 2)) *
               (PP(i1, i2, i3, comp_PP + 0) + PP(i1, i2, i3 - 1, comp_PP + 0) +
                PP(i1 - 1, i2, i3, comp_PP + 0) +
                PP(i1 - 1, i2, i3 - 1, comp_PP + 0));
        E2 = -INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2 - 1, i3, comp_Bfs + 0)) *
               (PP(i1, i2, i3, comp_PP + 1) + PP(i1, i2 - 1, i3, comp_PP + 1) +
                PP(i1 - 1, i2, i3, comp_PP + 1) +
                PP(i1 - 1, i2 - 1, i3, comp_PP + 1)) +
             INV_8 *
               (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) *
               (PP(i1, i2, i3, comp_PP + 0) + PP(i1, i2 - 1, i3, comp_PP + 0) +
                PP(i1 - 1, i2, i3, comp_PP + 0) +
                PP(i1 - 1, i2 - 1, i3, comp_PP + 0));

        E0 +=
          coeff *
          (INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1, i2, i3 - 1, comp_Bfs + 1)) *
             (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
              Bfs(i1 + 1, i2, i3 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 - 1, i3, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 - 1, i3 - 1, comp_Bfs + 0) +
              Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2, i3 - 1, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2, i3, comp_Bfs + 1) -
              Bfs(i1 + 1, i2, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3 - 1, comp_Bfs + 1)) +
           INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1, i2 - 1, i3, comp_Bfs + 2)) *
             (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) -
              Bfs(i1 + 1, i2, i3 - 1, comp_Bfs + 0) +
              Bfs(i1 + 1, i2 - 1, i3, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 - 1, i3 - 1, comp_Bfs + 0) +
              Bfs(i1, i2, i3, comp_Bfs + 0) - Bfs(i1, i2, i3 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 - 1, i3, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2, i3, comp_Bfs + 2) -
              Bfs(i1 + 1, i2 - 1, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3, comp_Bfs + 2)));
        E1 +=
          coeff *
          (-INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2, i3 - 1, comp_Bfs + 0)) *
             (Bfs(i1, i2 + 1, i3, comp_Bfs + 0) +
              Bfs(i1, i2 + 1, i3 - 1, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3 - 1, comp_Bfs + 0) -
              Bfs(i1, i2 + 1, i3, comp_Bfs + 1) -
              Bfs(i1, i2 + 1, i3 - 1, comp_Bfs + 1) -
              Bfs(i1, i2, i3, comp_Bfs + 1) - Bfs(i1, i2, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, i3, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3 - 1, comp_Bfs + 1)) +
           INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1 - 1, i2, i3, comp_Bfs + 2)) *
             (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) -
              Bfs(i1, i2 + 1, i3 - 1, comp_Bfs + 1) +
              Bfs(i1, i2, i3, comp_Bfs + 1) - Bfs(i1, i2, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, i3, comp_Bfs + 1) -
              Bfs(i1 - 1, i2 + 1, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3, comp_Bfs + 1) -
              Bfs(i1 - 1, i2, i3 - 1, comp_Bfs + 1) -
              Bfs(i1, i2 + 1, i3, comp_Bfs + 2) + Bfs(i1, i2 - 1, i3, comp_Bfs + 2) -
              Bfs(i1 - 1, i2 + 1, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3, comp_Bfs + 2)));
        E2 +=
          coeff *
          (-INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2 - 1, i3, comp_Bfs + 0)) *
             (Bfs(i1, i2, i3 + 1, comp_Bfs + 0) - Bfs(i1, i2, i3 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 - 1, i3 + 1, comp_Bfs + 0) -
              Bfs(i1, i2 - 1, i3 - 1, comp_Bfs + 0) -
              Bfs(i1, i2, i3 + 1, comp_Bfs + 2) - Bfs(i1, i2, i3, comp_Bfs + 2) -
              Bfs(i1, i2 - 1, i3 + 1, comp_Bfs + 2) -
              Bfs(i1, i2 - 1, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2, i3 + 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3 + 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3, comp_Bfs + 2)) -
           INV_8 *
             (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) *
             (Bfs(i1, i2, i3 + 1, comp_Bfs + 1) - Bfs(i1, i2, i3 - 1, comp_Bfs + 1) +
              Bfs(i1 - 1, i2, i3 + 1, comp_Bfs + 1) -
              Bfs(i1 - 1, i2, i3 - 1, comp_Bfs + 1) -
              Bfs(i1, i2, i3 + 1, comp_Bfs + 2) - Bfs(i1, i2, i3, comp_Bfs + 2) +
              Bfs(i1, i2 - 1, i3 + 1, comp_Bfs + 2) +
              Bfs(i1, i2 - 1, i3, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, i3 + 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, i3, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3 + 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 - 1, i3, comp_Bfs + 2)));

        {
          const real_t vac0 { vac_factor(N0r) };
          const real_t vac1 { vac_factor(N1r) };
          const real_t vac2 { vac_factor(N2r) };
          E0 *= -vac0 / N0;
          E1 *= -vac1 / N1;
          E2 *= -vac2 / N2;
          if (resist_vac > ZERO or resist_hyper > ZERO) {
            real_t c0, c1, c2;
            res_curl(i, c0, c1, c2);
            if (resist_vac > ZERO) {
              // resistive vacuum: crossfade to E = eta * curl B below the
              // threshold (see the note at vac_factor)
              const real_t eta = eta_vac();
              E0 += (ONE - vac0) * eta * c0;
              E1 += (ONE - vac1) * eta * c1;
              E2 += (ONE - vac2) * eta * c2;
            }
            if (resist_hyper > ZERO) {
              // hyper-resistivity, everywhere: E -= eta_H grad^2(curl B)
              // (see the note at eta_hyper)
              real_t a0, a1, a2, lap0, lap1, lap2;
              lap0 = -static_cast<real_t>(6) * c0;
              lap1 = -static_cast<real_t>(6) * c1;
              lap2 = -static_cast<real_t>(6) * c2;
              res_curl({ i1 - 1, i2, i3 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              res_curl({ i1 + 1, i2, i3 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              res_curl({ i1, i2 - 1, i3 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              res_curl({ i1, i2 + 1, i3 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              res_curl({ i1, i2, i3 - 1 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              res_curl({ i1, i2, i3 + 1 }, a0, a1, a2);
              lap0 += a0; lap1 += a1; lap2 += a2;
              const real_t etah = eta_hyper() * SQR(dx_inv);
              E0 -= etah * lap0;
              E1 -= etah * lap1;
              E2 -= etah * lap2;
            }
          }
        }
      }
    }

    Inline void compute_Ec(const tuple_t<ncells_t, D>& i,
                           real_t&                     E0,
                           real_t&                     E1,
                           real_t&                     E2) const {
      const real_t coeff_1 { rho0 * gamma_ad * theta };
      // Hall coefficient, same sign convention as `coeff` in compute_Ee above
      // (leading minus corrects the (curl B) x B sign; see note there).
      // hall_limiter caps the implied local whistler Courant (see its doc).
      const real_t coeff_2 { -SQR(d0) / rho0 * hall_limiter(i) };
      if constexpr (D == Dim::_1D) {
        const auto i1 = i[0];

        // floored cell-centered density: the pusher reads Ec raw, so an
        // unfloored empty cell would divide by zero into the particle push
        const real_t Ncr { NN(i1, comp_NN) };
        const real_t Nc { math::max(Ncr, dens_min) };

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        // output is PHYSICAL (pusher reads bckp raw); b1 = Bx/dx, b2/b3 physical
        E0 = -Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 2) +
             Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 1);
        E1 = dx *
               (INV_2 * Bfs(i1 + 1, comp_Bfs + 0) +
                INV_2 * Bfs(i1, comp_Bfs + 0)) *
               PP(i1, comp_PP + 2) -
             Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 0);
        E2 = -dx *
               (INV_2 * Bfs(i1 + 1, comp_Bfs + 0) +
                INV_2 * Bfs(i1, comp_Bfs + 0)) *
               PP(i1, comp_PP + 1) +
             Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 0);

        E0 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv * INV_2 *
              (NN(i1 + 1, comp_NN) - NN(i1 - 1, comp_NN));

        E0 += coeff_2 * dx_inv *
              (-INV_2 * (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                 Bfs(i1, comp_Bfs + 1) -
               INV_2 * (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                 Bfs(i1, comp_Bfs + 2));
        E1 += coeff_2 * INV_4 *
              (Bfs(i1 + 1, comp_Bfs + 0) + Bfs(i1, comp_Bfs + 0)) *
              (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1));
        E2 += coeff_2 * INV_4 *
              (Bfs(i1 + 1, comp_Bfs + 0) + Bfs(i1, comp_Bfs + 0)) *
              (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2));

        E0 *= -vac_factor(Ncr) / Nc;
        E1 *= -vac_factor(Ncr) / Nc;
        E2 *= -vac_factor(Ncr) / Nc;

      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];

        // floored cell-centered density (see 1D note above)
        const real_t Ncr { NN(i1, i2, comp_NN) };
        const real_t Nc { math::max(Ncr, dens_min) };

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        // output is PHYSICAL (pusher reads bckp raw); b1/b2 = B/dx, b3 physical
        E0 = -dx * INV_2 *
               (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
               PP(i1, i2, comp_PP + 2) +
             Bfs(i1, i2, comp_Bfs + 2) * PP(i1, i2, comp_PP + 1);
        E1 = dx * INV_2 *
               (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
               PP(i1, i2, comp_PP + 2) -
             Bfs(i1, i2, comp_Bfs + 2) * PP(i1, i2, comp_PP + 0);
        E2 = dx *
             (-INV_2 *
                (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
                PP(i1, i2, comp_PP + 1) +
              INV_2 *
                (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
                PP(i1, i2, comp_PP + 0));

        E0 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv * INV_2 *
              (NN(i1 + 1, i2, comp_NN) - NN(i1 - 1, i2, comp_NN));
        E1 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv * INV_2 *
              (NN(i1, i2 + 1, comp_NN) - NN(i1, i2 - 1, comp_NN));
        E0 +=
          coeff_2 *
          (dx * (INV_8) *
             (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
             (Bfs(i1 + 1, i2 + 1, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 + 1, comp_Bfs + 1) - Bfs(i1 + 1, i2, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) -
           dx_inv * INV_2 *
             (Bfs(i1 + 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2, comp_Bfs + 2)) *
             Bfs(i1, i2, comp_Bfs + 2));
        E1 +=
          coeff_2 *
          (-dx * INV_8 *
             (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
             (Bfs(i1 + 1, i2 + 1, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 + 1, comp_Bfs + 1) - Bfs(i1 + 1, i2, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) -
           dx_inv * INV_2 *
             (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2)) *
             Bfs(i1, i2, comp_Bfs + 2));
        E2 += coeff_2 *
              ((INV_4) *
                 (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
                 (Bfs(i1 + 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2, comp_Bfs + 2)) +
               (INV_4) *
                 (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
                 (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2)));

        E0 *= -vac_factor(Ncr) / Nc;
        E1 *= -vac_factor(Ncr) / Nc;
        E2 *= -vac_factor(Ncr) / Nc;
      } else if constexpr (D == Dim::_3D) {
        const auto i1 = i[0];
        const auto i2 = i[1];
        const auto i3 = i[2];

        // floored cell-centered density (see 1D note above)
        const real_t Ncr { NN(i1, i2, i3, comp_NN) };
        const real_t Nc { math::max(Ncr, dens_min) };
        // Ee* = EMF(N^(n), P^(n), Bf*)

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        // output is PHYSICAL (pusher reads bckp raw); all b comps = B/dx in 3D
        E0 = dx * (-INV_2 *
                     (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                      Bfs(i1, i2, i3, comp_Bfs + 1)) *
                     PP(i1, i2, i3, comp_PP + 2) +
                   INV_2 *
                     (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                      Bfs(i1, i2, i3, comp_Bfs + 2)) *
                     PP(i1, i2, i3, comp_PP + 1));
        E1 = dx * (INV_2 *
                     (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                      Bfs(i1, i2, i3, comp_Bfs + 0)) *
                     PP(i1, i2, i3, comp_PP + 2) -
                   INV_2 *
                     (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                      Bfs(i1, i2, i3, comp_Bfs + 2)) *
                     PP(i1, i2, i3, comp_PP + 0));
        E2 = dx * (-INV_2 *
                     (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                      Bfs(i1, i2, i3, comp_Bfs + 0)) *
                     PP(i1, i2, i3, comp_PP + 1) +
                   INV_2 *
                     (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                      Bfs(i1, i2, i3, comp_Bfs + 1)) *
                     PP(i1, i2, i3, comp_PP + 0));

        E0 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv *
              INV_2 * (NN(i1 + 1, i2, i3, comp_NN) - NN(i1 - 1, i2, i3, comp_NN));
        E1 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv *
              INV_2 * (NN(i1, i2 + 1, i3, comp_NN) - NN(i1, i2 - 1, i3, comp_NN));
        E2 += coeff_1 * math::pow(Nc, gamma_ad - ONE) * dx_inv *
              INV_2 * (NN(i1, i2, i3 + 1, comp_NN) - NN(i1, i2, i3 - 1, comp_NN));

        E0 += coeff_2 * dx * (INV_8 *
                           (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1, i2, i3, comp_Bfs + 1)) *
                           (Bfs(i1 + 1, i2 + 1, i3, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2 - 1, i3, comp_Bfs + 0) +
                            Bfs(i1, i2 + 1, i3, comp_Bfs + 0) -
                            Bfs(i1, i2 - 1, i3, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2 + 1, i3, comp_Bfs + 1) -
                            Bfs(i1 + 1, i2, i3, comp_Bfs + 1) +
                            Bfs(i1 - 1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) +
                         INV_8 *
                           (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2, i3, comp_Bfs + 2)) *
                           (Bfs(i1 + 1, i2, i3 + 1, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2, i3 - 1, comp_Bfs + 0) +
                            Bfs(i1, i2, i3 + 1, comp_Bfs + 0) -
                            Bfs(i1, i2, i3 - 1, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2, i3 + 1, comp_Bfs + 2) -
                            Bfs(i1 + 1, i2, i3, comp_Bfs + 2) +
                            Bfs(i1 - 1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1 - 1, i2, i3, comp_Bfs + 2)));
        E1 += coeff_2 * dx * (-INV_8 *
                           (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                            Bfs(i1, i2, i3, comp_Bfs + 0)) *
                           (Bfs(i1 + 1, i2 + 1, i3, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2 - 1, i3, comp_Bfs + 0) +
                            Bfs(i1, i2 + 1, i3, comp_Bfs + 0) -
                            Bfs(i1, i2 - 1, i3, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2 + 1, i3, comp_Bfs + 1) -
                            Bfs(i1 + 1, i2, i3, comp_Bfs + 1) +
                            Bfs(i1 - 1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) +
                         INV_8 *
                           (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2, i3, comp_Bfs + 2)) *
                           (Bfs(i1, i2 + 1, i3 + 1, comp_Bfs + 1) -
                            Bfs(i1, i2 + 1, i3 - 1, comp_Bfs + 1) +
                            Bfs(i1, i2, i3 + 1, comp_Bfs + 1) -
                            Bfs(i1, i2, i3 - 1, comp_Bfs + 1) -
                            Bfs(i1, i2 + 1, i3 + 1, comp_Bfs + 2) -
                            Bfs(i1, i2 + 1, i3, comp_Bfs + 2) +
                            Bfs(i1, i2 - 1, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2 - 1, i3, comp_Bfs + 2)));
        E2 += coeff_2 * dx * (-INV_8 *
                           (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                            Bfs(i1, i2, i3, comp_Bfs + 0)) *
                           (Bfs(i1 + 1, i2, i3 + 1, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2, i3 - 1, comp_Bfs + 0) +
                            Bfs(i1, i2, i3 + 1, comp_Bfs + 0) -
                            Bfs(i1, i2, i3 - 1, comp_Bfs + 0) -
                            Bfs(i1 + 1, i2, i3 + 1, comp_Bfs + 2) -
                            Bfs(i1 + 1, i2, i3, comp_Bfs + 2) +
                            Bfs(i1 - 1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1 - 1, i2, i3, comp_Bfs + 2)) -
                         INV_8 *
                           (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1, i2, i3, comp_Bfs + 1)) *
                           (Bfs(i1, i2 + 1, i3 + 1, comp_Bfs + 1) -
                            Bfs(i1, i2 + 1, i3 - 1, comp_Bfs + 1) +
                            Bfs(i1, i2, i3 + 1, comp_Bfs + 1) -
                            Bfs(i1, i2, i3 - 1, comp_Bfs + 1) -
                            Bfs(i1, i2 + 1, i3 + 1, comp_Bfs + 2) -
                            Bfs(i1, i2 + 1, i3, comp_Bfs + 2) +
                            Bfs(i1, i2 - 1, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2 - 1, i3, comp_Bfs + 2)));

        E0 *= -vac_factor(Ncr) / Nc;
        E1 *= -vac_factor(Ncr) / Nc;
        E2 *= -vac_factor(Ncr) / Nc;
      }
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if constexpr (INIT) {
          /// terms of Ee and Ec
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1 }, Eestar0, Eestar1, Eestar2);

          Ee_out(i1, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, comp_Ee_out + 2) = Eestar2;

          if constexpr (not EEONLY) {
            real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
            compute_Ec({ i1 }, Ecstar0, Ecstar1, Ecstar2);

            Ec_out(i1, comp_Ec_out + 0) = Ecstar0;
            Ec_out(i1, comp_Ec_out + 1) = Ecstar1;
            Ec_out(i1, comp_Ec_out + 2) = Ecstar2;
          }
        } else {
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1 }, Eestar0, Eestar1, Eestar2);
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1 }, Ecstar0, Ecstar1, Ecstar2);

          // Ee' = 0.5 * (Ee* + Ee^(n))
          Ee_out(i1, comp_Ee_out + 0) = INV_2 *
                                        (Eestar0 + Ee_in(i1, comp_Ee_in + 0));
          Ee_out(i1, comp_Ee_out + 1) = INV_2 *
                                        (Eestar1 + Ee_in(i1, comp_Ee_in + 1));
          Ee_out(i1, comp_Ee_out + 2) = INV_2 *
                                        (Eestar2 + Ee_in(i1, comp_Ee_in + 2));

          // Ec' = 0.5 * (Ec* + Ec^(n))
          Ec_out(i1, comp_Ec_out + 0) = INV_2 * (Ecstar0 + Ec(i1, comp_Ec + 0));
          Ec_out(i1, comp_Ec_out + 1) = INV_2 * (Ecstar1 + Ec(i1, comp_Ec + 1));
          Ec_out(i1, comp_Ec_out + 2) = INV_2 * (Ecstar2 + Ec(i1, comp_Ec + 2));

          // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
          // (PHYSICAL output: b1 is stored as Bx/dx in 1D, b2/b3 physical)
          Bc_out(i1, comp_Bc_out + 0) = dx * INV_2 *
                                        ((INV_2 * Bf(i1 + 1, comp_Bf + 0) +
                                          INV_2 * Bf(i1, comp_Bf + 0)) +
                                         (INV_2 * Bfs(i1 + 1, comp_Bfs + 0) +
                                          INV_2 * Bfs(i1, comp_Bfs + 0)));

          Bc_out(i1, comp_Bc_out + 1) = INV_2 * (Bf(i1, comp_Bf + 1) +
                                                 Bfs(i1, comp_Bfs + 1));
          Bc_out(i1, comp_Bc_out + 2) = INV_2 * (Bf(i1, comp_Bf + 2) +
                                                 Bfs(i1, comp_Bfs + 2));
        }
      } else {
        raise::KernelError(HERE, "EMF_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (INIT) {
          /// terms of Ee and Ec
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1, i2 }, Eestar0, Eestar1, Eestar2);

          Ee_out(i1, i2, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, i2, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, i2, comp_Ee_out + 2) = Eestar2;

          if constexpr (not EEONLY) {
            real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
            compute_Ec({ i1, i2 }, Ecstar0, Ecstar1, Ecstar2);

            Ec_out(i1, i2, comp_Ec_out + 0) = Ecstar0;
            Ec_out(i1, i2, comp_Ec_out + 1) = Ecstar1;
            Ec_out(i1, i2, comp_Ec_out + 2) = Ecstar2;
          }
        } else {
          // Ee* = EMF(N^(n), P^(n), Bf*)
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1, i2 }, Eestar0, Eestar1, Eestar2);
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1, i2 }, Ecstar0, Ecstar1, Ecstar2);

          // Ee' = 0.5 * (Ee* + Ee^(n))
          Ee_out(i1, i2, comp_Ee_out + 0) = INV_2 *
                                            (Eestar0 +
                                             Ee_in(i1, i2, comp_Ee_in + 0));
          Ee_out(i1, i2, comp_Ee_out + 1) = INV_2 *
                                            (Eestar1 +
                                             Ee_in(i1, i2, comp_Ee_in + 1));
          Ee_out(i1, i2, comp_Ee_out + 2) = INV_2 *
                                            (Eestar2 +
                                             Ee_in(i1, i2, comp_Ee_in + 2));

          // Ec' = 0.5 * (Ec* + Ec^(n))
          Ec_out(i1, i2, comp_Ec_out + 0) = INV_2 *
                                            (Ecstar0 + Ec(i1, i2, comp_Ec + 0));
          Ec_out(i1, i2, comp_Ec_out + 1) = INV_2 *
                                            (Ecstar1 + Ec(i1, i2, comp_Ec + 1));
          Ec_out(i1, i2, comp_Ec_out + 2) = INV_2 *
                                            (Ecstar2 + Ec(i1, i2, comp_Ec + 2));

          // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
          // (PHYSICAL output: b1/b2 are stored as B/dx in 2D, b3 physical)
          Bc_out(i1, i2, comp_Bc_out + 0) = dx * INV_4 *
                                            (Bf(i1 + 1, i2, comp_Bf + 0) +
                                             Bf(i1, i2, comp_Bf + 0) +
                                             Bfs(i1 + 1, i2, comp_Bfs + 0) +
                                             Bfs(i1, i2, comp_Bfs + 0));

          Bc_out(i1, i2, comp_Bc_out + 1) = dx * INV_4 *
                                            (Bf(i1, i2 + 1, comp_Bf + 1) +
                                             Bf(i1, i2, comp_Bf + 1) +
                                             Bfs(i1, i2 + 1, comp_Bfs + 1) +
                                             Bfs(i1, i2, comp_Bfs + 1));
          Bc_out(i1, i2, comp_Bc_out + 2) = INV_2 * (Bf(i1, i2, comp_Bf + 2) +
                                                     Bfs(i1, i2, comp_Bfs + 2));
        }
      } else {
        raise::KernelError(HERE, "EMF_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (INIT) {
          /// terms of Ee and Ec
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1, i2, i3 }, Eestar0, Eestar1, Eestar2);

          Ee_out(i1, i2, i3, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, i2, i3, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, i2, i3, comp_Ee_out + 2) = Eestar2;

          if constexpr (not EEONLY) {
            real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
            compute_Ec({ i1, i2, i3 }, Ecstar0, Ecstar1, Ecstar2);

            Ec_out(i1, i2, i3, comp_Ec_out + 0) = Ecstar0;
            Ec_out(i1, i2, i3, comp_Ec_out + 1) = Ecstar1;
            Ec_out(i1, i2, i3, comp_Ec_out + 2) = Ecstar2;
          }
        } else {
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1, i2, i3 }, Eestar0, Eestar1, Eestar2);
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1, i2, i3 }, Ecstar0, Ecstar1, Ecstar2);

          // Ee' = 0.5 * (Ee* + Ee^(n))
          Ee_out(i1, i2, i3, comp_Ee_out + 0) = INV_2 *
                                                (Eestar0 +
                                                 Ee_in(i1, i2, i3, comp_Ee_in + 0));
          Ee_out(i1, i2, i3, comp_Ee_out + 1) = INV_2 *
                                                (Eestar1 +
                                                 Ee_in(i1, i2, i3, comp_Ee_in + 1));
          Ee_out(i1, i2, i3, comp_Ee_out + 2) = INV_2 *
                                                (Eestar2 +
                                                 Ee_in(i1, i2, i3, comp_Ee_in + 2));

          // Ec' = 0.5 * (Ec* + Ec^(n))
          Ec_out(i1, i2, i3, comp_Ec_out + 0) = INV_2 *
                                                (Ecstar0 +
                                                 Ec(i1, i2, i3, comp_Ec + 0));
          Ec_out(i1, i2, i3, comp_Ec_out + 1) = INV_2 *
                                                (Ecstar1 +
                                                 Ec(i1, i2, i3, comp_Ec + 1));
          Ec_out(i1, i2, i3, comp_Ec_out + 2) = INV_2 *
                                                (Ecstar2 +
                                                 Ec(i1, i2, i3, comp_Ec + 2));

          // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
          // (PHYSICAL output: all b comps are stored as B/dx in 3D)
          Bc_out(i1, i2, i3, comp_Bc_out + 0) = dx * INV_4 *
                                                (Bf(i1 + 1, i2, i3, comp_Bf + 0) +
                                                 Bf(i1, i2, i3, comp_Bf + 0) +
                                                 Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                                                 Bfs(i1, i2, i3, comp_Bfs + 0));

          Bc_out(i1, i2, i3, comp_Bc_out + 1) = dx * INV_4 *
                                                (Bf(i1, i2 + 1, i3, comp_Bf + 1) +
                                                 Bf(i1, i2, i3, comp_Bf + 1) +
                                                 Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                                                 Bfs(i1, i2, i3, comp_Bfs + 1));
          Bc_out(i1, i2, i3, comp_Bc_out + 2) = dx * INV_4 *
                                                (Bf(i1, i2, i3 + 1, comp_Bf + 2) +
                                                 Bf(i1, i2, i3, comp_Bf + 2) +
                                                 Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                                                 Bfs(i1, i2, i3, comp_Bfs + 2));
        }
      } else {
        raise::KernelError(HERE, "EMF_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_EMF_HPP

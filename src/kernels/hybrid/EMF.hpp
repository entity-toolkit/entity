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
  template <Dimension D, bool INIT>
  class EMF_kernel {
    static constexpr uint8_t N1 = INIT ? 3 : 6;
    static constexpr uint8_t N2 = INIT ? 6 : 3;

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
      , dx { dx }
      , dx_inv { ONE / dx } {}

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
      const real_t coeff { -SQR(d0) / rho0 };
      if constexpr (D == Dim::_1D) {
        const auto   i1 = i[0];
        // Ee* = EMF(N^(n), P^(n), Bf*)
        const real_t N0 { math::max(NN(i1, comp_NN), dens_min) };
        const real_t N1 {
          math::max(INV_2 * (NN(i1, comp_NN) + NN(i1 - 1, comp_NN)), dens_min)
        };
        const real_t N2 {
          math::max(INV_2 * (NN(i1, comp_NN) + NN(i1 - 1, comp_NN)), dens_min)
        };

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

        E0 *= -ONE / N0;
        E1 *= -ONE / N1;
        E2 *= -ONE / N2;
      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];

        const real_t N0 { math::max(
          INV_2 * (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN)),
          dens_min) };
        const real_t N1 { math::max(
          INV_2 * (NN(i1, i2, comp_NN) + NN(i1 - 1, i2, comp_NN)),
          dens_min) };
        const real_t N2 { math::max(
          INV_4 * (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN) +
                   NN(i1 - 1, i2, comp_NN) + NN(i1 - 1, i2 - 1, comp_NN)),
          dens_min) };

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

        E0 *= -ONE / N0;
        E1 *= -ONE / N1;
        E2 *= -ONE / N2;

      } else if constexpr (D == Dim::_3D) {
        const auto   i1 = i[0];
        const auto   i2 = i[1];
        const auto   i3 = i[2];
        const real_t N0 { math::max(
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2, i3 - 1, comp_NN) +
                   NN(i1, i2 - 1, i3, comp_NN) + NN(i1, i2 - 1, i3 - 1, comp_NN)),
          dens_min) };
        const real_t N1 { math::max(
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2, i3 - 1, comp_NN) +
                   NN(i1 - 1, i2, i3, comp_NN) + NN(i1 - 1, i2, i3 - 1, comp_NN)),
          dens_min) };
        const real_t N2 { math::max(
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2 - 1, i3, comp_NN) +
                   NN(i1 - 1, i2, i3, comp_NN) + NN(i1 - 1, i2 - 1, i3, comp_NN)),
          dens_min) };
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

        E0 *= -ONE / N0;
        E1 *= -ONE / N1;
        E2 *= -ONE / N2;
      }
    }

    Inline void compute_Ec(const tuple_t<ncells_t, D>& i,
                           real_t&                     E0,
                           real_t&                     E1,
                           real_t&                     E2) const {
      const real_t coeff_1 { rho0 * gamma_ad * theta };
      // Hall coefficient, same sign convention as `coeff` in compute_Ee above
      // (leading minus corrects the (curl B) x B sign; see note there).
      const real_t coeff_2 { -SQR(d0) / rho0 };
      if constexpr (D == Dim::_1D) {
        const auto i1 = i[0];

        // floored cell-centered density: the pusher reads Ec, so an empty cell
        // here would inject Inf into the particle push (NaN cascade)
        const real_t Nc { math::max(NN(i1, comp_NN), dens_min) };

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

        E0 *= -ONE / Nc;
        E1 *= -ONE / Nc;
        E2 *= -ONE / Nc;

      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];

        // floored cell-centered density (see 1D note above)
        const real_t Nc { math::max(NN(i1, i2, comp_NN), dens_min) };

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

        E0 *= -ONE / Nc;
        E1 *= -ONE / Nc;
        E2 *= -ONE / Nc;
      } else if constexpr (D == Dim::_3D) {
        const auto i1 = i[0];
        const auto i2 = i[1];
        const auto i3 = i[2];

        // floored cell-centered density (see 1D note above)
        const real_t Nc { math::max(NN(i1, i2, i3, comp_NN), dens_min) };
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

        E0 *= -ONE / Nc;
        E1 *= -ONE / Nc;
        E2 *= -ONE / Nc;
      }
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if constexpr (INIT) {
          /// terms of Ee and Ec
          real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
          compute_Ee({ i1 }, Eestar0, Eestar1, Eestar2);
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1 }, Ecstar0, Ecstar1, Ecstar2);

          Ee_out(i1, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, comp_Ee_out + 2) = Eestar2;

          Ec_out(i1, comp_Ec_out + 0) = Ecstar0;
          Ec_out(i1, comp_Ec_out + 1) = Ecstar1;
          Ec_out(i1, comp_Ec_out + 2) = Ecstar2;
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
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1, i2 }, Ecstar0, Ecstar1, Ecstar2);

          Ee_out(i1, i2, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, i2, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, i2, comp_Ee_out + 2) = Eestar2;

          Ec_out(i1, i2, comp_Ec_out + 0) = Ecstar0;
          Ec_out(i1, i2, comp_Ec_out + 1) = Ecstar1;
          Ec_out(i1, i2, comp_Ec_out + 2) = Ecstar2;
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
          real_t Ecstar0 { ZERO }, Ecstar1 { ZERO }, Ecstar2 { ZERO };
          compute_Ec({ i1, i2, i3 }, Ecstar0, Ecstar1, Ecstar2);

          Ee_out(i1, i2, i3, comp_Ee_out + 0) = Eestar0;
          Ee_out(i1, i2, i3, comp_Ee_out + 1) = Eestar1;
          Ee_out(i1, i2, i3, comp_Ee_out + 2) = Eestar2;

          Ec_out(i1, i2, i3, comp_Ec_out + 0) = Ecstar0;
          Ec_out(i1, i2, i3, comp_Ec_out + 1) = Ecstar1;
          Ec_out(i1, i2, i3, comp_Ec_out + 2) = Ecstar2;
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

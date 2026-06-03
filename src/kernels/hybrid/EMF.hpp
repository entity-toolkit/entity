#ifndef KERNELS_HYBRID_EMF_HPP
#define KERNELS_HYBRID_EMF_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::hybrid {

  template <Dimension D, uint8_t N>
  class EMF_kernel {
    ndfield_t<D, 6> PP;
    ndfield_t<D, 6> NN;
    ndfield_t<D, 6> Ee_in;
    ndfield_t<D, 6> Bf;
    ndfield_t<D, 6> Ec;
    ndfield_t<D, 3> Bfs;

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

  public:
    EMF_kernel(const ndfield_t<D, 6>& PP,
               const ndfield_t<D, 6>& NN,
               const ndfield_t<D, 6>& Ee_in,
               const ndfield_t<D, 6>& Bf,
               const ndfield_t<D, 6>& Ec,
               const ndfield_t<D, 3>& Bfs,
               ndfield_t<D, 6>&       Ee_out,
               ndfield_t<D, 6>&       Ec_out,
               ndfield_t<D, 6>&       Bc_out,
               uint8_t                comp_PP,
               uint8_t                comp_NN,
               uint8_t                comp_Ee_in,
               uint8_t                comp_Bf,
               uint8_t                comp_Ec,
               uint8_t                comp_Bfs,
               uint8_t                comp_Ee_out,
               uint8_t                comp_Ec_out,
               uint8_t                comp_Bc_out,
               real_t                 dt,
               real_t                 gamma_ad,
               real_t                 theta,
               real_t                 d0,
               real_t                 rho0)
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
      , rho0 { rho0 } {}

    Inline void compute_Ee(tuple_t<ncells_t, D>& i,
                           real_t&               E1,
                           real_t&               E2,
                           real_t&               E3) const {
      if constexpr (D == Dim::_1D) {
        const auto   i1 = i[0];
        // Ee* = EMF(N^(n), P^(n), Bf*)
        const real_t N0 { NN(i1, comp_NN) };
        const real_t N1 { INV_2 * (NN(i1, comp_NN) + NN(i1 - 1, comp_NN)) };
        const real_t N2 { INV_2 * (NN(i1, comp_NN) + NN(i1 - 1, comp_NN)) };

        E1 = -Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 2) +
             Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 1);
        E2 = -INV_4 * (Bfs(i1, comp_Bfs + 2) + Bfs(i1 - 1, comp_Bfs + 2)) *
               (PP(i1, comp_PP + 0) + PP(i1 - 1, comp_PP + 0)) +
             (INV_2) * (PP(i1, comp_PP + 2) + PP(i1 - 1, comp_PP + 2)) *
               Bfs(i1, comp_Bfs + 0);
        E3 = (INV_4) * (Bfs(i1, comp_Bfs + 1) + Bfs(i1 - 1, comp_Bfs + 1)) *
               (PP(i1, comp_PP + 0) + PP(i1 - 1, comp_PP + 0)) -
             INV_2 * (PP(i1, comp_PP + 1) + PP(i1 - 1, comp_PP + 1)) *
               Bfs(i1, comp_Bfs + 0);

        const real_t coeff_1 { rho0 * gamma_ad * theta };
        const real_t coeff_2 { SQR(d0) / rho0 };
        E1 += coeff_2 *
              (-INV_2 * (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                 Bfs(i1, comp_Bfs + 1) -
               INV_2 * (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                 Bfs(i1, comp_Bfs + 2));
        E2 += coeff_2 * ((Bfs(i1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                         Bfs(i1, comp_Bfs + 0));
        E3 += coeff_2 * ((Bfs(i1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                         Bfs(i1, comp_Bfs + 0));

        E1 *= -ONE / N0;
        E2 *= -ONE / N1;
        E3 *= -ONE / N2;
      } else if constexpr (D == Dim::_2D) {
        const auto i1 = i[0];
        const auto i2 = i[1];
        ...
      }
    }

    Inline void operator()(STEP0, cellidx_t i1) const {
      ...; /// terms of Ee and Ec
      real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
      compute_Ee(i1, Eestar0, Eestar1, Eestar2);

      Ee_out(i1, ...) = ...;
      Ec_out(i1, ...) = ...;
    };

    Inline void operator()(NOTSTEP0, cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {

        real_t Eestar0 { ZERO }, Eestar1 { ZERO }, Eestar2 { ZERO };
        compute_Ee({ i1 }, Eestar0, Eestar1, Eestar2);

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        real_t Ecstar0 = -Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 2) +
                         Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 1);
        real_t Ecstar1 = (INV_2 * Bfs(i1 + 1, comp_Bfs + 0) +
                          INV_2 * Bfs(i1, comp_Bfs + 0)) *
                           PP(i1, comp_PP + 2) -
                         Bfs(i1, comp_Bfs + 2) * PP(i1, comp_PP + 0);
        real_t Ecstar2 = -(INV_2 * Bfs(i1 + 1, comp_Bfs + 0) +
                           INV_2 * Bfs(i1, comp_Bfs + 0)) *
                           PP(i1, comp_PP + 1) +
                         Bfs(i1, comp_Bfs + 1) * PP(i1, comp_PP + 0);

        Ecstar0 += coeff_1 * math::pow(NN(i1, comp_NN), gamma_ad - ONE) *
                   INV_2 * (NN(i1 + 1, comp_NN) - NN(i1 - 1, comp_NN));

        Ecstar0 += coeff_2 *
                   (-INV_2 *
                      (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1)) *
                      Bfs(i1, comp_Bfs + 1) -
                    INV_2 *
                      (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2)) *
                      Bfs(i1, comp_Bfs + 2));
        Ecstar1 += coeff_2 * INV_4 *
                   (Bfs(i1 + 1, comp_Bfs + 0) + Bfs(i1, comp_Bfs + 0)) *
                   (Bfs(i1 + 1, comp_Bfs + 1) - Bfs(i1 - 1, comp_Bfs + 1));
        Ecstar2 += coeff_2 * INV_4 *
                   (Bfs(i1 + 1, comp_Bfs + 0) + Bfs(i1, comp_Bfs + 0)) *
                   (Bfs(i1 + 1, comp_Bfs + 2) - Bfs(i1 - 1, comp_Bfs + 2));

        Ecstar0 *= -ONE / NN(i1, comp_NN);
        Ecstar1 *= -ONE / NN(i1, comp_NN);
        Ecstar2 *= -ONE / NN(i1, comp_NN);

        // Ee' = 0.5 * (Ee* + Ee^(n))
        Ee_out(i1, comp_Ee_out + 0) = INV_2 *
                                      (Eestar0 + Ee_in(i1, comp_Ee_in + 0));
        Ee_out(i1, comp_Ee_out + 1) = INV_2 *
                                      (Eestar1 + Ee_in(i1, comp_Ee_in + 1));
        Ee_out(i1, comp_Ee_out + 2) = INV_2 *
                                      (Eestar2 + Ee_in(i1, comp_Ee_in + 2));

        // Ec' = 0.5 * (Ec* + Ec^(n))
        Ec_out(i1, comp_Ee_out + 0) = INV_2 * (Ecstar0 + Ec(i1, comp_Ec + 0));
        Ec_out(i1, comp_Ee_out + 1) = INV_2 * (Ecstar1 + Ec(i1, comp_Ec + 1));
        Ec_out(i1, comp_Ee_out + 2) = INV_2 * (Ecstar2 + Ec(i1, comp_Ec + 2));

        // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
        Bc_out(i1, comp_Bc_out + 0) = INV_2 * ((INV_2 * Bf(i1 + 1, comp_Bf + 0) +
                                                INV_2 * Bf(i1, comp_Bf + 0)) +
                                               (INV_2 * Bfs(i1 + 1, comp_Bf + 0) +
                                                INV_2 * Bfs(i1, comp_Bf + 0)));

        Bc_out(i1, comp_Bc_out + 1) = INV_2 * (Bf(i1, comp_Bf + 1) +
                                               Bfs(i1, comp_Bf + 1));
        Bc_out(i1, comp_Bc_out + 2) = INV_2 * (Bf(i1, comp_Bf + 2) +
                                               Bfs(i1, comp_Bf + 2));

      } else {
        raise::KernelError(HERE, "EMF_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        // Ee* = EMF(N^(n), P^(n), Bf*)
        const real_t N0 { INV_2 * (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN)) };
        const real_t N1 { INV_2 * (NN(i1, i2, comp_NN) + NN(i1 - 1, i2, comp_NN)) };
        const real_t N2 { INV_4 * (NN(i1, i2, comp_NN) + NN(i1, i2 - 1, comp_NN) +
                                   NN(i1 - 1, i2, comp_NN) +
                                   NN(i1 - 1, i2 - 1, comp_NN)) };
        real_t Eestar0 =
          INV_4 * (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2)) *
            (PP(i1, i2, comp_PP + 1) + PP(i1, i2 - 1, comp_PP + 1)) -
          INV_2 * (PP(i1, i2, comp_PP + 2) + PP(i1, i2 - 1, comp_PP + 2)) *
            Bfs(i1, i2, comp_Bfs + 1);
        real_t Eestar1 =
          -INV_4 * (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1 - 1, i2, comp_Bfs + 2)) *
            (PP(i1, i2, comp_PP + 0) + PP(i1 - 1, i2, comp_PP + 0)) +
          (INV_2) * (PP(i1, i2, comp_PP + 2) + PP(i1 - 1, i2, comp_PP + 2)) *
            Bfs(i1, i2, comp_Bfs + 0);
        real_t Eestar2 =
          -INV_8 * (Bfs(i1, i2, comp_Bfs + 0) + Bfs(i1, i2 - 1, comp_Bfs + 0)) *
            (PP(i1, i2, comp_PP + 1) + PP(i1, i2 - 1, comp_PP + 1) +
             PP(i1 - 1, i2, comp_PP + 1) + PP(i1 - 1, i2 - 1, comp_PP + 1)) +
          (INV_8) * (Bfs(i1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
            (PP(i1, i2, comp_PP + 0) + PP(i1, i2 - 1, comp_PP + 0) +
             PP(i1 - 1, i2, comp_PP + 0) + PP(i1 - 1, i2 - 1, comp_PP + 0));

        const real_t coeff_1 { rho0 * gamma_ad * theta };
        const real_t coeff_2 { SQR(d0) / rho0 };
        Eestar0 +=
          coeff_2 *
          (-INV_8 * (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2)) *
             (Bfs(i1 + 1, i2, comp_Bfs + 2) + Bfs(i1 + 1, i2 - 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) +
           (INV_2) *
             (Bfs(i1 + 1, i2, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
             Bfs(i1, i2, comp_Bfs + 1));
        Eestar1 +=
          coeff_2 *
          (-INV_8 * (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1 - 1, i2, comp_Bfs + 2)) *
             (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) -
           INV_2 *
             (Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1, i2 + 1, comp_Bfs + 1) - Bfs(i1, i2, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
             Bfs(i1, i2, comp_Bfs + 0));
        Eestar2 +=
          coeff_2 *
          ((INV_4) * (Bfs(i1, i2, comp_Bfs + 0) + Bfs(i1, i2 - 1, comp_Bfs + 0)) *
             (Bfs(i1, i2, comp_Bfs + 2) + Bfs(i1, i2 - 1, comp_Bfs + 2) -
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)) +
           (INV_4) * (Bfs(i1, i2, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) *
             (Bfs(i1, i2, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2) +
              Bfs(i1 - 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2 - 1, comp_Bfs + 2)));

        Eestar0 *= -ONE / N0;
        Eestar1 *= -ONE / N1;
        Eestar2 *= -ONE / N2;

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        real_t Ecstar0 = -INV_2 *
                           (Bfs(i1, i2 + 1, comp_Bfs + 1) +
                            Bfs(i1, i2, comp_Bfs + 1)) *
                           PP(i1, i2, comp_PP + 2) +
                         Bfs(i1, i2, comp_Bfs + 2) * PP(i1, i2, comp_PP + 1);
        real_t Ecstar1 = INV_2 *
                           (Bfs(i1 + 1, i2, comp_Bfs + 0) +
                            Bfs(i1, i2, comp_Bfs + 0)) *
                           PP(i1, i2, comp_PP + 2) -
                         Bfs(i1, i2, comp_Bfs + 2) * PP(i1, i2, comp_PP + 0);
        real_t Ecstar2 =
          -INV_2 * (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
            PP(i1, i2, comp_PP + 1) +
          INV_2 * (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
            PP(i1, i2, comp_PP + 0);

        Ecstar0 += coeff_1 * math::pow(NN(i1, i2, comp_NN), gamma_ad - ONE) *
                   INV_2 * (NN(i1 + 1, i2, comp_NN) - NN(i1 - 1, i2, comp_NN));
        Ecstar1 += coeff_1 * math::pow(NN(i1, i2, comp_NN), gamma_ad - ONE) *
                   INV_2 * (NN(i1, i2 + 1, comp_NN) - NN(i1, i2 - 1, comp_NN));
        Ecstar0 +=
          coeff_2 *
          ((INV_8) * (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
             (Bfs(i1 + 1, i2 + 1, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 + 1, comp_Bfs + 1) - Bfs(i1 + 1, i2, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) -
           INV_2 * (Bfs(i1 + 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2, comp_Bfs + 2)) *
             Bfs(i1, i2, comp_Bfs + 2));
        Ecstar1 +=
          coeff_2 *
          (-INV_8 * (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
             (Bfs(i1 + 1, i2 + 1, comp_Bfs + 0) - Bfs(i1 + 1, i2 - 1, comp_Bfs + 0) +
              Bfs(i1, i2 + 1, comp_Bfs + 0) - Bfs(i1, i2 - 1, comp_Bfs + 0) -
              Bfs(i1 + 1, i2 + 1, comp_Bfs + 1) - Bfs(i1 + 1, i2, comp_Bfs + 1) +
              Bfs(i1 - 1, i2 + 1, comp_Bfs + 1) + Bfs(i1 - 1, i2, comp_Bfs + 1)) -
           INV_2 * (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2)) *
             Bfs(i1, i2, comp_Bfs + 2));
        Ecstar2 +=
          coeff_2 *
          ((INV_4) * (Bfs(i1 + 1, i2, comp_Bfs + 0) + Bfs(i1, i2, comp_Bfs + 0)) *
             (Bfs(i1 + 1, i2, comp_Bfs + 2) - Bfs(i1 - 1, i2, comp_Bfs + 2)) +
           (INV_4) * (Bfs(i1, i2 + 1, comp_Bfs + 1) + Bfs(i1, i2, comp_Bfs + 1)) *
             (Bfs(i1, i2 + 1, comp_Bfs + 2) - Bfs(i1, i2 - 1, comp_Bfs + 2)));

        Ecstar0 *= -ONE / NN(i1, i2, comp_NN);
        Ecstar1 *= -ONE / NN(i1, i2, comp_NN);
        Ecstar2 *= -ONE / NN(i1, i2, comp_NN);

        // Ee' = 0.5 * (Ee* + Ee^(n))
        Ee_out(i1, i2, comp_Ee_out + 0) = INV_2 * (Eestar0 +
                                                   Ee_in(i1, i2, comp_Ee_in + 0));
        Ee_out(i1, i2, comp_Ee_out + 1) = INV_2 * (Eestar1 +
                                                   Ee_in(i1, i2, comp_Ee_in + 1));
        Ee_out(i1, i2, comp_Ee_out + 2) = INV_2 * (Eestar2 +
                                                   Ee_in(i1, i2, comp_Ee_in + 2));

        // Ec' = 0.5 * (Ec* + Ec^(n))
        Ec_out(i1, i2, comp_Ee_out + 0) = INV_2 *
                                          (Ecstar0 + Ec(i1, i2, comp_Ec + 0));
        Ec_out(i1, i2, comp_Ee_out + 1) = INV_2 *
                                          (Ecstar1 + Ec(i1, i2, comp_Ec + 1));
        Ec_out(i1, i2, comp_Ee_out + 2) = INV_2 *
                                          (Ecstar2 + Ec(i1, i2, comp_Ec + 2));

        // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
        Bc_out(i1, i2, comp_Bc_out + 0) = INV_4 * (Bf(i1 + 1, i2, comp_Bf + 0) +
                                                   Bf(i1, i2, comp_Bf + 0) +
                                                   Bfs(i1 + 1, i2, comp_Bfs + 0) +
                                                   Bfs(i1, i2, comp_Bfs + 0));

        Bc_out(i1, i2, comp_Bc_out + 1) = INV_4 * (Bf(i1, i2 + 1, comp_Bf + 1) +
                                                   Bf(i1, i2, comp_Bf + 1) +
                                                   Bfs(i1, i2 + 1, comp_Bfs + 1) +
                                                   Bfs(i1, i2, comp_Bfs + 1));
        Bc_out(i1, i2, comp_Bc_out + 2) = INV_2 * (Bf(i1, i2, comp_Bf + 2) +
                                                   Bfs(i1, i2, comp_Bfs + 2));
      } else {
        raise::KernelError(HERE, "EMF_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        // Ee* = EMF(N^(n), P^(n), Bf*)
        const real_t N0 {
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2, i3 - 1, comp_NN) +
                   NN(i1, i2 - 1, i3, comp_NN) + NN(i1, i2 - 1, i3 - 1, comp_NN))
        };
        const real_t N1 {
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2, i3 - 1, comp_NN) +
                   NN(i1 - 1, i2, i3, comp_NN) + NN(i1 - 1, i2, i3 - 1, comp_NN))
        };
        const real_t N2 {
          INV_4 * (NN(i1, i2, i3, comp_NN) + NN(i1, i2 - 1, i3, comp_NN) +
                   NN(i1 - 1, i2, i3, comp_NN) + NN(i1 - 1, i2 - 1, i3, comp_NN))
        };
        real_t Eestar0 =
          -INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1, i2, i3 - 1, comp_Bfs + 1)) *
            (PP(i1, i2, i3, comp_PP + 2) + PP(i1, i2, i3 - 1, comp_PP + 2) +
             PP(i1, i2 - 1, i3, comp_PP + 2) +
             PP(i1, i2 - 1, i3 - 1, comp_PP + 2)) +
          INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1, i2 - 1, i3, comp_Bfs + 2)) *
            (PP(i1, i2, i3, comp_PP + 1) + PP(i1, i2, i3 - 1, comp_PP + 1) +
             PP(i1, i2 - 1, i3, comp_PP + 1) + PP(i1, i2 - 1, i3 - 1, comp_PP + 1));
        real_t Eestar1 =
          INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2, i3 - 1, comp_Bfs + 0)) *
            (PP(i1, i2, i3, comp_PP + 2) + PP(i1, i2, i3 - 1, comp_PP + 2) +
             PP(i1 - 1, i2, i3, comp_PP + 2) +
             PP(i1 - 1, i2, i3 - 1, comp_PP + 2)) -
          INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 2) + Bfs(i1 - 1, i2, i3, comp_Bfs + 2)) *
            (PP(i1, i2, i3, comp_PP + 0) + PP(i1, i2, i3 - 1, comp_PP + 0) +
             PP(i1 - 1, i2, i3, comp_PP + 0) + PP(i1 - 1, i2, i3 - 1, comp_PP + 0));
        real_t Eestar2 =
          -INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 0) + Bfs(i1, i2 - 1, i3, comp_Bfs + 0)) *
            (PP(i1, i2, i3, comp_PP + 1) + PP(i1, i2 - 1, i3, comp_PP + 1) +
             PP(i1 - 1, i2, i3, comp_PP + 1) +
             PP(i1 - 1, i2 - 1, i3, comp_PP + 1)) +
          INV_8 *
            (Bfs(i1, i2, i3, comp_Bfs + 1) + Bfs(i1 - 1, i2, i3, comp_Bfs + 1)) *
            (PP(i1, i2, i3, comp_PP + 0) + PP(i1, i2 - 1, i3, comp_PP + 0) +
             PP(i1 - 1, i2, i3, comp_PP + 0) + PP(i1 - 1, i2 - 1, i3, comp_PP + 0));

        const real_t coeff_1 { rho0 * gamma_ad * theta };
        const real_t coeff_2 { SQR(d0) / rho0 };
        Eestar0 +=
          coeff_2 *
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
        Eestar1 +=
          coeff_2 *
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
        Eestar2 +=
          coeff_2 *
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

        Eestar0 *= -ONE / N0;
        Eestar1 *= -ONE / N1;
        Eestar2 *= -ONE / N2;

        // Ec* = EMF(N^(n), P^(n), Bc*), where Bc* = interpolate Bf*
        real_t Ecstar0 = -INV_2 *
                           (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1, i2, i3, comp_Bfs + 1)) *
                           PP(i1, i2, i3, comp_PP + 2) +
                         INV_2 *
                           (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2, i3, comp_Bfs + 2)) *
                           PP(i1, i2, i3, comp_PP + 1);
        real_t Ecstar1 = INV_2 *
                           (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                            Bfs(i1, i2, i3, comp_Bfs + 0)) *
                           PP(i1, i2, i3, comp_PP + 2) -
                         INV_2 *
                           (Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                            Bfs(i1, i2, i3, comp_Bfs + 2)) *
                           PP(i1, i2, i3, comp_PP + 0);
        real_t Ecstar2 = -INV_2 *
                           (Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                            Bfs(i1, i2, i3, comp_Bfs + 0)) *
                           PP(i1, i2, i3, comp_PP + 1) +
                         INV_2 *
                           (Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                            Bfs(i1, i2, i3, comp_Bfs + 1)) *
                           PP(i1, i2, i3, comp_PP + 0);

        Ecstar0 += coeff_1 *
                   math::pow(NN(i1, i2, i3, comp_NN), gamma_ad - ONE) * INV_2 *
                   (NN(i1 + 1, i2, i3, comp_NN) - NN(i1 - 1, i2, i3, comp_NN));
        Ecstar1 += coeff_1 *
                   math::pow(NN(i1, i2, i3, comp_NN), gamma_ad - ONE) * INV_2 *
                   (NN(i1, i2 + 1, i3, comp_NN) - NN(i1, i2 - 1, i3, comp_NN));
        Ecstar2 += coeff_1 *
                   math::pow(NN(i1, i2, i3, comp_NN), gamma_ad - ONE) * INV_2 *
                   (NN(i1, i2, i3 + 1, comp_NN) - NN(i1, i2, i3 - 1, comp_NN));

        Ecstar0 += coeff_2 * (INV_8 *
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
        Ecstar1 += coeff_2 * (-INV_8 *
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
        Ecstar2 += coeff_2 * (-INV_8 *
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

        Ecstar0 *= -ONE / NN(i1, i2, i3, comp_NN);
        Ecstar1 *= -ONE / NN(i1, i2, i3, comp_NN);
        Ecstar2 *= -ONE / NN(i1, i2, i3, comp_NN);

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
        Ec_out(i1, i2, i3, comp_Ee_out + 0) = INV_2 *
                                              (Ecstar0 +
                                               Ec(i1, i2, i3, comp_Ec + 0));
        Ec_out(i1, i2, i3, comp_Ee_out + 1) = INV_2 *
                                              (Ecstar1 +
                                               Ec(i1, i2, i3, comp_Ec + 1));
        Ec_out(i1, i2, i3, comp_Ee_out + 2) = INV_2 *
                                              (Ecstar2 +
                                               Ec(i1, i2, i3, comp_Ec + 2));

        // Bc' = 0.5 * (Bc* + Bc^(n)), where Bc* = interpolate Bf*
        Bc_out(i1, i2, i3, comp_Bc_out + 0) = INV_4 *
                                              (Bf(i1 + 1, i2, i3, comp_Bf + 0) +
                                               Bf(i1, i2, i3, comp_Bf + 0) +
                                               Bfs(i1 + 1, i2, i3, comp_Bfs + 0) +
                                               Bfs(i1, i2, i3, comp_Bfs + 0));

        Bc_out(i1, i2, i3, comp_Bc_out + 1) = INV_4 *
                                              (Bf(i1, i2 + 1, i3, comp_Bf + 1) +
                                               Bf(i1, i2, i3, comp_Bf + 1) +
                                               Bfs(i1, i2 + 1, i3, comp_Bfs + 1) +
                                               Bfs(i1, i2, i3, comp_Bfs + 1));
        Bc_out(i1, i2, i3, comp_Bc_out + 2) = INV_4 *
                                              (Bf(i1, i2, i3 + 1, comp_Bf + 2) +
                                               Bf(i1, i2, i3, comp_Bf + 2) +
                                               Bfs(i1, i2, i3 + 1, comp_Bfs + 2) +
                                               Bfs(i1, i2, i3, comp_Bfs + 2));
      } else {
        raise::KernelError(HERE, "EMF_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_EMF_HPP

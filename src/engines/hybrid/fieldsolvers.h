#ifndef ENGINES_HYBRID_FIELDSOLVERS_H
#define ENGINES_HYBRID_FIELDSOLVERS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/param_container.h"

#include "metrics/minkowski.h"

#include "engines/hybrid/fields_bcs.h"
#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"
#include "kernels/hybrid/EMF.hpp"
#include "kernels/hybrid/faraday.hpp"
#include "kernels/hybrid/subcycle.hpp"

#include <cmath>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

namespace ntt {
  namespace hybrid {

    enum class faraday : uint8_t {
      push1,
      push2,
      push3,
    };

    enum class emf : uint8_t {
      push0,
      push12,
    };

    template <Dimension D>
    void Faraday(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                 const prm::Parameters&                           engine_params,
                 const faraday&                                   flag) {
      const auto dt = engine_params.get<real_t>("dt");
      const auto dx = domain.mesh.metric.get_dx();
      if (flag == faraday::push1) {
        Kokkos::parallel_for(
          "FaradayPush1",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 3>(domain.fields.em,
                                               domain.fields.em,
                                               domain.fields.cur,
                                               0,
                                               3,
                                               0,
                                               dt,
                                               dx));
      } else if (flag == faraday::push2) {
        Kokkos::parallel_for(
          "FaradayPush2",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 3>(domain.fields.em0,
                                               domain.fields.em,
                                               domain.fields.cur,
                                               3,
                                               3,
                                               0,
                                               dt,
                                               dx));
      } else if (flag == faraday::push3) {
        Kokkos::parallel_for(
          "FaradayPush3",
          domain.mesh.rangeActiveCells(),
          kernel::hybrid::Faraday_kernel<D, 6>(domain.fields.em0,
                                               domain.fields.em,
                                               domain.fields.em,
                                               3,
                                               3,
                                               3,
                                               dt,
                                               dx));
      } else {
        raise::Error("Wrong option for `flag`", HERE);
      }
    }

    /**
     * @brief Active cells extended by one ghost layer on every upper side whose
     *        ghosts are refreshed by the halo exchange (periodic or
     *        inter-domain); physical-boundary sides are not extended.
     *
     * The Faraday update of the active cells reads the edge-E only at i and
     * i + 1 (forward differences). The EMF stencil reads B, N, V at most one
     * cell above its own index, so the edge-E of the first upper ghost layer
     * needs only the two ghost layers of its inputs; every input there is a
     * ghost copy of the neighbor's data, so it equals the value the neighbor
     * computes in its first active layer, and E needs no halo exchange before
     * that update. This requires the B, N, V ghosts to hold the current values
     * of the cells they copy whenever the EMF runs (hybrid.hpp refills the B
     * ghosts after a problem-generator post-step).
     */
    template <Dimension D>
    auto ActiveWithUpperHalo(const Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain)
      -> range_t<D> {
      static_assert(N_GHOSTS >= 2, "ActiveWithUpperHalo needs two ghost layers");
      const auto  flds_bc = domain.mesh.flds_bc();
      const auto  hi      = [&](unsigned short d) -> ncells_t {
        const auto bc = flds_bc[d].second;
        return ((bc == FldsBC::PERIODIC) or (bc == FldsBC::SYNC)) ? 1u : 0u;
      };
      if constexpr (D == Dim::_1D) {
        return CreateRangePolicy<Dim::_1D>({ domain.mesh.i_min(in::x1) },
                                           { domain.mesh.i_max(in::x1) + hi(0) });
      } else if constexpr (D == Dim::_2D) {
        return CreateRangePolicy<Dim::_2D>(
          { domain.mesh.i_min(in::x1), domain.mesh.i_min(in::x2) },
          { domain.mesh.i_max(in::x1) + hi(0), domain.mesh.i_max(in::x2) + hi(1) });
      } else {
        return CreateRangePolicy<Dim::_3D>(
          { domain.mesh.i_min(in::x1),
            domain.mesh.i_min(in::x2),
            domain.mesh.i_min(in::x3) },
          { domain.mesh.i_max(in::x1) + hi(0),
            domain.mesh.i_max(in::x2) + hi(1),
            domain.mesh.i_max(in::x3) + hi(2) });
      }
    }

    template <Dimension D>
    void EMF(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
             const prm::Parameters&                           engine_params,
             const SimulationParams&                          params,
             const emf&                                       flag) {
      const auto dt       = engine_params.get<real_t>("dt");
      const auto gamma_ad = params.get<real_t>("hybrid.gamma_ad");
      const auto theta    = params.get<real_t>("hybrid.theta0");
      const auto d0       = params.get<real_t>("scales.skindepth0");
      const auto rho0     = params.get<real_t>("scales.larmor0");
      const auto dens_min = params.get<real_t>("hybrid.dens_min");
      // The Hall limiter bounds the whistler Courant number of the explicit field
      // advance that consumes this E. With hybrid.subcycle, B is advanced by
      // SubcycledFaraday (which applies the limiter at the sub-step dt) and these
      // full-step fields only feed the particle pusher and the output, so the
      // limiter is off here.
      const auto hall_lim = params.get<bool>("hybrid.subcycle")
                              ? ZERO
                              : params.get<real_t>("hybrid.hall_lim");
      const auto res_vac  = params.get<real_t>("hybrid.resist_vac");
      const auto res_hyp  = params.get<real_t>("hybrid.resist_hyper");
      const auto res_uni  = params.get<real_t>("hybrid.resist");
      const auto dx       = domain.mesh.metric.get_dx();
      // the edge-E outputs are read on the active cells and the first upper
      // ghost layer, so both are computed here (see ActiveWithUpperHalo)
      const auto range    = ActiveWithUpperHalo(domain);
      if (flag == emf::push0) {
        // clang-format off
        Kokkos::parallel_for(
          "EMFPush",
          range,
          kernel::hybrid::EMF_kernel<D, true>(domain.fields.aux,  // P
                                              domain.fields.aux,  // N
                                              domain.fields.em0,  // Ee_in
                                              domain.fields.cur,  // Bf
                                              domain.fields.bckp, // Ec
                                              domain.fields.em,   // Bfs
                                              domain.fields.em,   // Ee_out
                                              domain.fields.em0,  // Ec_out
                                              domain.fields.bckp, // Bc_out
                                              0, 3, 3,            // P, N, Ee_in
                                              0, 0, 3,            // Bf, Ec, Bfs
                                              0, 0, 3,            // Ee_out, Ec_out, Bc_out
                                              dt, gamma_ad, theta, d0, rho0, dens_min, hall_lim, res_vac, res_hyp, res_uni, dx));
        // clang-format on
      } else {
        // clang-format off
        Kokkos::parallel_for(
          "EMFPush",
          range,
          kernel::hybrid::EMF_kernel<D, false>(domain.fields.aux,  // P
                                               domain.fields.aux,  // N
                                               domain.fields.em,   // Ee_in
                                               domain.fields.em,   // Bf
                                               domain.fields.em0,  // Ec
                                               domain.fields.cur,  // Bfs
                                               domain.fields.em0,  // Ee_out
                                               domain.fields.bckp, // Ec_out
                                               domain.fields.bckp, // Bc_out
                                               0, 3, 0,            // P, N, Ee_in
                                               3, 0, 0,            // Bf, Ec, Bfs
                                               3, 0, 3,            // Ee_out, Ec_out, Bc_out
                                               dt, gamma_ad, theta, d0, rho0, dens_min, hall_lim, res_vac, res_hyp, res_uni, dx));
        // clang-format on
      }
    }

    /**
     * @brief Sub-cycled (Pegasus-style) magnetic-field advance: integrates
     *        dB/dt = -curl E(B) over one full timestep with the Ohm's-law E
     *        refreshed from the (frozen) deposited moments every sub-step, so
     *        the whistler-stiff part of hybrid Ohm's law is integrated at its
     *        own CFL instead of the particle dt.
     *
     * Reads Bf^(n) from em::345 and the frozen moments (V in comps 0..2, N in
     * comp 3, valid ghosts) from `moments`; leaves the
     * advanced B (with valid ghosts) in `cur` -- the same contract as the
     * single-Euler Faraday push #1/#2 it replaces. The third (accepting)
     * Faraday push is redundant here: this advance already integrates the full
     * interval, so the caller accepts `cur` as B^(n+1) (AcceptSubcycledB).
     *
     * Number of sub-steps m: adaptive from the GLOBAL max of the local whistler
     * speed v_w = (d0^2/rho0) pi |B| / (dx max(N, dens_min)), targeting a per-sub-step
     * whistler Courant `hybrid.subcycle_courant`, capped at `hybrid.subcycle_max`
     * (beyond the cap the per-cell Hall limiter, which here receives the
     * SUB-step dt, takes over). The max is MPI-allreduced so every rank runs the
     * same m and the halo exchanges inside the loop pair up.
     *
     * Each sub-step is SSP-RK3 (Shu-Osher): 2-stage schemes are marginally
     * unstable for purely oscillatory (whistler) modes (|G| = 1 + (w dt)^4/8),
     * while RK3 is damped up to w dt < sqrt(3).
     *
     * Scratch usage: cur = running B (in place), buff = sub-step base for the
     * RK3 combinations, em0::345 = stage E. Combines run over the FULL extent so
     * valid ghosts stay valid. Each stage computes E on the active cells and the
     * upper halo layer (no E exchange, see ActiveWithUpperHalo), then the B
     * halo exchange + wall conditions run after the Faraday update.
     */
    template <Dimension D>
    auto SubcycledFaraday(Metadomain<SimEngine::HYBRID, metric::Minkowski<D>>& metadomain,
                          Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                          const prm::Parameters&  engine_params,
                          const SimulationParams& params,
                          const ndfield_t<D, 6>&  moments) -> int {
      const auto dt       = engine_params.get<real_t>("dt");
      const auto dx       = domain.mesh.metric.get_dx();
      const auto gamma_ad = params.get<real_t>("hybrid.gamma_ad");
      const auto theta    = params.get<real_t>("hybrid.theta0");
      const auto d0       = params.get<real_t>("scales.skindepth0");
      const auto rho0     = params.get<real_t>("scales.larmor0");
      const auto dens_min = params.get<real_t>("hybrid.dens_min");
      const auto hall_lim = params.get<real_t>("hybrid.hall_lim");
      const auto res_vac  = params.get<real_t>("hybrid.resist_vac");
      const auto res_hyp  = params.get<real_t>("hybrid.resist_hyper");
      const auto res_uni  = params.get<real_t>("hybrid.resist");
      const auto courant  = params.get<real_t>("hybrid.subcycle_courant");
      auto       m_max    = params.get<int>("hybrid.subcycle_max");
      raise::ErrorIf(courant <= ZERO,
                     "hybrid.subcycle_courant must be positive",
                     HERE);
      if (m_max < 1) {
        m_max = 1;
      }

      // pick m from the global max whistler speed at the advance start
      real_t vw_max = ZERO;
      Kokkos::parallel_reduce("VwMax",
                              domain.mesh.rangeActiveCells(),
                              kernel::hybrid::VwMax_kernel<D>(domain.fields.em,
                                                              moments,
                                                              3,
                                                              3,
                                                              d0,
                                                              rho0,
                                                              dens_min,
                                                              dx),
                              Kokkos::Max<real_t>(vw_max));
#if defined(MPI_ENABLED)
      MPI_Allreduce(MPI_IN_PLACE,
                    &vw_max,
                    1,
                    mpi::get_type<real_t>(),
                    MPI_MAX,
                    MPI_COMM_WORLD);
#endif
      // clamp in double before the int cast: at runaway field amplitudes
      // vw_max*dt/(courant*dx) exceeds INT_MAX, and casting that to int is UB
      // (wraps negative), which the m < 1 -> m = 1 clamp would then turn into
      // m = 1, disabling the subcycler exactly when it is needed. Comparing in
      // double and taking the else-branch also routes a NaN/Inf vw_max to m_max.
      const double m_want = std::ceil(static_cast<double>(vw_max) * dt /
                                      (courant * dx));
      int          m;
      if (m_want < static_cast<double>(m_max)) {
        m = static_cast<int>(m_want);
        if (m < 1) {
          m = 1;
        }
      } else {
        m = m_max;
      }
      const real_t dt_sub = dt / static_cast<real_t>(m);

      const auto range_act = domain.mesh.rangeActiveCells();
      const auto range_all = domain.mesh.rangeAllCells();
      const auto range_emf = ActiveWithUpperHalo(domain);

      // stage E from frozen moments and the current sub-stepped B (cur):
      // raw edge-E only -> em0::345 on the active cells and the upper halo
      // layer the Faraday stage reads (see ActiveWithUpperHalo), + wall
      // condition. The Hall limiter inside receives the SUB-step dt.
      auto emf_stage = [&]() {
        // clang-format off
        Kokkos::parallel_for(
          "EMFSub",
          range_emf,
          kernel::hybrid::EMF_kernel<D, true, true>(moments,            // P
                                                    moments,            // N
                                                    domain.fields.em0,  // Ee_in (unused)
                                                    domain.fields.cur,  // Bf (unused)
                                                    domain.fields.em0,  // Ec (unused)
                                                    domain.fields.cur,  // Bfs = B_s
                                                    domain.fields.em0,  // Ee_out
                                                    domain.fields.em0,  // Ec_out (unused)
                                                    domain.fields.bckp, // Bc_out (unused)
                                                    0, 3, 0,            // P, N, Ee_in
                                                    0, 0, 0,            // Bf, Ec, Bfs
                                                    3, 0, 0,            // Ee_out, Ec_out, Bc_out
                                                    dt_sub, gamma_ad, theta, d0, rho0, dens_min, hall_lim, res_vac, res_hyp, res_uni, dx));
        // clang-format on
        WallEPrime(domain, metadomain.mesh());
      };
      // cur += dt_sub * curl(em0::345), in place (+ halo + wall condition)
      auto faraday_stage = [&]() {
        Kokkos::parallel_for(
          "FaradaySub",
          range_act,
          kernel::hybrid::Faraday_kernel<D, 3, 3>(domain.fields.em0,
                                                  domain.fields.cur,
                                                  domain.fields.cur,
                                                  3,
                                                  0,
                                                  0,
                                                  dt_sub,
                                                  dx));
        metadomain.CommunicateFields(domain, ::Comm::CUR);
        WallScratchB(domain, metadomain.mesh());
      };
      // cur = a*cur + b*buff over the full extent (ghosts of both are valid)
      auto combine = [&](real_t a, real_t b) {
        Kokkos::parallel_for(
          "FieldCombine",
          range_all,
          kernel::hybrid::FieldCombine_kernel<D, 3, 3>(domain.fields.cur,
                                                       domain.fields.buff,
                                                       a,
                                                       b,
                                                       0,
                                                       0));
      };

      // start from Bf^(n), ghosts included (valid: comm'd + wall'd at the end
      // of the previous step)
      Kokkos::parallel_for(
        "FieldCombine",
        range_all,
        kernel::hybrid::FieldCombine_kernel<D, 3, 6>(domain.fields.cur,
                                                     domain.fields.em,
                                                     ZERO,
                                                     ONE,
                                                     0,
                                                     3));
      for (int s = 0; s < m; ++s) {
        // sub-step base: buff <- cur
        Kokkos::parallel_for(
          "FieldCombine",
          range_all,
          kernel::hybrid::FieldCombine_kernel<D, 3, 3>(domain.fields.buff,
                                                       domain.fields.cur,
                                                       ZERO,
                                                       ONE,
                                                       0,
                                                       0));
        // SSP-RK3 (Shu-Osher): u1 = u + dt L(u)
        emf_stage();
        faraday_stage();
        // u2 = 3/4 u + 1/4 (u1 + dt L(u1))
        emf_stage();
        faraday_stage();
        combine(static_cast<real_t>(0.25), static_cast<real_t>(0.75));
        // u^(s+1) = 1/3 u + 2/3 (u2 + dt L(u2))
        emf_stage();
        faraday_stage();
        combine(static_cast<real_t>(2.0 / 3.0), static_cast<real_t>(1.0 / 3.0));
      }
      return m;
    }

    /**
     * @brief Accept the sub-cycled advance as B^(n+1): em::345 <- cur (full
     *        extent -- the ghosts of cur are valid). Replaces Faraday push #3,
     *        which in sub-cycled mode would redundantly re-integrate the same
     *        interval with a stiff-unstable single Euler step.
     */
    template <Dimension D>
    void AcceptSubcycledB(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain) {
      Kokkos::parallel_for(
        "FieldCombine",
        domain.mesh.rangeAllCells(),
        kernel::hybrid::FieldCombine_kernel<D, 6, 3>(domain.fields.em,
                                                     domain.fields.cur,
                                                     ZERO,
                                                     ONE,
                                                     3,
                                                     0));
    }

  } // namespace hybrid
} // namespace ntt

#endif

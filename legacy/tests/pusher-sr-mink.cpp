#include "wrapper.h"

#include "particle_macros.h"
#include "pic.h"
#include "sim_params.h"

#include "io/input.h"
#include "meshblock/meshblock.h"
#include "utilities/qmath.h"

#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml.hpp>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    using namespace toml::literals::toml_literals;
    const auto inputdata = R"(
      [domain]
      resolution  = [256, 128, 128]
      extent      = [-16.0, 16.0, -8.0, 8.0, -8.0, 8.0]
      boundaries  = [["PERIODIC"], ["PERIODIC"], ["PERIODIC"]]

      [units]
      ppc0       = 1.0
      larmor0    = 2.0
      skindepth0 = 1.0

      [particles]
      n_species = 1

      [species_1]
      label    = "e+"
      mass     = 1.0
      charge   = 1.0
      maxnpart = 10.0

      [output]
      format = "disabled"
    )"_toml;

    auto sim = ntt::PIC<ntt::Dim3>(inputdata);

    real_t       bx1 = 0.256, bx2 = 0.953, bx3 = -0.234;
    const real_t bmag     = 2.0;
    const real_t u_part   = 2.0;
    const auto   nperiods = 5;

    const auto bb           = math::sqrt(SQR(bx1) + SQR(bx2) + SQR(bx3));
    bx1                    /= bb;
    bx2                    /= bb;
    bx3                    /= bb;
    const real_t beta_part  = u_part / math::sqrt(ONE + SQR(u_part));
    const real_t ax1 = bx2, ax2 = -bx3, ax3 = bx1;
    real_t       perp_x1 = ax2 * bx3 - ax3 * bx2;
    real_t       perp_x2 = ax3 * bx1 - ax1 * bx3;
    real_t       perp_x3 = ax1 * bx2 - ax2 * bx1;
    const real_t perp  = math::sqrt(SQR(perp_x1) + SQR(perp_x2) + SQR(perp_x3));
    perp_x1           /= perp;
    perp_x2           /= perp;
    perp_x3           /= perp;
    const auto ux1     = u_part * perp_x1;
    const auto ux2     = u_part * perp_x2;
    const auto ux3     = u_part * perp_x3;

    auto& mblock = sim.meshblock;
    {
      Kokkos::parallel_for(
        "InitFields",
        mblock.rangeActiveCells(),
        Lambda(ntt::index_t i1, ntt::index_t i2, ntt::index_t i3) {
          ntt::coord_t<ntt::Dim3> xi { ZERO };
          ntt::vec_t<ntt::Dim3>   b_cntrv { ZERO };
          mblock.metric.v3_PhysCntrv2Cntrv(xi,
                                           { bmag * bx1, bmag * bx2, bmag * bx3 },
                                           b_cntrv);
          mblock.em(i1, i2, i3, ntt::em::bx1) = b_cntrv[0];
          mblock.em(i1, i2, i3, ntt::em::bx2) = b_cntrv[1];
          mblock.em(i1, i2, i3, ntt::em::bx3) = b_cntrv[2];
        });
      sim.Communicate(ntt::Comm_B);
    }

    {
      using namespace ntt;
      mblock.particles[0].setNpart(1);
      auto positrons = mblock.particles[0];
      Kokkos::parallel_for(
        "InitParticles",
        positrons.rangeActiveParticles(),
        Lambda(ntt::index_t p) {
          init_prtl_3d(mblock, positrons, p, 0.0, 0.0, 0.0, ux1, ux2, ux3, 1.0);
        });
    }

    {
      const auto dt        = mblock.timestep();
      const auto larmor    = sim.params()->larmor0() * u_part / bmag;
      const auto period    = ntt::constant::TWO_PI * larmor / beta_part;
      auto       positrons = mblock.particles[0];
      auto       maxdist = ZERO, maxupar = ZERO;
      const auto nmax = static_cast<int>(nperiods * period / dt);
      for (auto n { 0 }; n < nmax + 1; ++n) {
        if (n < nmax) {
          sim.ParticlesPush();
        } else {
          const auto fraction = (nperiods * period / dt - nmax);
          sim.ParticlesPush(fraction);
        }
        positrons.SyncHostDevice();

        {
          ntt::coord_t<ntt::Dim3> xprtl { ZERO };
          ntt::coord_t<ntt::Dim3> xi { static_cast<real_t>(positrons.i1_h(0)) +
                                         static_cast<real_t>(positrons.dx1_h(0)),
                                       static_cast<real_t>(positrons.i2_h(0)) +
                                         static_cast<real_t>(positrons.dx2_h(0)),
                                       static_cast<real_t>(positrons.i3_h(0)) +
                                         static_cast<real_t>(positrons.dx3_h(0)) };
          mblock.metric.x_Code2Cart(xi, xprtl);
          const auto dist = math::sqrt(
            SQR(xprtl[0]) + SQR(xprtl[1]) + SQR(xprtl[2]));
          if (dist > maxdist) {
            maxdist = dist;
          }
          if (n == nmax) {
            !(ntt::AlmostZero(SQR(dist), (real_t)1e-4))
              ? throw std::logic_error(fmt::format(
                  "particle not in init position: %.6e != 0.0, L2 = %.6e",
                  dist,
                  SQR(dist)))
              : (void)0;
            !(ntt::AlmostEqual(maxdist, TWO * larmor, (real_t)1e-3))
              ? throw std::logic_error(
                  fmt::format("maxdist is incorrect: %.6f != %.6f",
                              maxdist,
                              TWO * larmor))
              : (void)0;
            !(ntt::AlmostZero(maxupar))
              ? throw std::logic_error(
                  fmt::format("maxupar is nonzero: %f", maxupar))
              : (void)0;
            const auto L2_u = SQR(positrons.ux1_h(0) - ux1) +
                              SQR(positrons.ux2_h(0) - ux2) +
                              SQR(positrons.ux3_h(0) - ux3);
            !(ntt::AlmostZero(L2_u, (real_t)1e-4))
              ? throw std::logic_error(
                  fmt::format("u_init != u_final: L2 = %.2e", L2_u))
              : (void)0;
          }
        }

        {
          const auto u_mag = math::sqrt(SQR(positrons.ux1_h(0)) +
                                        SQR(positrons.ux2_h(0)) +
                                        SQR(positrons.ux3_h(0)));
          !(ntt::AlmostEqual(u_mag, u_part))
            ? throw std::logic_error(
                fmt::format("u_mag is incorrect after %d pushes: %.6f != %.6f",
                            n,
                            u_mag,
                            u_part))
            : (void)0;
          const auto upar = (positrons.ux1_h(0) * bx1 + positrons.ux2_h(0) * bx2 +
                             positrons.ux3_h(0) * bx3) /
                            (u_part * bmag);
          if (math::abs(upar) > maxupar) {
            maxupar = math::abs(upar);
          }
          !(ntt::AlmostZero(upar))
            ? throw std::logic_error(
                fmt::format("u_|| is nonzero after %d pushes: %.2e", n, upar))
            : (void)0;
        }
      }
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
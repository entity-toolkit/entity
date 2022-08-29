#ifndef TEST_PIC_MINKOWSKI_H
#define TEST_PIC_MINKOWSKI_H

#include "global.h"
#include "qmath.h"

#include "pic.h"
#include "fields.h"
#include "particle_macros.h"

#include "output_csv.h"

#include <toml/toml.hpp>

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>

TEST_CASE("testing PIC") {
  Kokkos::initialize();
  /* -------------------------------------------------------------------------- */
  /*                            Minkowski metric test                           */
  /* -------------------------------------------------------------------------- */
  SUBCASE("Minkowski") {
    /* ------------------------- 2D particle pusher test ------------------------ */
    SUBCASE("2D pusher [E x B]") {
      std::string input_toml = R"TOML(
        [domain]
        resolution      = [64, 64]
        extent          = [-2.0, 2.0, -2.0, 2.0]
        boundaries      = ["PERIODIC", "PERIODIC"]

        [units]
        ppc0            = 1.0
        larmor0         = 1.0
        skindepth0      = 0.1

        [particles]
        n_species       = 2

        [species_1]
        mass            = 1.0
        charge          = -1.0
        maxnpart        = 10.0

        [species_2]
        mass            = 1.0
        charge          = 1.0
        maxnpart        = 10.0
      )TOML";
      try {
        std::istringstream is(input_toml, std::ios_base::binary | std::ios_base::in);
        auto               inputdata = toml::parse(is, "std::string");
        ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
        sim.initialize();

        CHECK(ntt::AlmostEqual(sim.mblock()->timestep(), 0.041984464973f));
        CHECK(ntt::AlmostEqual(sim.sim_params()->sigma0(), 0.01f));

        real_t beta_drift = 0.1f;

        // initialize Bz
        auto mblock = sim.mblock();
        Kokkos::parallel_for(
          "set fields",
          sim.rangeActiveCells(),
          Lambda(const std::size_t i, const std::size_t j) {
            real_t i_ {(real_t)(static_cast<int>(i) - ntt::N_GHOSTS)};
            real_t j_ {(real_t)(static_cast<int>(j) - ntt::N_GHOSTS)};
            ntt::vec_t<ntt::Dimension::THREE_D> e_cntrv, b_cntrv;
            mblock->metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, ONE}, b_cntrv);
            mblock->em(i, j, ntt::em::bx3) = b_cntrv[2];

            mblock->metric.v_Hat2Cntrv({i_, j_ + HALF}, {ZERO, beta_drift, ZERO}, e_cntrv);
            mblock->em(i, j, ntt::em::ex2) = e_cntrv[1];
          });
        sim.fieldBoundaryConditions(0.0f);

        // initialize particles
        real_t                              gammabeta = 1.0f;
        ntt::coord_t<ntt::Dimension::TWO_D> x_init {0.1, 0.12};

        Kokkos::parallel_for(
          "set particles",
          ntt::CreateRangePolicy<ntt::Dimension::ONE_D>({0}, {1}),
          Lambda(const std::size_t p) {
            PICPRTL_XYZ_2D(mblock, 0, p, x_init[0], x_init[1], gammabeta, ZERO, ZERO);
            PICPRTL_XYZ_2D(mblock, 1, p, x_init[0], x_init[1], gammabeta, ZERO, ZERO);
          });
        (mblock->particles[0]).set_npart(1);
        (mblock->particles[1]).set_npart(1);
        for (int i {0}; i < 2; ++i) {
          CHECK(mblock->particles[i].i1(0) == 33);
          CHECK(mblock->particles[i].i2(0) == 33);
          CHECK(ntt::AlmostEqual(mblock->particles[i].dx1(0), 0.6f));
          CHECK(ntt::AlmostEqual(mblock->particles[i].dx2(0), 0.92f));
        }

        // run for some # of timesteps
        real_t runtime = 500.0f;
        for (int i {0}; i < (int)runtime; ++i) {
          sim.pushParticlesSubstep(0.0f, ONE);
          sim.particleBoundaryConditions(0.0f);
        }

        // compare with analytics
        real_t larmor0 = sim.sim_params()->larmor0();
        real_t dt      = sim.mblock()->timestep();

        real_t beta        = gammabeta / math::sqrt(1.0f + SQR(gammabeta));
        real_t gamma_drift = 1.0f / math::sqrt(1.0f - SQR(beta_drift));
        real_t time        = (runtime - 1.0f) * dt / gamma_drift;

        real_t gammabeta1 = gammabeta * (gamma_drift - gamma_drift * beta_drift / beta);
        real_t beta1      = gammabeta1 / math::sqrt(1.0f + SQR(gammabeta1));
        real_t larmor1    = larmor0 * gammabeta1;
        real_t omega1     = beta1 / larmor1;

        real_t phase    = (time * omega1);
        real_t x_p1_fin = (x_init[0] + beta_drift * time) + (larmor1 * std::sin(phase));
        real_t y_p1_fin = (x_init[1] + larmor1) - (larmor1 * std::cos(phase));
        real_t x_p2_fin = (x_init[0] + beta_drift * time) + (larmor1 * std::sin(phase));
        real_t y_p2_fin = (x_init[1] - larmor1) + (larmor1 * std::cos(phase));
        while (x_p1_fin >= 2.0f) {
          x_p1_fin -= 4.0f;
        }
        while (x_p1_fin < -2.0f) {
          x_p1_fin += 4.0f;
        }
        while (x_p2_fin >= 2.0f) {
          x_p2_fin -= 4.0f;
        }
        while (x_p2_fin < -2.0f) {
          x_p2_fin += 4.0f;
        }
        while (y_p1_fin >= 2.0f) {
          y_p1_fin -= 4.0f;
        }
        while (y_p1_fin < -2.0f) {
          y_p1_fin += 4.0f;
        }
        while (y_p2_fin >= 2.0f) {
          y_p2_fin -= 4.0f;
        }
        while (y_p2_fin < -2.0f) {
          y_p2_fin += 4.0f;
        }

        // check: 1 % accurate after ~ 5 rotations
        ntt::coord_t<ntt::Dimension::TWO_D> x_p {ZERO, ZERO};
        mblock->metric.x_Code2Cart(
          {(real_t)(mblock->particles[0].i1(0)) + mblock->particles[0].dx1(0),
           (real_t)(mblock->particles[0].i2(0)) + mblock->particles[0].dx2(0)},
          x_p);
        CHECK(ntt::AlmostEqual(x_p[0], x_p1_fin, 0.01f));
        CHECK(ntt::AlmostEqual(x_p[1], y_p1_fin, 0.01f));

        mblock->metric.x_Code2Cart(
          {(real_t)(mblock->particles[1].i1(0)) + mblock->particles[1].dx1(0),
           (real_t)(mblock->particles[1].i2(0)) + mblock->particles[1].dx2(0)},
          x_p);
        CHECK(ntt::AlmostEqual(x_p[0], x_p2_fin, 0.01f));
        CHECK(ntt::AlmostEqual(x_p[1], y_p2_fin, 0.01f));

        sim.finalize();
      }
      catch (std::exception& err) {
        std::cerr << err.what() << std::endl;
      }
    }
    /* -------------------------- 2D field solver test -------------------------- */
    SUBCASE("2D fieldsolver") {
      // test field solver
      std::string input_toml = R"TOML(
        [domain]
        resolution      = [256, 256]
        extent          = [0.0, 1.0, 0.0, 1.0]
        boundaries      = ["PERIODIC", "PERIODIC"]

        [units]
        ppc0            = 1.0
        larmor0         = 1.0
        skindepth0      = 0.1
      )TOML";
      try {

        std::istringstream is(input_toml, std::ios_base::binary | std::ios_base::in);
        auto               inputdata = toml::parse(is, "std::string");
        ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
        sim.initialize();

        CHECK(ntt::AlmostEqual(sim.mblock()->timestep(), 0.0026240291f));

        real_t sx = sim.mblock()->metric.x1_max - sim.mblock()->metric.x1_min;
        real_t sy = sim.mblock()->metric.x2_max - sim.mblock()->metric.x2_min;

        real_t nx1       = 2.0f;
        real_t nx2       = 3.0f;
        real_t amplitude = 1.0f;
        auto   kx        = (real_t)(ntt::constant::TWO_PI)*nx1 / sx;
        auto   ky        = (real_t)(ntt::constant::TWO_PI)*nx2 / sy;
        real_t ex_ampl, ey_ampl, bz_ampl = amplitude;
        ex_ampl = amplitude * (-ky) / math::sqrt(SQR(kx) + SQR(ky));
        ey_ampl = amplitude * (kx) / math::sqrt(SQR(kx) + SQR(ky));

        auto mblock = sim.mblock();
        Kokkos::parallel_for(
          "userInitFlds",
          sim.rangeActiveCells(),
          Lambda(const std::size_t i, const std::size_t j) {
            // index to code units
            real_t i_ {(real_t)(static_cast<int>(i) - ntt::N_GHOSTS)},
              j_ {(real_t)(static_cast<int>(j) - ntt::N_GHOSTS)};

            // code units to cartesian (physical units)
            ntt::coord_t<ntt::Dimension::TWO_D> x_y, xp_y, x_yp, xp_yp;
            mblock->metric.x_Code2Cart({i_, j_}, x_y);
            mblock->metric.x_Code2Cart({i_ + HALF, j_}, xp_y);
            mblock->metric.x_Code2Cart({i_, j_ + HALF}, x_yp);
            mblock->metric.x_Code2Cart({i_ + HALF, j_ + HALF}, xp_yp);

            ntt::vec_t<ntt::Dimension::THREE_D> ex_cntr, ey_cntr, bz_cntr;

            // hatted fields
            ntt::vec_t<ntt::Dimension::THREE_D> e_hat {ZERO, ZERO, ZERO};
            // i + 1/2, j
            e_hat[0] = ex_ampl * math::sin(kx * xp_y[0] + ky * xp_y[1]);
            e_hat[1] = ey_ampl * math::sin(kx * xp_y[0] + ky * xp_y[1]);
            e_hat[2] = ZERO;
            mblock->metric.v_Hat2Cntrv({i_ + HALF, j_}, e_hat, ex_cntr);
            // i, j + 1/2
            e_hat[0] = ex_ampl * math::sin(kx * x_yp[0] + ky * x_yp[1]);
            e_hat[1] = ey_ampl * math::sin(kx * x_yp[0] + ky * x_yp[1]);
            e_hat[2] = ZERO;
            mblock->metric.v_Hat2Cntrv({i_, j_ + HALF}, e_hat, ey_cntr);

            real_t bz_hat = bz_ampl * math::sin(kx * xp_yp[0] + ky * xp_yp[1]);
            mblock->metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bz_hat}, bz_cntr);

            mblock->em(i, j, ntt::em::ex1) = ex_cntr[0];
            mblock->em(i, j, ntt::em::ex2) = ey_cntr[1];
            mblock->em(i, j, ntt::em::bx3) = bz_cntr[2];
          });
        sim.fieldBoundaryConditions(0.0f);

        // run for some # of timesteps
        real_t runtime = 250.0f;
        for (int i {0}; i < (int)runtime; ++i) {
          sim.faradaySubstep(0.0f, HALF);
          sim.fieldBoundaryConditions(0.0f);

          sim.ampereSubstep(0.0f, ONE);
          sim.fieldBoundaryConditions(0.0f);

          sim.faradaySubstep(0.0f, HALF);
          sim.fieldBoundaryConditions(0.0f);
        }

        // check that the fields are correct
        CHECK(ntt::AlmostEqual(
          sim.mblock()->em(100 + ntt::N_GHOSTS, 150 + ntt::N_GHOSTS, ntt::em::ex1),
          -191.516f));
        CHECK(ntt::AlmostEqual(
          sim.mblock()->em(10 + ntt::N_GHOSTS, 54 + ntt::N_GHOSTS, ntt::em::ex2), 114.057f));
        CHECK(ntt::AlmostEqual(
          sim.mblock()->em(60 + ntt::N_GHOSTS, 186 + ntt::N_GHOSTS, ntt::em::bx3), 246.654f));

        real_t all_zeros = 0.0f;
        for (std::size_t i {0}; i < 256; ++i) {
          for (std::size_t j {0}; j < 256; ++j) {
            all_zeros += math::abs(
              sim.mblock()->em(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS, ntt::em::ex3));
            all_zeros += math::abs(
              sim.mblock()->em(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS, ntt::em::bx1));
            all_zeros += math::abs(
              sim.mblock()->em(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS, ntt::em::bx2));
          }
        }
        CHECK(ntt::AlmostEqual(all_zeros, 0.0f));
        sim.finalize();
      }
      catch (std::exception& err) {
        std::cerr << err.what() << std::endl;
      }
    }
    /* --------------------------- 2D current deposit --------------------------- */
    SUBCASE("2D deposit") {
      std::string input_toml = R"TOML(
        [domain]
        resolution      = [256, 64]
        extent          = [0.0, 4.0, 0.0, 1.0]
        boundaries      = ["PERIODIC", "PERIODIC"]

        [units]
        ppc0            = 1.0
        larmor0         = 1.0
        skindepth0      = 0.1

        [particles]
        n_species       = 2

        [species_1]
        mass            = 1.0
        charge          = -1.0
        maxnpart        = 10.0

        [species_2]
        mass            = 1.0
        charge          = 1.6
        maxnpart        = 10.0
      )TOML";
      try {
        std::istringstream is(input_toml, std::ios_base::binary | std::ios_base::in);
        auto               inputdata = toml::parse(is, "std::string");
        ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
        sim.initialize();

        CHECK(ntt::AlmostEqual(sim.mblock()->timestep(), 0.010496116243f));
        CHECK(ntt::AlmostEqual(sim.sim_params()->sigma0(), 0.01f));

        {
          auto                                mblock       = sim.mblock();
          real_t                              gammabeta    = 12.0f;
          real_t                              pitch_angle1 = ntt::constant::PI / 6.0f;
          real_t                              pitch_angle2 = ntt::constant::PI / 4.0f;
          ntt::coord_t<ntt::Dimension::TWO_D> x_init1 {2.5233f, 0.1265f};
          ntt::coord_t<ntt::Dimension::TWO_D> x_init2 {0.254f, 0.834f};

          Kokkos::parallel_for(
            "set particles",
            ntt::CreateRangePolicy<ntt::Dimension::ONE_D>({0}, {1}),
            Lambda(const std::size_t p) {
              PICPRTL_XYZ_2D(mblock,
                             0,
                             p,
                             x_init1[0],
                             x_init1[1],
                             ZERO,
                             gammabeta * math::cos(pitch_angle1),
                             gammabeta * math::sin(pitch_angle1));
              PICPRTL_XYZ_2D(mblock,
                             1,
                             p,
                             x_init2[0],
                             x_init2[1],
                             -gammabeta * math::cos(pitch_angle2),
                             -gammabeta * math::sin(pitch_angle2),
                             ZERO);
            });
          (mblock->particles[0]).set_npart(1);
          (mblock->particles[1]).set_npart(1);

          sim.pushParticlesSubstep(ZERO, ONE);

          sim.depositCurrentsSubstep(ZERO);

          CHECK(ntt::AlmostEqual(
            sim.mblock()->cur(161 + ntt::N_GHOSTS, 8 + ntt::N_GHOSTS, ntt::cur::jx2)
              + sim.mblock()->cur(162 + ntt::N_GHOSTS, 8 + ntt::N_GHOSTS, ntt::cur::jx2),
            -55.23417f));
          CHECK(ntt::AlmostEqual(
            sim.mblock()->cur(161 + ntt::N_GHOSTS, 8 + ntt::N_GHOSTS, ntt::cur::jx3)
              + sim.mblock()->cur(162 + ntt::N_GHOSTS, 8 + ntt::N_GHOSTS, ntt::cur::jx3)
              + sim.mblock()->cur(161 + ntt::N_GHOSTS, 9 + ntt::N_GHOSTS, ntt::cur::jx3)
              + sim.mblock()->cur(162 + ntt::N_GHOSTS, 9 + ntt::N_GHOSTS, ntt::cur::jx3),
            -31.889464f));

          CHECK(ntt::AlmostEqual(
            sim.mblock()->cur(15 + ntt::N_GHOSTS, 52 + ntt::N_GHOSTS, ntt::cur::jx1)
              + sim.mblock()->cur(15 + ntt::N_GHOSTS, 53 + ntt::N_GHOSTS, ntt::cur::jx1)
              + sim.mblock()->cur(16 + ntt::N_GHOSTS, 53 + ntt::N_GHOSTS, ntt::cur::jx1)
              + sim.mblock()->cur(16 + ntt::N_GHOSTS, 54 + ntt::N_GHOSTS, ntt::cur::jx1),
            -1.6f * 45.098512f));

          CHECK(ntt::AlmostEqual(
            sim.mblock()->cur(15 + ntt::N_GHOSTS, 52 + ntt::N_GHOSTS, ntt::cur::jx2)
              + sim.mblock()->cur(16 + ntt::N_GHOSTS, 52 + ntt::N_GHOSTS, ntt::cur::jx2)
              + sim.mblock()->cur(16 + ntt::N_GHOSTS, 53 + ntt::N_GHOSTS, ntt::cur::jx2)
              + sim.mblock()->cur(17 + ntt::N_GHOSTS, 53 + ntt::N_GHOSTS, ntt::cur::jx2),
            -1.6f * 45.098512f));

          sim.addCurrentsSubstep(ZERO);

          real_t ex1 {0.0f};
          Kokkos::parallel_reduce(
            "post-deposit",
            sim.rangeActiveCells(),
            Lambda(ntt::index_t i, ntt::index_t j, real_t & sum) {
              sum += mblock->em(i, j, ntt::em::ex1);
            },
            ex1);

          real_t ex2 {0.0f};
          Kokkos::parallel_reduce(
            "post-deposit",
            sim.rangeActiveCells(),
            Lambda(ntt::index_t i, ntt::index_t j, real_t & sum) {
              sum += mblock->em(i, j, ntt::em::ex2);
            },
            ex2);

          real_t ex3 {0.0f};
          Kokkos::parallel_reduce(
            "post-deposit",
            sim.rangeActiveCells(),
            Lambda(ntt::index_t i, ntt::index_t j, real_t & sum) {
              sum += mblock->em(i, j, ntt::em::ex3);
            },
            ex3);

          real_t c0 {275148.964f};
          CHECK(ntt::AlmostEqual(ex1, c0 * 72.1576208f));
          CHECK(ntt::AlmostEqual(ex2, c0 * 127.3917908f));
          CHECK(ntt::AlmostEqual(ex3, c0 * 31.889464f));
        }
      }
      catch (std::exception& err) {
        std::cerr << err.what() << std::endl;
      }
    }
  }
  Kokkos::finalize();
}

#endif
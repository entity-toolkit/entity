#ifndef TEST_PIC_MINKOWSKI_H
#define TEST_PIC_MINKOWSKI_H

#include "wrapper.h"
#include "qmath.h"

#include "pic.h"
#include "fields.h"
#include "particle_macros.h"

#include "output_csv.h"

#include <doctest.h>
#include <toml/toml.hpp>

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>

TEST_CASE("testing PIC") {
  using namespace ntt;
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
        PIC<Dim2>          sim(inputdata);
        sim.Initialize();

        auto& mblock = sim.meshblock;
        auto  params = *(sim.params());

        CHECK(AlmostEqual(mblock.timestep(), 0.041984464973f));
        CHECK(AlmostEqual(params.sigma0(), 0.01f));

        real_t beta_drift = 0.1f;

        // initialize Bz
        {
          Kokkos::parallel_for(
            "set fields",
            sim.rangeActiveCells(),
            Lambda(const std::size_t i, const std::size_t j) {
              real_t      i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)};
              real_t      j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};
              vec_t<Dim3> e_cntrv, b_cntrv;
              mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, ONE}, b_cntrv);
              mblock.em(i, j, em::bx3) = b_cntrv[2];

              mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, {ZERO, beta_drift, ZERO}, e_cntrv);
              mblock.em(i, j, em::ex2) = e_cntrv[1];
            });
          sim.FieldsExchange();
        }

        // initialize particles
        real_t        gammabeta = 1.0f;
        coord_t<Dim2> x_init {0.1, 0.12};

        {
          auto& electrons = mblock.particles[0];
          auto& positrons = mblock.particles[1];
          Kokkos::parallel_for(
            "set particles", CreateRangePolicy<Dim1>({0}, {1}), Lambda(const std::size_t p) {
              init_prtl_2d_XYZ(
                mblock, electrons, p, x_init[0], x_init[1], gammabeta, ZERO, ZERO);
              init_prtl_2d_XYZ(
                mblock, positrons, p, x_init[0], x_init[1], gammabeta, ZERO, ZERO);
            });
          electrons.setNpart(1);
          positrons.setNpart(1);
        }

        sim.SynchronizeHostDevice();

        for (int i {0}; i < 2; ++i) {
          CHECK(mblock.particles[i].i1_h(0) == 33);
          CHECK(mblock.particles[i].i2_h(0) == 33);
          CHECK(AlmostEqual(mblock.particles[i].dx1_h(0), 0.6f));
          CHECK(AlmostEqual(mblock.particles[i].dx2_h(0), 0.92f));
        }

        // run for some # of timesteps
        real_t runtime = 500.0f;
        for (int i {0}; i < (int)runtime; ++i) {
          sim.ParticlesPush();
          sim.ParticlesExchange();
        }

        // compare with analytics
        real_t larmor0 = params.larmor0();
        real_t dt      = mblock.timestep();

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

        sim.SynchronizeHostDevice();

        // check: 1 % accurate after ~ 5 rotations
        coord_t<Dim2> x_p {ZERO, ZERO};
        mblock.metric.x_Code2Cart(
          {(real_t)(mblock.particles[0].i1_h(0)) + mblock.particles[0].dx1_h(0),
           (real_t)(mblock.particles[0].i2_h(0)) + mblock.particles[0].dx2_h(0)},
          x_p);
        CHECK(AlmostEqual(x_p[0], x_p1_fin, 0.01f));
        CHECK(AlmostEqual(x_p[1], y_p1_fin, 0.01f));

        mblock.metric.x_Code2Cart(
          {(real_t)(mblock.particles[1].i1_h(0)) + mblock.particles[1].dx1_h(0),
           (real_t)(mblock.particles[1].i2_h(0)) + mblock.particles[1].dx2_h(0)},
          x_p);
        CHECK(AlmostEqual(x_p[0], x_p2_fin, 0.01f));
        CHECK(AlmostEqual(x_p[1], y_p2_fin, 0.01f));

        sim.Finalize();
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
        PIC<Dim2>          sim(inputdata);
        sim.Initialize();

        auto& mblock = sim.meshblock;

        CHECK(AlmostEqual(mblock.timestep(), 0.0026240291f));

        real_t sx = mblock.metric.x1_max - mblock.metric.x1_min;
        real_t sy = mblock.metric.x2_max - mblock.metric.x2_min;

        real_t nx1       = 2.0f;
        real_t nx2       = 3.0f;
        real_t amplitude = 1.0f;
        auto   kx        = (real_t)(constant::TWO_PI)*nx1 / sx;
        auto   ky        = (real_t)(constant::TWO_PI)*nx2 / sy;
        real_t ex_ampl, ey_ampl, bz_ampl = amplitude;
        ex_ampl = amplitude * (-ky) / math::sqrt(SQR(kx) + SQR(ky));
        ey_ampl = amplitude * (kx) / math::sqrt(SQR(kx) + SQR(ky));

        Kokkos::parallel_for(
          "userInitFlds",
          sim.rangeActiveCells(),
          Lambda(const std::size_t i, const std::size_t j) {
            // index to code units
            real_t i_ {(real_t)(static_cast<int>(i) - N_GHOSTS)},
              j_ {(real_t)(static_cast<int>(j) - N_GHOSTS)};

            // code units to cartesian (physical units)
            coord_t<Dim2> x_y, xp_y, x_yp, xp_yp;
            mblock.metric.x_Code2Cart({i_, j_}, x_y);
            mblock.metric.x_Code2Cart({i_ + HALF, j_}, xp_y);
            mblock.metric.x_Code2Cart({i_, j_ + HALF}, x_yp);
            mblock.metric.x_Code2Cart({i_ + HALF, j_ + HALF}, xp_yp);

            vec_t<Dim3> ex_cntr, ey_cntr, bz_cntr;

            // hatted fields
            vec_t<Dim3> e_hat {ZERO, ZERO, ZERO};
            // i + 1/2, j
            e_hat[0] = ex_ampl * math::sin(kx * xp_y[0] + ky * xp_y[1]);
            e_hat[1] = ey_ampl * math::sin(kx * xp_y[0] + ky * xp_y[1]);
            e_hat[2] = ZERO;
            mblock.metric.v_Hat2Cntrv({i_ + HALF, j_}, e_hat, ex_cntr);
            // i, j + 1/2
            e_hat[0] = ex_ampl * math::sin(kx * x_yp[0] + ky * x_yp[1]);
            e_hat[1] = ey_ampl * math::sin(kx * x_yp[0] + ky * x_yp[1]);
            e_hat[2] = ZERO;
            mblock.metric.v_Hat2Cntrv({i_, j_ + HALF}, e_hat, ey_cntr);

            real_t bz_hat = bz_ampl * math::sin(kx * xp_yp[0] + ky * xp_yp[1]);
            mblock.metric.v_Hat2Cntrv({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bz_hat}, bz_cntr);

            mblock.em(i, j, em::ex1) = ex_cntr[0];
            mblock.em(i, j, em::ex2) = ey_cntr[1];
            mblock.em(i, j, em::bx3) = bz_cntr[2];
          });
        sim.FieldsExchange();

        // run for some # of timesteps
        real_t runtime = 250.0f;
        for (int i {0}; i < (int)runtime; ++i) {
          sim.Faraday();
          sim.FieldsExchange();

          sim.Ampere();
          sim.FieldsExchange();

          sim.Faraday();
          sim.FieldsExchange();
        }

        sim.SynchronizeHostDevice();

        // check that the fields are correct
        CHECK(AlmostEqual(mblock.em_h(100 + N_GHOSTS, 150 + N_GHOSTS, em::ex1), -191.516f));
        CHECK(AlmostEqual(mblock.em_h(10 + N_GHOSTS, 54 + N_GHOSTS, em::ex2), 114.057f));
        CHECK(AlmostEqual(mblock.em_h(60 + N_GHOSTS, 186 + N_GHOSTS, em::bx3), 246.654f));

        real_t all_zeros = 0.0f;
        for (std::size_t i {0}; i < 256; ++i) {
          for (std::size_t j {0}; j < 256; ++j) {
            all_zeros += math::abs(mblock.em_h(i + N_GHOSTS, j + N_GHOSTS, em::ex3));
            all_zeros += math::abs(mblock.em_h(i + N_GHOSTS, j + N_GHOSTS, em::bx1));
            all_zeros += math::abs(mblock.em_h(i + N_GHOSTS, j + N_GHOSTS, em::bx2));
          }
        }
        CHECK(AlmostEqual(all_zeros, 0.0f));
        sim.Finalize();
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
        PIC<Dim2>          sim(inputdata);
        sim.Initialize();

        auto& mblock = sim.meshblock;
        auto  params = *(sim.params());

        CHECK(AlmostEqual(mblock.timestep(), 0.010496116243f));
        CHECK(AlmostEqual(params.sigma0(), 0.01f));

        real_t        gammabeta    = 12.0f;
        real_t        pitch_angle1 = constant::PI / 6.0f;
        real_t        pitch_angle2 = constant::PI / 4.0f;
        coord_t<Dim2> x_init1 {2.5233f, 0.1265f};
        coord_t<Dim2> x_init2 {0.254f, 0.834f};

        {
          auto electrons = mblock.particles[0];
          auto positrons = mblock.particles[1];
          Kokkos::parallel_for(
            "set particles", CreateRangePolicy<Dim1>({0}, {1}), Lambda(const std::size_t p) {
              init_prtl_2d_XYZ(mblock,
                               electrons,
                               p,
                               x_init1[0],
                               x_init1[1],
                               ZERO,
                               gammabeta * math::cos(pitch_angle1),
                               gammabeta * math::sin(pitch_angle1));
              init_prtl_2d_XYZ(mblock,
                               positrons,
                               p,
                               x_init2[0],
                               x_init2[1],
                               -gammabeta * math::cos(pitch_angle2),
                               -gammabeta * math::sin(pitch_angle2),
                               ZERO);
            });
          (mblock.particles[0]).setNpart(1);
          (mblock.particles[1]).setNpart(1);
        }

        sim.ParticlesPush();

        sim.CurrentsDeposit();

        sim.SynchronizeHostDevice();

        CHECK(AlmostEqual(mblock.cur_h(161 + N_GHOSTS, 8 + N_GHOSTS, cur::jx2)
                            + mblock.cur_h(162 + N_GHOSTS, 8 + N_GHOSTS, cur::jx2),
                          -55.23417f));
        CHECK(AlmostEqual(mblock.cur_h(161 + N_GHOSTS, 8 + N_GHOSTS, cur::jx3)
                            + mblock.cur_h(162 + N_GHOSTS, 8 + N_GHOSTS, cur::jx3)
                            + mblock.cur_h(161 + N_GHOSTS, 9 + N_GHOSTS, cur::jx3)
                            + mblock.cur_h(162 + N_GHOSTS, 9 + N_GHOSTS, cur::jx3),
                          -31.889464f));

        CHECK(AlmostEqual(mblock.cur_h(15 + N_GHOSTS, 52 + N_GHOSTS, cur::jx1)
                            + mblock.cur_h(15 + N_GHOSTS, 53 + N_GHOSTS, cur::jx1)
                            + mblock.cur_h(16 + N_GHOSTS, 53 + N_GHOSTS, cur::jx1)
                            + mblock.cur_h(16 + N_GHOSTS, 54 + N_GHOSTS, cur::jx1),
                          -1.6f * 45.098512f));

        CHECK(AlmostEqual(mblock.cur_h(15 + N_GHOSTS, 52 + N_GHOSTS, cur::jx2)
                            + mblock.cur_h(16 + N_GHOSTS, 52 + N_GHOSTS, cur::jx2)
                            + mblock.cur_h(16 + N_GHOSTS, 53 + N_GHOSTS, cur::jx2)
                            + mblock.cur_h(17 + N_GHOSTS, 53 + N_GHOSTS, cur::jx2),
                          -1.6f * 45.098512f));

        sim.AmpereCurrents();

        real_t ex1 {0.0f};
        Kokkos::parallel_reduce(
          "post-deposit",
          sim.rangeActiveCells(),
          Lambda(index_t i, index_t j, real_t & sum) { sum += mblock.em(i, j, em::ex1); },
          ex1);

        real_t ex2 {0.0f};
        Kokkos::parallel_reduce(
          "post-deposit",
          sim.rangeActiveCells(),
          Lambda(index_t i, index_t j, real_t & sum) { sum += mblock.em(i, j, em::ex2); },
          ex2);

        real_t ex3 {0.0f};
        Kokkos::parallel_reduce(
          "post-deposit",
          sim.rangeActiveCells(),
          Lambda(index_t i, index_t j, real_t & sum) { sum += mblock.em(i, j, em::ex3); },
          ex3);

        real_t c0 {275148.964f};
        CHECK(AlmostEqual(ex1, c0 * 72.1576208f));
        CHECK(AlmostEqual(ex2, c0 * 127.3917908f));
        CHECK(AlmostEqual(ex3, c0 * 31.889464f));

        sim.Finalize();
      }
      catch (std::exception& err) {
        std::cerr << err.what() << std::endl;
      }
    }
  }
}

#endif
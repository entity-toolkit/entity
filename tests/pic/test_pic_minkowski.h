#ifndef TEST_PIC_MINKOWSKI_H
#define TEST_PIC_MINKOWSKI_H

#include "global.h"
#include "qmath.h"

#include "pic.h"
#include "fields.h"

#include <rapidcsv.h>
#include <toml/toml.hpp>

#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>

TEST_CASE("testing PIC") {
  Kokkos::initialize();
  SUBCASE("Minkowski") {
    SUBCASE("2D pusher [E x B]") {
      std::string        input_toml = R"TOML(
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
      std::istringstream is(input_toml, std::ios_base::binary | std::ios_base::in);
      auto               inputdata = toml::parse(is, "std::string");
      ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
      sim.initialize();

      CHECK(ntt::AlmostEqual(sim.mblock()->timestep(), 0.041984464973f));
      CHECK(ntt::AlmostEqual(sim.sim_params()->B0(), 1.0f));
      CHECK(ntt::AlmostEqual(sim.sim_params()->charge0(), 100.0f));
      CHECK(ntt::AlmostEqual(sim.sim_params()->sigma0(), 0.01f));

      real_t beta_drift = 0.1f;

      // initialize Bz
      auto mblock = sim.mblock();
      Kokkos::parallel_for(
        "set fields", sim.loopActiveCells(), Lambda(const std::size_t i, const std::size_t j) {
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
        ntt::NTTRange<ntt::Dimension::ONE_D>({0}, {1}),
        Lambda(const std::size_t p) {
          ntt::coord_t<ntt::Dimension::TWO_D> x_CU;
          mblock->metric.x_Cart2Code(x_init, x_CU);
          auto [i1, dx1] = mblock->metric.CU_to_Idi(x_CU[0]);
          auto [i2, dx2] = mblock->metric.CU_to_Idi(x_CU[1]);
          // electron
          mblock->particles[0].i1(p)  = i1;
          mblock->particles[0].i2(p)  = i2;
          mblock->particles[0].dx1(p) = dx1;
          mblock->particles[0].dx2(p) = dx2;
          mblock->particles[0].ux1(p) = gammabeta;
          // positron
          mblock->particles[1].i1(p)  = i1;
          mblock->particles[1].i2(p)  = i2;
          mblock->particles[1].dx1(p) = dx1;
          mblock->particles[1].dx2(p) = dx2;
          mblock->particles[1].ux1(p) = gammabeta;
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
    SUBCASE("2D fieldsolver") {
      // test field solver
    }
    // test deposition
  }
  Kokkos::finalize();
}

#endif
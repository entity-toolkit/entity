#include "wrapper.h"

#include "pic.h"
template <ntt::Dimension D>
using SimEngine = ntt::PIC<D>;

#include <Kokkos_Core.hpp>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    const auto simname = "Deposit-" + std::string(SIMULATION_METRIC);

    const real_t x1_c  = 128.5;
    const real_t x2_c  = 128.5;
    const real_t r     = 0.4;
    real_t       omega = 10.0;

    const auto enable_output = false;

    auto inputdata = toml::table {
      {"simulation",
       {
       { "title", simname },
       { "runtime", 2.0 * ntt::constant::TWO_PI / omega },
       }},
      {    "domain",
       {
       { "resolution", { 256, 256 } },
       #ifdef MINKOWSKI_METRIC
       { "extent", { 1.0, 10.0, 1.0, 10.0 } },
       #else
          { "extent", { 1.0, 10.0 } },
          { "qsph_r0", 0.0 },
          { "qsph_h", 0.0 },
       #endif
       {
       "boundaries",
       {
       #ifdef MINKOWSKI_METRIC
       toml::array { "PERIODIC" },
       toml::array { "PERIODIC" },
       #else
              toml::array { "CUSTOM", "ABSORB" },
              toml::array { "AXIS" },
       #endif
       },
       },
       }},
      {     "units",
       {
       { "ppc0", 1.0 },
       { "larmor0", 1.0 },
       { "skindepth0", 1.0 },
       }},
      { "particles",
       {
       { "n_species", 1 },
       }},
      { "algorithm",
       {
       { "CFL", 0.9 },
       { "current_filters", 0 },
       }},
      { "species_1",
       {
       { "mass", 1.0 },
       { "charge", -1.0 },
       { "maxnpart", 1e2 },
       }},
    };

    if (enable_output) {
      plog::ColorConsoleAppender<plog::TxtFormatter> logapp;
      plog::init<ntt::LogFile>(plog::verbose, &logapp);

      auto infofile_name = simname + ".info";
      std::remove(infofile_name.c_str());
      plog::RollingFileAppender<plog::Nt2InfoFormatter> infofileAppender(
        infofile_name.c_str());
      plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);
    }

    SimEngine<ntt::Dim2> sim(inputdata);
    sim.ResetSimulation();

    {
      auto& mblock    = sim.meshblock;
      auto& electrons = mblock.particles[0];

      const auto x1  = x1_c + r;
      const auto x2  = x2_c;
      const auto ux1 = ZERO;
      const auto ux2 = r * omega;

      Kokkos::parallel_for(
        "InitParticle",
        1,
        Lambda(ntt::index_t p) {
          electrons.i1(p)  = (int)x1;
          electrons.i2(p)  = (int)x2;
          electrons.dx1(p) = x1 - math::floor(x1);
          electrons.dx2(p) = x2 - math::floor(x2);

          ntt::vec_t<ntt::Dim3> u_Cart { ZERO };
#ifdef MINKOWSKI_METRIC
          mblock.metric.v3_Cntrv2Cart({ x1, x2 }, { ux1, ux2, ZERO }, u_Cart);
#else
          mblock.metric.v3_Cntrv2Cart({ x1, x2, ZERO }, { ux1, ux2, ZERO }, u_Cart);
#endif
          electrons.ux1(p) = u_Cart[0];
          electrons.ux2(p) = u_Cart[1];
          electrons.ux3(p) = u_Cart[2];
          electrons.tag(p) = ntt::ParticleTag::alive;
        });
      electrons.setNpart(1);
    }

    sim.PrintDetails();
    sim.Verify();
    sim.InitialStep();

    auto prtl_file   = std::ofstream("prtl.csv");
    auto q_file      = std::ofstream("q.csv");
    auto ex1_file    = std::ofstream("ex1.csv");
    auto ex2_file    = std::ofstream("ex2.csv");
    auto metric_file = std::ofstream("metric.csv");
    if (enable_output) {
      prtl_file << "i1,i2,dx1,dx2,ux1,ux2,ux3" << std::endl;
      q_file << "q_A,q_B,q_C,q_D,q_E,q_F" << std::endl;
      ex1_file << std::setprecision(std::numeric_limits<real_t>::digits10);
      ex2_file << std::setprecision(std::numeric_limits<real_t>::digits10);
      metric_file << std::setprecision(std::numeric_limits<real_t>::digits10);
      ex1_file
        << "ex1_{i-1/2;j},ex1_{i+1/2;j},ex1_{i+3/2;j},ex1_{i+5/2;j},"
           "ex1_{i-1/2;j+1},ex1_{i+1/2;j+1},ex1_{i+3/2;j+1},ex1_{i+5/2;j+1}"
        << std::endl;
      ex2_file << "ex2_{i;j-1/2},ex2_{i;j+1/2},ex2_{i;j+3/2},"
                  "ex2_{i+1;j-1/2},ex2_{i+1;j+1/2},"
                  "ex2_{i+1;j+3/2},ex2_{i+2;j-1/2},"
                  "ex2_{i+2;j+1/2},ex2_{i+2;j+3/2}"
               << std::endl;
      metric_file << "h_{i-1/2;j},h_{i+1/2;j},h_{i+3/2;j},"
                     "h_{i+5/2;j},h_{i-1/2;j+1},h_{i+1/2;j+1},"
                     "h_{i+3/2;j+1},h_{i+5/2;j+1},h_{i;j-1/2},"
                     "h_{i;j+1/2},h_{i;j+3/2},h_{i+1;j-1/2},"
                     "h_{i+1;j+1/2},h_{i+1;j+3/2},h_{i+2;j-1/2},"
                     "h_{i+2;j+1/2},h_{i+2;j+3/2}"
                  << std::endl;
    }

    auto& mblock    = sim.meshblock;
    auto& electrons = mblock.particles[0];

    // charges in 6 nodes of two neighboring cells
    std::vector<real_t> q_A, q_B, q_C, q_D, q_E, q_F;

    while (sim.time() < sim.params()->totalRuntime()) {
      sim.StepForward(ntt::DiagFlags_None);

      const auto t = sim.time();
      real_t     ux1, ux2;
      if (t > sim.params()->totalRuntime() * HALF) {
        ux1 = r * omega * math::sin(omega * t);
        ux2 = r * omega * math::cos(omega * t);
      } else {
        ux1 = -r * omega * math::sin(omega * t);
        ux2 = r * omega * math::cos(omega * t);
      }

      Kokkos::parallel_for(
        "UpdParticle",
        1,
        Lambda(ntt::index_t p) {
          const auto x1_p = static_cast<real_t>(electrons.i1(p)) +
                            static_cast<real_t>(electrons.dx1(p));
          const auto x2_p = static_cast<real_t>(electrons.i2(p)) +
                            static_cast<real_t>(electrons.dx2(p));
          ntt::vec_t<ntt::Dim3> u_Cart { ZERO };
#ifdef MINKOWSKI_METRIC
          mblock.metric.v3_Cntrv2Cart({ x1_p, x2_p }, { ux1, ux2, ZERO }, u_Cart);
#else
          mblock.metric.v3_Cntrv2Cart({ x1_p, x2_p, ZERO },
                                      { ux1, ux2, ZERO },
                                      u_Cart);
#endif
          electrons.ux1(p) = u_Cart[0];
          electrons.ux2(p) = u_Cart[1];
          electrons.ux3(p) = u_Cart[2];
        });

      electrons.SyncHostDevice();
      auto em_h = Kokkos::create_mirror_view(mblock.em);
      Kokkos::deep_copy(em_h, mblock.em);

      const auto   i0 = 128 + N_GHOSTS, j0 = 128 + N_GHOSTS;
      const real_t x0 = 128.0;
      const real_t y0 = 128.0;
      q_A.push_back(
        em_h(i0, j0, ntt::em::ex1) * mblock.metric.sqrt_det_h({ x0 + HALF, y0 }) -
        em_h(i0 - 1, j0, ntt::em::ex1) *
          mblock.metric.sqrt_det_h({ x0 - HALF, y0 }) +
        em_h(i0, j0, ntt::em::ex2) * mblock.metric.sqrt_det_h({ x0, y0 + HALF }) -
        em_h(i0, j0 - 1, ntt::em::ex2) *
          mblock.metric.sqrt_det_h({ x0, y0 - HALF }));
      q_B.push_back(em_h(i0 + 1, j0, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 3.0 * HALF, y0 }) -
                    em_h(i0, j0, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + HALF, y0 }) +
                    em_h(i0 + 1, j0, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + ONE, y0 + HALF }) -
                    em_h(i0 + 1, j0 - 1, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + ONE, y0 - HALF }));
      q_C.push_back(em_h(i0 + 2, j0, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 5.0 * HALF, y0 }) -
                    em_h(i0 + 1, j0, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 3.0 * HALF, y0 }) +
                    em_h(i0 + 2, j0, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + TWO, y0 + HALF }) -
                    em_h(i0 + 2, j0 - 1, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + TWO, y0 - HALF }));
      q_D.push_back(em_h(i0, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + HALF, y0 + ONE }) -
                    em_h(i0 - 1, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 - HALF, y0 + ONE }) +
                    em_h(i0, j0 + 1, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0, y0 + 3.0 * HALF }) -
                    em_h(i0, j0, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0, y0 + HALF }));
      q_E.push_back(em_h(i0 + 1, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 3.0 * HALF, y0 + ONE }) -
                    em_h(i0, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + HALF, y0 + ONE }) +
                    em_h(i0 + 1, j0 + 1, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + ONE, y0 + 3.0 * HALF }) -
                    em_h(i0 + 1, j0, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + ONE, y0 + HALF }));
      q_F.push_back(em_h(i0 + 2, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 5.0 * HALF, y0 + ONE }) -
                    em_h(i0 + 1, j0 + 1, ntt::em::ex1) *
                      mblock.metric.sqrt_det_h({ x0 + 3.0 * HALF, y0 + ONE }) +
                    em_h(i0 + 2, j0 + 1, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + TWO, y0 + 3.0 * HALF }) -
                    em_h(i0 + 2, j0, ntt::em::ex2) *
                      mblock.metric.sqrt_det_h({ x0 + TWO, y0 + HALF }));
      if (enable_output) {
        q_file << q_A.back() << "," << q_B.back() << "," << q_C.back() << ","
               << q_D.back() << "," << q_E.back() << "," << q_F.back()
               << std::endl;

        prtl_file << electrons.i1_h(0) << "," << electrons.i2_h(0) << ","
                  << electrons.dx1_h(0) << "," << electrons.dx2_h(0) << ","
                  << electrons.ux1_h(0) << "," << electrons.ux2_h(0) << ","
                  << electrons.ux3_h(0) << std::endl;

        for (auto j { 0 }; j < 2; ++j) {
          for (auto i { -1 }; i < 3; ++i) {
            ex1_file << em_h(128 + N_GHOSTS + i, 128 + N_GHOSTS + j, ntt::em::ex1);
            if (i != 2 || j != 1) {
              ex1_file << ",";
            }
          }
        }
        ex1_file << std::endl;
        for (auto i { 0 }; i < 3; ++i) {
          for (auto j { -1 }; j < 2; ++j) {
            ex2_file << em_h(128 + N_GHOSTS + i, 128 + N_GHOSTS + j, ntt::em::ex2);
            if (i != 2 || j != 1) {
              ex2_file << ",";
            }
          }
        }
        ex2_file << std::endl;
      }
    }
    if (enable_output) {
      prtl_file.close();
      q_file.close();
      ex1_file.close();
      ex2_file.close();
      for (auto j { 0 }; j < 2; ++j) {
        for (auto i { -1 }; i < 3; ++i) {

          metric_file
            << mblock.metric.sqrt_det_h(
                 { (real_t)128.0 + (real_t)i + HALF, (real_t)128.0 + (real_t)j })
            << ",";
        }
      }
      for (auto i { 0 }; i < 3; ++i) {
        for (auto j { -1 }; j < 2; ++j) {
          metric_file << mblock.metric.sqrt_det_h(
            { (real_t)128.0 + (real_t)i, (real_t)128.0 + (real_t)j + HALF });
          if (i != 2 || j != 1) {
            metric_file << ",";
          }
        }
      }
      metric_file << std::endl;
      metric_file.close();
    }
    std::vector<real_t> q_error;
    auto                q_max   = std::numeric_limits<real_t>::min();
    auto                q_max_A = std::max_element(q_A.begin(), q_A.end());
    auto                q_max_B = std::max_element(q_B.begin(), q_B.end());
    auto                q_max_C = std::max_element(q_C.begin(), q_C.end());
    auto                q_max_D = std::max_element(q_D.begin(), q_D.end());
    auto                q_max_E = std::max_element(q_E.begin(), q_E.end());
    auto                q_max_F = std::max_element(q_F.begin(), q_F.end());
    q_max                       = std::max(*q_max_A, *q_max_B);
    q_max                       = std::max(q_max, *q_max_C);
    q_max                       = std::max(q_max, *q_max_D);
    q_max                       = std::max(q_max, *q_max_E);
    q_max                       = std::max(q_max, *q_max_F);
    for (std::size_t t { 0 }; t < q_A.size(); ++t) {
      q_error.push_back(
        math::abs(q_A[t] + q_B[t] + q_C[t] + q_D[t] + q_E[t] + q_F[t]) / q_max);
    }
    auto q_err_max = *std::max_element(q_error.begin(), q_error.end());
    if (q_err_max > 10.0 * std::numeric_limits<real_t>::epsilon()) {
      throw std::runtime_error("max(q_error) = " + std::to_string(q_err_max));
    }
  }

  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }

  ntt::GlobalFinalize();

  return 0;
}
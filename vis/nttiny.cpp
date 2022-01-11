#include "nttiny/vis.h"
#include "nttiny/api.h"

#include "global.h"
#include "cargs.h"
#include "input.h"
#include "pic.h"

#include <toml/toml.hpp>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

struct NTTSimulationVis : public nttiny::SimulationAPI<float> {
  int nx1, nx2;
  ntt::PIC<ntt::Dimension::TWO_D>& m_sim;
  nttiny::Data<float> m_ex1, m_ex2, m_ex3;
  nttiny::Data<float> m_bx1, m_bx2, m_bx3;

  real_t m_time;

  std::vector<std::unique_ptr<nttiny::Data<float>>> prtl_pointers;

  NTTSimulationVis(ntt::PIC<ntt::Dimension::TWO_D>& sim)
    : nttiny::SimulationAPI<float> {sim.mblock().metric->label},
      nx1(sim.mblock().Ni() + 2 * ntt::N_GHOSTS),
      nx2(sim.mblock().Nj() + 2 * ntt::N_GHOSTS),
      m_sim(sim),
      m_ex1 {nx1, nx2},
      m_ex2 {nx1, nx2},
      m_ex3 {nx1, nx2},
      m_bx1 {nx1, nx2},
      m_bx2 {nx1, nx2},
      m_bx3 {nx1, nx2} {
    this->m_timestep = 0;
    this->m_time = 0.0;

    // if (this->coords == "qspherical") {
    //   m_x1x2_extent[0] = m_sim.mblock().metric->x1_min;
    //   m_x1x2_extent[1] = m_sim.mblock().metric->x1_max;

    //   for (int i {ntt::N_GHOSTS}; i <= nx1 - ntt::N_GHOSTS; ++i) {
    //     auto i_ {(real_t)(i - ntt::N_GHOSTS)};
    //     auto j_ {ZERO};
    //     auto [r_, th_] = m_sim.mblock().grid->coord_CU_to_Sph(i_, j_);
    //     m_ex1.grid_x1[i] = r_;
    //   }
    //   for (int i {ntt::N_GHOSTS - 1}; i >= 0; --i) {
    //     m_ex1.grid_x1[i] = m_ex1.grid_x1[i + 1] - (m_ex1.grid_x1[ntt::N_GHOSTS + 1] - m_ex1.grid_x1[ntt::N_GHOSTS]);
    //   }
    //   for (int i {nx1 - ntt::N_GHOSTS + 1}; i <= nx1; ++i) {
    //     m_ex1.grid_x1[i] = m_ex1.grid_x1[i - 1] + (m_ex1.grid_x1[nx1 - ntt::N_GHOSTS] - m_ex1.grid_x1[nx1 - ntt::N_GHOSTS - 1]);
    //   }

    //   for (int j {0}; j <= nx2; ++j) {
    //     auto i_ {ntt::ZERO};
    //     auto j_ {(ntt::real_t)(j - ntt::N_GHOSTS)};
    //     auto [r_, th_] = m_sim.get_meshblock().grid->coord_CU_to_Sph(i_, j_);
    //     m_ex1.grid_x2[j] = th_;
    //   }
    //   auto [r1_, th1_] = m_sim.get_meshblock().grid->coord_CU_to_Sph(ntt::ZERO, (ntt::real_t)(-ntt::N_GHOSTS));
    //   auto [r2_, th2_] = m_sim.get_meshblock().grid->coord_CU_to_Sph(ntt::ZERO, (ntt::real_t)(nx2 - ntt::N_GHOSTS));
    //   m_x1x2_extent[2] = th1_;
    //   m_x1x2_extent[3] = th2_;
    // } else {
      auto sx1 {m_sim.mblock().metric->x1_max - m_sim.mblock().metric->x1_min};
      auto dx1 {sx1 / m_sim.mblock().metric->nx1};
      auto sx2 {m_sim.mblock().metric->x2_max - m_sim.mblock().metric->x2_min};
      auto dx2 {sx2 / m_sim.mblock().metric->nx2};
      m_x1x2_extent[0] = m_sim.mblock().metric->x1_min - dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[1] = m_sim.mblock().metric->x1_max + dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[2] = m_sim.mblock().metric->x2_min - dx2 * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.mblock().metric->x2_max + dx2 * ntt::N_GHOSTS;
      for (int i {0}; i <= nx1; ++i) {
        m_ex1.grid_x1[i] = m_x1x2_extent[0] + (m_x1x2_extent[1] - m_x1x2_extent[0]) * (double)(i) / (double)(nx1);
      }
      for (int j {0}; j <= nx2; ++j) {
        m_ex1.grid_x2[j] = m_x1x2_extent[2] + (m_x1x2_extent[3] - m_x1x2_extent[2]) * (double)(j) / (double)(nx2);
      }
    // }

    this->fields.insert({{"ex1", &(this->m_ex1)},
                         {"ex2", &(this->m_ex2)},
                         {"ex3", &(this->m_ex3)},
                         {"bx1", &(this->m_bx1)},
                         {"bx2", &(this->m_bx2)},
                         {"bx3", &(this->m_bx3)}
                       });

    for (auto const& [fld, arr] : this->fields) {
      for (int i {0}; i <= nx1; ++i) {
        arr->grid_x1[i] = m_ex1.grid_x1[i];
      }
      for (int i {0}; i <= nx2; ++i) {
        arr->grid_x2[i] = m_ex1.grid_x2[i];
      }
    }
    setData();
  }

  void setData() override {
    for (int j{0}; j < nx2; ++j) {
      for (int i{0}; i < nx1; ++i) {
        auto i_ {(real_t)(i - ntt::N_GHOSTS)};
        auto j_ {(real_t)(j - ntt::N_GHOSTS)};

        auto ex1_cnt {m_sim.mblock().em(i, j, ntt::em::ex1)};
        auto ex2_cnt {m_sim.mblock().em(i, j, ntt::em::ex2)};
        auto ex3_cnt {m_sim.mblock().em(i, j, ntt::em::ex3)};
        auto bx1_cnt {m_sim.mblock().em(i, j, ntt::em::bx1)};
        auto bx2_cnt {m_sim.mblock().em(i, j, ntt::em::bx2)};
        auto bx3_cnt {m_sim.mblock().em(i, j, ntt::em::bx3)};

        ntt::vec_t<ntt::Dimension::THREE_D> ex1_hat, ex2_hat, ex3_hat;
        ntt::vec_t<ntt::Dimension::THREE_D> bx1_hat, bx2_hat, bx3_hat;
        m_sim.mblock().metric->v_Cntrv2Hat({i_ + HALF, j_}, {ex1_cnt, ZERO, ZERO}, ex1_hat);
        m_sim.mblock().metric->v_Cntrv2Hat({i_, j_ + HALF}, {ZERO, ex2_cnt, ZERO}, ex2_hat);
        m_sim.mblock().metric->v_Cntrv2Hat({i_, j_}, {ZERO, ZERO, ex3_cnt}, ex3_hat);

        m_sim.mblock().metric->v_Cntrv2Hat({i_, j_ + HALF}, {bx1_cnt, ZERO, ZERO}, bx1_hat);
        m_sim.mblock().metric->v_Cntrv2Hat({i_ + HALF, j_}, {ZERO, bx2_cnt, ZERO}, bx2_hat);
        m_sim.mblock().metric->v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {ZERO, ZERO, bx3_cnt}, bx3_hat);

        // convert from contravariant to hatted
        m_ex1.set(i, j, ex1_hat[0]);
        m_ex2.set(i, j, ex2_hat[1]);
        m_ex3.set(i, j, ex3_hat[2]);
        m_bx1.set(i, j, bx1_hat[0]);
        m_bx2.set(i, j, bx2_hat[1]);
        m_bx3.set(i, j, bx3_hat[2]);
      }
    }
  }
  void stepFwd() override {
    m_sim.step_forward(m_time);
    setData();
    ++m_timestep;
    m_time += m_sim.mblock().timestep();
  }
  void restart() override {
    m_sim.initializeSetup();
    setData();
    m_time = 0.0;
    m_timestep = 0;
  }
  void stepBwd() override {
    m_sim.step_backward(m_time);
    setData();
    --m_timestep;
    m_time += m_sim.mblock().timestep();
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto inputdata = toml::parse(static_cast<std::string>(inputfilename));

    ntt::PIC<ntt::Dimension::TWO_D> sim(inputdata);
    sim.initialize();
    sim.initializeSetup();
    sim.verify();
    sim.printDetails();
    NTTSimulationVis visApi(sim);

    nttiny::Visualization<float> vis;
    vis.setTPSLimit(30.0f);
    vis.bindSimulation(&visApi);
    vis.loop();
  }
  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}

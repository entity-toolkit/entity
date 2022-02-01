#include "nttiny/vis.h"
#include "nttiny/api.h"

#if (SIMTYPE == PIC_SIMTYPE)
#  include "pic.h"
#  define SIMULATION_CONTAINER PIC
#elif (SIMTYPE == GRPIC_SIMTYPE)
#  include "grpic.h"
#  define SIMULATION_CONTAINER GRPIC
#endif

#include "global.h"
#include "cargs.h"
#include "input.h"

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
  std::vector<std::unique_ptr<nttiny::Data<float>>> prtl_pointers;

  NTTSimulationVis(ntt::PIC<ntt::Dimension::TWO_D>& sim)
    : nttiny::SimulationAPI<float> {sim.mblock().metric.label},
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

    if (this->coords == "qspherical") {
      m_x1x2_extent[0] = m_sim.mblock().metric.x1_min;
      m_x1x2_extent[1] = m_sim.mblock().metric.x1_max;

      for (int i {ntt::N_GHOSTS}; i <= nx1 - ntt::N_GHOSTS; ++i) {
        auto i_ {(real_t)(i - ntt::N_GHOSTS)};
        auto j_ {ZERO};
        ntt::coord_t<ntt::Dimension::TWO_D> rth_;
        m_sim.mblock().metric.x_Code2Sph({i_, j_}, rth_);
        m_ex1.grid_x1[i] = rth_[0];
      }
      for (int i {ntt::N_GHOSTS - 1}; i >= 0; --i) {
        m_ex1.grid_x1[i] = m_ex1.grid_x1[i + 1] - (m_ex1.grid_x1[ntt::N_GHOSTS + 1] - m_ex1.grid_x1[ntt::N_GHOSTS]);
      }
      for (int i {nx1 - ntt::N_GHOSTS + 1}; i <= nx1; ++i) {
        m_ex1.grid_x1[i] = m_ex1.grid_x1[i - 1] + (m_ex1.grid_x1[nx1 - ntt::N_GHOSTS] - m_ex1.grid_x1[nx1 - ntt::N_GHOSTS - 1]);
      }

      for (int j {0}; j <= nx2; ++j) {
        auto i_ {ZERO};
        auto j_ {(real_t)(j - ntt::N_GHOSTS)};
        ntt::coord_t<ntt::Dimension::TWO_D> rth_;
        m_sim.mblock().metric.x_Code2Sph({i_, j_}, rth_);
        m_ex1.grid_x2[j] = rth_[1];
      }
      ntt::coord_t<ntt::Dimension::TWO_D> rth1_, rth2_;
      m_sim.mblock().metric.x_Code2Sph({ZERO, (real_t)(-ntt::N_GHOSTS)}, rth1_);
      m_sim.mblock().metric.x_Code2Sph({ZERO, (real_t)(nx2 - ntt::N_GHOSTS)}, rth2_);
      m_x1x2_extent[2] = rth1_[1];
      m_x1x2_extent[3] = rth2_[1];
    } else {
      auto sx1 {m_sim.mblock().metric.x1_max - m_sim.mblock().metric.x1_min};
      auto dx1 {sx1 / m_sim.mblock().metric.nx1};
      auto sx2 {m_sim.mblock().metric.x2_max - m_sim.mblock().metric.x2_min};
      auto dx2 {sx2 / m_sim.mblock().metric.nx2};
      m_x1x2_extent[0] = m_sim.mblock().metric.x1_min - dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[1] = m_sim.mblock().metric.x1_max + dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[2] = m_sim.mblock().metric.x2_min - dx2 * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.mblock().metric.x2_max + dx2 * ntt::N_GHOSTS;
      for (int i {0}; i <= nx1; ++i) {
        m_ex1.grid_x1[i] = m_x1x2_extent[0] + (m_x1x2_extent[1] - m_x1x2_extent[0]) * (double)(i) / (double)(nx1);
      }
      for (int j {0}; j <= nx2; ++j) {
        m_ex1.grid_x2[j] = m_x1x2_extent[2] + (m_x1x2_extent[3] - m_x1x2_extent[2]) * (double)(j) / (double)(nx2);
      }
    }

    this->fields.insert({{"ex1", &(this->m_ex1)},
                         {"ex2", &(this->m_ex2)},
                         {"ex3", &(this->m_ex3)},
                         {"bx1", &(this->m_bx1)},
                         {"bx2", &(this->m_bx2)},
                         {"bx3", &(this->m_bx3)}
                       });

    int s {0}, i {0};
    for (auto& species : m_sim.mblock().particles) {
      auto nprt {m_sim.mblock().particles[s].npart()};
      auto x_prtl {std::make_unique<nttiny::Data<float>>(nprt, 1)};
      auto y_prtl {std::make_unique<nttiny::Data<float>>(nprt, 1)};
      this->prtl_pointers.push_back(std::move(x_prtl));
      this->prtl_pointers.push_back(std::move(y_prtl));
      this->particles.insert({species.label(), {(this->prtl_pointers[i].get()), (this->prtl_pointers[i + 1].get())}});
      s ++;
      i += 2;
    }
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
        real_t ex1_cnt, ex2_cnt, ex3_cnt;
        real_t bx1_cnt, bx2_cnt, bx3_cnt;

        if ((i < ntt::N_GHOSTS) || (j < ntt::N_GHOSTS) || (i >= nx2 - ntt::N_GHOSTS) || (j >= nx1 - ntt::N_GHOSTS)) {
          ex1_cnt = m_sim.mblock().em(i, j, ntt::em::ex1);
          ex2_cnt = m_sim.mblock().em(i, j, ntt::em::ex2);
          ex3_cnt = m_sim.mblock().em(i, j, ntt::em::ex3);
          bx1_cnt = m_sim.mblock().em(i, j, ntt::em::bx1);
          bx2_cnt = m_sim.mblock().em(i, j, ntt::em::bx2);
          bx3_cnt = m_sim.mblock().em(i, j, ntt::em::bx3);
        } else {
          ex1_cnt = 0.5 * (m_sim.mblock().em(i, j, ntt::em::ex1) + m_sim.mblock().em(i, j + 1, ntt::em::ex1));
          ex2_cnt = 0.5 * (m_sim.mblock().em(i, j, ntt::em::ex2) + m_sim.mblock().em(i + 1, j, ntt::em::ex2));
          ex3_cnt = 0.25
                    * (m_sim.mblock().em(i, j, ntt::em::ex3) + m_sim.mblock().em(i + 1, j, ntt::em::ex3)
                       + m_sim.mblock().em(i, j + 1, ntt::em::ex3) + m_sim.mblock().em(i + 1, j + 1, ntt::em::ex3));
          bx1_cnt = 0.5 * (m_sim.mblock().em(i, j, ntt::em::bx1) + m_sim.mblock().em(i + 1, j, ntt::em::bx1));
          bx2_cnt = 0.5 * (m_sim.mblock().em(i, j, ntt::em::bx2) + m_sim.mblock().em(i, j + 1, ntt::em::bx2));
          bx3_cnt = m_sim.mblock().em(i, j, ntt::em::bx3);
        }

        ntt::vec_t<ntt::Dimension::THREE_D> e_hat, b_hat;

#if (SIMTYPE == PIC_SIMTYPE)
        m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {ex1_cnt, ex2_cnt, ex3_cnt}, e_hat);
        m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {bx1_cnt, bx2_cnt, bx3_cnt}, b_hat);
#elif (SIMTYPE == GRPIC_SIMTYPE)
        e_hat[0] = ex1_cnt;
        e_hat[1] = ex2_cnt;
        e_hat[2] = ex3_cnt;
        b_hat[0] = bx1_cnt;
        b_hat[1] = bx2_cnt;
        b_hat[2] = bx3_cnt;
#endif
        // convert from contravariant to hatted
        m_ex1.set(i, j, e_hat[0]);
        m_ex2.set(i, j, e_hat[1]);
        m_ex3.set(i, j, e_hat[2]);
        m_bx1.set(i, j, b_hat[0]);
        m_bx2.set(i, j, b_hat[1]);
        m_bx3.set(i, j, b_hat[2]);
        // m_ex1.set(i, j, ex1_cnt);
        // m_ex2.set(i, j, ex2_cnt);
        // m_ex3.set(i, j, ex3_cnt);
        // m_bx1.set(i, j, bx1_cnt);
        // m_bx2.set(i, j, bx2_cnt);
        // m_bx3.set(i, j, bx3_cnt);
      }
    }
    int i {0};
    for (auto& species : m_sim.mblock().particles) {
      for (int k {0}; k < this->prtl_pointers[i]->get_size(0); ++k) {
        float x1 {(float)(species.i1(k)) + species.dx1(k)};
        float x2 {(float)(species.i2(k)) + species.dx2(k)};
        ntt::coord_t<ntt::Dimension::TWO_D> xy {ZERO, ZERO};
        m_sim.mblock().metric.x_Code2Cart({x1, x2}, xy);
        this->prtl_pointers[i]->set(k, 0, xy[0]);
        this->prtl_pointers[i + 1]->set(k, 0, xy[1]);
      }
      i += 2;
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
    m_sim.fieldBoundaryConditions(ZERO);
    setData();
    m_time = 0.0;
    m_timestep = 0;
  }
  void stepBwd() override {
    m_sim.step_backward(m_time);
    setData();
    --m_timestep;
    m_time -= m_sim.mblock().timestep();
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto inputdata = toml::parse(static_cast<std::string>(inputfilename));

    ntt::SIMULATION_CONTAINER<ntt::Dimension::TWO_D> sim(inputdata);
    sim.initialize();
    sim.initializeSetup();
    sim.verify();
    sim.printDetails();
    sim.fieldBoundaryConditions(ZERO);
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

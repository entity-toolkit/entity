#include "nttiny/vis.h"
#include "nttiny/api.h"

#include "global.h"
#include "cargs.h"
#include "input.h"
#include "simulation.h"

#include <toml/toml.hpp>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

class NTTSimulationVis : public nttiny::SimulationAPI<float> {
public:
  int nx1, nx2;
  ntt::Simulation<ntt::TWO_D>& m_sim;
  nttiny::Data<float> m_ex1, m_ex2, m_ex3;
  nttiny::Data<float> m_bx1, m_bx2, m_bx3;

  ntt::real_t m_time;

  std::vector<std::unique_ptr<nttiny::Data<float>>> prtl_pointers;

  NTTSimulationVis(ntt::Simulation<ntt::TWO_D>& sim)
    : nttiny::SimulationAPI<float>{sim.get_meshblock().m_coord_system->label},
      nx1(sim.get_meshblock().Ni + 2 * ntt::N_GHOSTS),
      nx2(sim.get_meshblock().Nj + 2 * ntt::N_GHOSTS),
      m_sim(sim),
      m_ex1{nx1, nx2}, m_ex2{nx1, nx2}, m_ex3{nx1, nx2},
      m_bx1{nx1, nx2}, m_bx2{nx1, nx2}, m_bx3{nx1, nx2} {
    this->m_timestep = 0;
    this->m_time = 0.0;

    if (this->coords == "qspherical") {
      // m_x1x2_extent[0] = m_sim.get_meshblock().m_coord_system->x1min_PHU();
      // m_x1x2_extent[1] = m_sim.get_meshblock().m_coord_system->x1max_PHU();
      //
      // int j {0};
      // for (int i {ntt::N_GHOSTS}; i <= nx1 - ntt::N_GHOSTS; ++i) {
      //   double x1 {m_x1x2_extent[0] + (m_x1x2_extent[1] - m_x1x2_extent[0]) * (double)(j) / (double)(nx1 - 2 * ntt::N_GHOSTS)};
      //   m_ex1.grid_x1[i] = m_sim.get_meshblock().m_coord_system->getSpherical_r(x1, 0.0);
      //   ++j;
      // }
      // for (int i {ntt::N_GHOSTS - 1}; i >= 0; --i) {
      //   m_ex1.grid_x1[i] = m_ex1.grid_x1[i + 1] - (m_ex1.grid_x1[ntt::N_GHOSTS + 1] - m_ex1.grid_x1[ntt::N_GHOSTS]);
      // }
      // for (int i {nx1 - ntt::N_GHOSTS + 1}; i <= nx1; ++i) {
      //   m_ex1.grid_x1[i] = m_ex1.grid_x1[i - 1] + (m_ex1.grid_x1[nx1 - ntt::N_GHOSTS] - m_ex1.grid_x1[nx1 - ntt::N_GHOSTS - 1]);
      // }
      // m_x1x2_extent[0] = std::exp(m_sim.get_meshblock().m_extent[0]);
      // m_x1x2_extent[1] = std::exp(m_sim.get_meshblock().m_extent[1]);
      //
      // m_x1x2_extent[2] = m_sim.get_meshblock().m_extent[2] - m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
      // m_x1x2_extent[3] = m_sim.get_meshblock().m_extent[3] + m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
      // for (int j {0}; j <= nx2; ++j) {
      //   double x2 {m_x1x2_extent[2] + (m_x1x2_extent[3] - m_x1x2_extent[2]) * (double)(j) / (double)(nx2)};
      //   m_ex1.grid_x2[j] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, x2);
      // }
      // m_x1x2_extent[2] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, m_x1x2_extent[2]);
      // m_x1x2_extent[3] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, m_x1x2_extent[3]);
    } else {
      auto sx1 {m_sim.get_meshblock().m_coord_system->x1_max - m_sim.get_meshblock().m_coord_system->x1_min};
      auto dx1 {sx1 / (ntt::real_t)(m_sim.get_meshblock().m_coord_system->Nx1)};
      auto sx2 {m_sim.get_meshblock().m_coord_system->x2_max - m_sim.get_meshblock().m_coord_system->x2_min};
      auto dx2 {sx2 / (ntt::real_t)(m_sim.get_meshblock().m_coord_system->Nx2)};
      m_x1x2_extent[0] = m_sim.get_meshblock().m_coord_system->x1_min - dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[1] = m_sim.get_meshblock().m_coord_system->x1_max + dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[2] = m_sim.get_meshblock().m_coord_system->x2_min - dx2 * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.get_meshblock().m_coord_system->x2_max + dx2 * ntt::N_GHOSTS;
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

  ~NTTSimulationVis() = default;
  void setData() override {
    for (int j{0}; j < nx2; ++j) {
      for (int i{0}; i < nx1; ++i) {
        auto i_ {(ntt::real_t)(i - ntt::N_GHOSTS)};
        auto j_ {(ntt::real_t)(j - ntt::N_GHOSTS)};

        auto ex1_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex1)};
        auto ex2_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex2)};
        auto ex3_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex3)};
        auto bx1_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx1)};
        auto bx2_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx2)};
        auto bx3_CNT {m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx3)};

        // convert from contravariant to hatted
        m_ex1.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x1(ex1_CNT, i_ + 0.5, j_));
        m_ex2.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x2(ex2_CNT, i_, j_ + 0.5));
        m_ex3.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x3(ex3_CNT, i_, j_));
        m_bx1.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x1(bx1_CNT, i_, j_ + 0.5));
        m_bx2.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x2(bx2_CNT, i_ + 0.5, j_));
        m_bx3.set(i, j, m_sim.get_meshblock().m_coord_system->vec_CNT_to_HAT_x3(bx3_CNT, i_ + 0.5, j_ + 0.5));
        // m_bx3.set(i, j, bx3_CNT);
      }
    }
  }
  void stepFwd() override {
    m_sim.step_forward(m_time);
    setData();
    ++m_timestep;
    m_time += m_sim.get_params().get_timestep();
  }
  void restart() override {
    m_sim.userInitialize();
    setData();
    m_time = 0.0;
    m_timestep = 0;
  }
  void stepBwd() override {
    m_sim.step_backward(m_time);
    setData();
    --m_timestep;
    m_time += m_sim.get_params().get_timestep();
  }
};

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::DEF_input_filename);
    auto outputpath = cl_args.getArgument("-output", ntt::DEF_output_path);
    auto inputdata = toml::parse(static_cast<std::string>(inputfilename));

    ntt::Simulation<ntt::TWO_D> sim(inputdata);
    sim.setIO(inputfilename, outputpath);
    sim.userInitialize();
    sim.verify();
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

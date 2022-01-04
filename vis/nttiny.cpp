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
    : nttiny::SimulationAPI<float>{sim.get_meshblock().m_coord_system->get_label()},
      nx1(sim.get_meshblock().get_n1() + 2 * ntt::N_GHOSTS),
      nx2(sim.get_meshblock().get_n2() + 2 * ntt::N_GHOSTS),
      m_sim(sim),
      m_ex1{nx1, nx2}, m_ex2{nx1, nx2}, m_ex3{nx1, nx2},
      m_bx1{nx1, nx2}, m_bx2{nx1, nx2}, m_bx3{nx1, nx2} {
    this->m_timestep = 0;
    this->m_time = 0.0;

    if (this->coords == "qspherical") {
      m_x1x2_extent[0] = m_sim.get_meshblock().m_extent[0];
      m_x1x2_extent[1] = m_sim.get_meshblock().m_extent[1];

      int j {0};
      for (int i {ntt::N_GHOSTS}; i <= nx1 - ntt::N_GHOSTS; ++i) {
        double x1 {m_x1x2_extent[0] + (m_x1x2_extent[1] - m_x1x2_extent[0]) * (double)(j) / (double)(nx1 - 2 * ntt::N_GHOSTS)};
        m_ex1.grid_x1[i] = m_sim.get_meshblock().m_coord_system->getSpherical_r(x1, 0.0);
        ++j;
      }
      for (int i {ntt::N_GHOSTS - 1}; i >= 0; --i) {
        m_ex1.grid_x1[i] = m_ex1.grid_x1[i + 1] - (m_ex1.grid_x1[ntt::N_GHOSTS + 1] - m_ex1.grid_x1[ntt::N_GHOSTS]);
      }
      for (int i {nx1 - ntt::N_GHOSTS + 1}; i <= nx1; ++i) {
        m_ex1.grid_x1[i] = m_ex1.grid_x1[i - 1] + (m_ex1.grid_x1[nx1 - ntt::N_GHOSTS] - m_ex1.grid_x1[nx1 - ntt::N_GHOSTS - 1]);
      }
      m_x1x2_extent[0] = std::exp(m_sim.get_meshblock().m_extent[0]);
      m_x1x2_extent[1] = std::exp(m_sim.get_meshblock().m_extent[1]);

      m_x1x2_extent[2] = m_sim.get_meshblock().m_extent[2] - m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.get_meshblock().m_extent[3] + m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
      for (int j {0}; j <= nx2; ++j) {
        double x2 {m_x1x2_extent[2] + (m_x1x2_extent[3] - m_x1x2_extent[2]) * (double)(j) / (double)(nx2)};
        m_ex1.grid_x2[j] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, x2);
      }
      m_x1x2_extent[2] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, m_x1x2_extent[2]);
      m_x1x2_extent[3] = m_sim.get_meshblock().m_coord_system->getSpherical_theta(0.0, m_x1x2_extent[3]);
    } else {
      m_x1x2_extent[0] = m_sim.get_meshblock().m_extent[0] - m_sim.get_meshblock().get_dx1() * ntt::N_GHOSTS;
      m_x1x2_extent[1] = m_sim.get_meshblock().m_extent[1] + m_sim.get_meshblock().get_dx1() * ntt::N_GHOSTS;
      m_x1x2_extent[2] = m_sim.get_meshblock().m_extent[2] - m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.get_meshblock().m_extent[3] + m_sim.get_meshblock().get_dx2() * ntt::N_GHOSTS;
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
    auto dx1 {m_sim.get_meshblock().get_dx1()};
    auto dx2 {m_sim.get_meshblock().get_dx2()};
    for (int j{0}; j < nx2; ++j) {
      for (int i{0}; i < nx1; ++i) {
        auto x1 {m_sim.get_meshblock().convert_iTOx1(i)};
        auto x2 {m_sim.get_meshblock().convert_jTOx2(j)};
        // convert from contravariant components to local orthonormal basis

        auto ex1 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x1(m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex1), x1 + 0.5 * dx1, x2)};
        m_ex1.set(i, j, ex1);

        auto ex2 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x2(m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex2), x1, x2 + 0.5 * dx2)};
        m_ex2.set(i, j, ex2);

        auto ex3 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x3(m_sim.get_meshblock().em_fields(i, j, ntt::fld::ex3), x1, x2)};
        m_ex3.set(i, j, ex3);

        auto bx1 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x1(m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx1), x1, x2 + 0.5 * dx2)};
        m_bx1.set(i, j, bx1);

        auto bx2 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x2(m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx2), x1 + 0.5 * dx1, x2)};
        m_bx2.set(i, j, bx2);

        auto bx3 {m_sim.get_meshblock().m_coord_system->convert_CNT_to_LOC_x3(m_sim.get_meshblock().em_fields(i, j, ntt::fld::bx3), x1 + 0.5 * dx1, x2 + 0.5 * dx2)};
        m_bx3.set(i, j, bx3);
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

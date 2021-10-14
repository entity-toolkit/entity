#include "nttiny/vis.h"
#include "nttiny/api.h"

#include "global.h"
#include "cargs.h"
#include "input.h"
#include "simulation.h"
#include "constants.h"

#include <toml/toml.hpp>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

class NTTSimulationVis : public nttiny::SimulationAPI<float> {
public:
  ntt::Simulation<ntt::TWO_D>& m_sim;
  nttiny::Data<float> m_ex;
  nttiny::Data<float> m_bx;
  nttiny::Data<float> m_ey;
  nttiny::Data<float> m_by;
  nttiny::Data<float> m_ez;
  nttiny::Data<float> m_bz;

  ntt::real_t m_time;

  std::vector<std::unique_ptr<nttiny::Data<float>>> prtl_pointers;

  // nttiny::Data<float> electrons_x;
  // nttiny::Data<float> electrons_y;
  // nttiny::Data<float> positrons_x;
  // nttiny::Data<float> positrons_y;

  NTTSimulationVis(ntt::Simulation<ntt::TWO_D>& sim)
    : nttiny::SimulationAPI<float>{0, 0}, m_sim(sim) {
    m_sx = m_sim.get_params().get_resolution()[0];
    m_sy = m_sim.get_params().get_resolution()[1];

    m_x1x2_extent[0] = static_cast<float>(m_sim.get_params().get_extent()[0]);
    m_x1x2_extent[1] = static_cast<float>(m_sim.get_params().get_extent()[1]);
    m_x1x2_extent[2] = static_cast<float>(m_sim.get_params().get_extent()[2]);
    m_x1x2_extent[3] = static_cast<float>(m_sim.get_params().get_extent()[3]);
    m_timestep = 0;
    m_time = 0;

    fields.insert({{"ex", &(m_ex)},
                   {"bx", &(m_bx)},
                   {"ey", &(m_ey)},
                   {"by", &(m_by)},
                   {"ez", &(m_ez)},
                   {"bz", &(m_bz)}});
    for (auto &f : fields) {
      f.second->allocate(m_sx * m_sy);
      f.second->set_size(1, m_sx);
      f.second->set_size(0, m_sy);
      f.second->set_dimension(2);
    }

    int i{0};
    for (auto& species : m_sim.get_meshblock().particles) {
      auto x_prtl {std::make_unique<nttiny::Data<float>>()};
      auto y_prtl {std::make_unique<nttiny::Data<float>>()};
      x_prtl->m_data = m_sim.get_meshblock().particles[i / 2].m_x1.data();
      y_prtl->m_data = m_sim.get_meshblock().particles[i / 2].m_x2.data();
      x_prtl->set_size(0, m_sim.get_meshblock().particles[i / 2].get_npart());
      y_prtl->set_size(0, m_sim.get_meshblock().particles[i / 2].get_npart());
      this->prtl_pointers.push_back(std::move(x_prtl));
      this->prtl_pointers.push_back(std::move(y_prtl));
      this->particles.insert({species.m_label,
                              {(this->prtl_pointers[i].get()),
                               (this->prtl_pointers[i + 1].get())}});
      i += 2;
    }
    setData();
  }
  ~NTTSimulationVis() = default;
  void setData() override {
    // TODO: there might be an easier way to map
    for (int j{0}; j < m_sy; ++j) {
      for (int i{0}; i < m_sx; ++i) {
        int lind{i + j * m_sx};
        m_ex.set(lind, m_sim.get_meshblock().ex1(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
        m_ey.set(lind, m_sim.get_meshblock().ex2(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
        m_ez.set(lind, m_sim.get_meshblock().ex3(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
        m_bx.set(lind, m_sim.get_meshblock().bx1(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
        m_by.set(lind, m_sim.get_meshblock().bx2(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
        m_bz.set(lind, m_sim.get_meshblock().bx3(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS));
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
    // --m_timestep;
    // m_sim.step_forward();
    // setData();
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
    auto resolution = ntt::readFromInput<std::vector<std::size_t>>(inputdata, "domain", "resolution");

    ntt::Simulation<ntt::TWO_D> sim(inputdata);
    sim.setIO(inputfilename, outputpath);
    sim.userInitialize();
    sim.verify();
    // sim.printDetails();
    // sim.mainloop();
    // sim.finalize();
    NTTSimulationVis visApi(sim);

    nttiny::Visualization<float> vis;
    vis.setTPSLimit(12.0f);
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

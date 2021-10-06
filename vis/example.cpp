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
  ntt::Simulation<ntt::Two_D>& m_sim;
  nttiny::Data<float> m_ex;
  nttiny::Data<float> m_bx;
  nttiny::Data<float> m_ey;
  nttiny::Data<float> m_by;
  nttiny::Data<float> m_ez;
  nttiny::Data<float> m_bz;

  ntt::real_t m_time;

  NTTSimulationVis(ntt::Simulation<ntt::Two_D>& sim)
    : nttiny::SimulationAPI<float>{0, 0}, m_sim(sim) {
    m_sx = m_sim.get_params().get_resolution()[0];
    m_sy = m_sim.get_params().get_resolution()[1];

    m_ex.allocate(m_sx * m_sy);
    m_bx.allocate(m_sx * m_sy);
    m_ex.set_size(1, m_sx);
    m_ex.set_size(0, m_sy);
    m_bx.set_size(1, m_sx);
    m_bx.set_size(0, m_sy);
    m_ex.set_dimension(2);
    m_bx.set_dimension(2);

    m_ey.allocate(m_sx * m_sy);
    m_by.allocate(m_sx * m_sy);
    m_ey.set_size(1, m_sx);
    m_ey.set_size(0, m_sy);
    m_by.set_size(1, m_sx);
    m_by.set_size(0, m_sy);
    m_ey.set_dimension(2);
    m_by.set_dimension(2);

    m_ez.allocate(m_sx * m_sy);
    m_bz.allocate(m_sx * m_sy);
    m_ez.set_size(1, m_sx);
    m_ez.set_size(0, m_sy);
    m_bz.set_size(1, m_sx);
    m_bz.set_size(0, m_sy);
    m_ez.set_dimension(2);
    m_bz.set_dimension(2);

    m_x1x2_extent[0] = static_cast<float>(m_sim.get_params().get_extent()[0]);
    m_x1x2_extent[1] = static_cast<float>(m_sim.get_params().get_extent()[1]);
    m_x1x2_extent[2] = static_cast<float>(m_sim.get_params().get_extent()[2]);
    m_x1x2_extent[3] = static_cast<float>(m_sim.get_params().get_extent()[3]);
    m_timestep = 0;
    m_time = 0;

    setData();
    fields.insert({{"ex", &(m_ex)},
                   {"bx", &(m_bx)},
                   {"ey", &(m_ey)},
                   {"by", &(m_by)},
                   {"ez", &(m_ez)},
                   {"bz", &(m_bz)}});
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
  void stepBwd() override {
    --m_timestep;
    // m_sim.step_forward();
    setData();
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

    ntt::Simulation<ntt::Two_D> sim(inputdata);
    sim.setIO(inputfilename, outputpath);
    sim.initialize();
    sim.verify();
    // sim.printDetails();
    // sim.mainloop();
    // sim.finalize();
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

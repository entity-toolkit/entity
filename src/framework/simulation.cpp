#include "global.h"
#include "simulation.h"
#include "metric.h"

#include "output_csv.h"
#include "utils.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <stdexcept>

namespace ntt {

  template <Dimension D, SimulationType S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_sim_params {inputdata, D},
      m_pGen {m_sim_params},
      m_mblock {m_sim_params.resolution(),
                m_sim_params.extent(),
                m_sim_params.metric_parameters(),
                m_sim_params.species()},
      m_random_pool{RandomSeed} {
    m_mblock.random_pool_ptr = &m_random_pool;
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::initialize() {
    m_mblock.boundaries = m_sim_params.boundaries();

    // find timestep and effective cell size
    m_mblock.set_min_cell_size(m_mblock.metric.findSmallestCell());
    m_mblock.set_timestep(m_sim_params.cfl() * m_mblock.min_cell_size());

    WaitAndSynchronize();
    PLOGD << "Simulation initialized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::initializeSetup() {
    m_pGen.userInitFields(m_sim_params, m_mblock);
    m_pGen.userInitParticles(m_sim_params, m_mblock);

    WaitAndSynchronize();
    PLOGD << "Setup initialized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::verify() {
    // m_sim_params.verify();
    // mblock.verify(m_sim_params);
    WaitAndSynchronize();
    PLOGD << "Prerun check passed.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::printDetails() {
    PLOGI << "[Simulation details]";
    PLOGI << "   title: " << m_sim_params.title();
    PLOGI << "   type: " << stringifySimulationType(S);
    PLOGI << "   total runtime: " << m_sim_params.total_runtime();
    PLOGI << "   dt: " << m_mblock.timestep() << " ["
          << static_cast<int>(m_sim_params.total_runtime() / m_mblock.timestep()) << " steps]";

    PLOGI << "[domain]";
    PLOGI << "   dimension: " << static_cast<short>(D) << "D";
    PLOGI << "   metric: " << (m_mblock.metric.label);

#ifdef GRPIC_SIMTYPE
    PLOGI << "   Spin parameter: " << (m_sim_params.metric_parameters()[3]);
#endif

    std::string bc {"   boundary conditions: { "};
    for (auto& b : m_sim_params.boundaries()) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";
    PLOGI << bc;

    std::string res {"   resolution: { "};
    for (auto& r : m_sim_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";
    PLOGI << res;

    std::string ext {"   extent: "};
    for (auto i {0}; i < (int)(m_sim_params.extent().size()); i += 2) {
      ext += "{" + std::to_string(m_sim_params.extent()[i]) + ", "
             + std::to_string(m_sim_params.extent()[i + 1]) + "} ";
    }
    PLOGI << ext;

    std::string cell {"   cell size: "};
    cell += std::to_string(m_mblock.min_cell_size());
    PLOGI << cell;

    PLOGI << "[fiducial parameters]";
    PLOGI << "   ppc0: " << m_sim_params.ppc0();
    PLOGI << "   rho0: " << m_sim_params.larmor0() << " ["
          << m_sim_params.larmor0() / m_mblock.min_cell_size() << " cells]";
    PLOGI << "   c_omp0: " << m_sim_params.skindepth0() << " ["
          << m_sim_params.skindepth0() / m_mblock.min_cell_size() << " cells]";
    PLOGI << "   sigma0: " << m_sim_params.sigma0();

    if (m_mblock.particles.size() > 0) {
      PLOGI << "[particles]";
      int i {0};
      for (auto& prtls : m_mblock.particles) {
        PLOGI << "   [species #" << i + 1 << "]";
        PLOGI << "      label: " << prtls.label();
        PLOGI << "      mass: " << prtls.mass();
        PLOGI << "      charge: " << prtls.charge();
        PLOGI << "      pusher: " << stringifyParticlePusher(prtls.pusher());
        PLOGI << "      maxnpart: " << prtls.maxnpart() << " (" << prtls.npart() << ")";
        ++i;
      }
    } else {
      PLOGI << "[no particles]";
    }
    WaitAndSynchronize();
    PLOGD << "Simulation details printed.";
  }

  template <Dimension D, SimulationType S>
  auto Simulation<D, S>::rangeActiveCells() -> range_t<D> {
    return m_mblock.rangeActiveCells();
  }

  template <Dimension D, SimulationType S>
  auto Simulation<D, S>::rangeAllCells() -> range_t<D> {
    return m_mblock.rangeAllCells();
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::writeOutput(const unsigned long& tstep) {
    WaitAndSynchronize();
    if (m_sim_params.output_format() == "disabled") {
      return;
    } else if (m_sim_params.output_format() == "csv") {
      if ((tstep % m_sim_params.output_interval() == 0)
          && (m_sim_params.output_interval() > 0)) {
        csv::writeField(
          "ex1-" + zeropad(std::to_string(tstep), 5) + ".csv", m_mblock, em::ex1);
        csv::writeField(
          "jx1-" + zeropad(std::to_string(tstep), 5) + ".csv", m_mblock, cur::jx1);
      }
    } else {
      NTTHostError("unrecognized output format");
    }
    PLOGD << "Output written.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::finalize() {
    WaitAndSynchronize();
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::synchronizeHostDevice() {
    WaitAndSynchronize();
    m_mblock.synchronizeHostDevice();
    for (auto& species : m_mblock.particles) {
      species.synchronizeHostDevice();
    }
    PLOGD << "... host-device synchronized";
  }

} // namespace ntt

#ifdef PIC_SIMTYPE
template class ntt::Simulation<ntt::Dim1, ntt::SimulationType::PIC>;
template class ntt::Simulation<ntt::Dim2, ntt::SimulationType::PIC>;
template class ntt::Simulation<ntt::Dim3, ntt::SimulationType::PIC>;
#elif defined(GRPIC_SIMTYPE)
template class ntt::Simulation<ntt::Dim2, ntt::SimulationType::GRPIC>;
template class ntt::Simulation<ntt::Dim3, ntt::SimulationType::GRPIC>;
#endif

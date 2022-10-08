#include "wrapper.h"
#include "simulation.h"
#include "metric.h"

#include "utils.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <stdexcept>
#include <string>

namespace ntt {
  auto stringifySimulationType(const SimulationType& sim) -> std::string {
    switch (sim) {
    case TypePIC:
      return "PIC";
    case SimulationType::GRPIC:
      return "GRPIC";
    default:
      return "N/A";
    }
  }
  auto stringifyBoundaryCondition(const BoundaryCondition& bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::OPEN:
      return "Open";
    case BoundaryCondition::USER:
      return "User";
    case BoundaryCondition::COMM:
      return "Communicate";
    default:
      return "N/A";
    }
  }
  auto stringifyParticlePusher(const ParticlePusher& pusher) -> std::string {
    switch (pusher) {
    case ParticlePusher::BORIS:
      return "Boris";
    case ParticlePusher::VAY:
      return "Vay";
    case ParticlePusher::PHOTON:
      return "Photon";
    default:
      return "N/A";
    }
  }

  template <Dimension D, SimulationType S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_params {inputdata, D},
      problem_generator {m_params},
      meshblock {m_params.resolution(),
                 m_params.extent(),
                 m_params.metricParameters(),
                 m_params.species()},
      random_pool {constant::RandomSeed} {
    meshblock.random_pool_ptr = &random_pool;
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::Initialize() {
    meshblock.boundaries = m_params.boundaries();

    // find timestep and effective cell size
    meshblock.setMinCellSize(meshblock.metric.findSmallestCell());
    meshblock.setTimestep(m_params.cfl() * meshblock.minCellSize());

    WaitAndSynchronize();
    PLOGD << "Simulation initialized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::InitializeSetup() {
    problem_generator.UserInitFields(m_params, meshblock);
    problem_generator.UserInitParticles(m_params, meshblock);

    WaitAndSynchronize();
    PLOGD << "Setup initialized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::Verify() {
    // m_params.verify();
    // mblock.verify(m_params);
    WaitAndSynchronize();
    PLOGD << "Prerun check passed.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::PrintDetails() {
    PLOGI << "[Simulation details]";
    PLOGI << "   title: " << m_params.title();
    PLOGI << "   type: " << stringifySimulationType(S);
    PLOGI << "   total runtime: " << m_params.totalRuntime();
    PLOGI << "   dt: " << meshblock.timestep() << " ["
          << static_cast<int>(m_params.totalRuntime() / meshblock.timestep()) << " steps]";

    PLOGI << "[domain]";
    PLOGI << "   dimension: " << static_cast<short>(D) << "D";
    PLOGI << "   metric: " << (meshblock.metric.label);

    if constexpr (S == SimulationType::GRPIC) {
      PLOGI << "   Spin parameter: " << (m_params.metricParameters()[3]);
    }

    std::string bc {"   boundary conditions: { "};
    for (auto& b : m_params.boundaries()) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";
    PLOGI << bc;

    std::string res {"   resolution: { "};
    for (auto& r : m_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";
    PLOGI << res;

    std::string ext {"   extent: "};
    for (auto i {0}; i < (int)(m_params.extent().size()); i += 2) {
      ext += "{" + std::to_string(m_params.extent()[i]) + ", "
             + std::to_string(m_params.extent()[i + 1]) + "} ";
    }
    PLOGI << ext;

    std::string cell {"   cell size: "};
    cell += std::to_string(meshblock.minCellSize());
    PLOGI << cell;

    PLOGI << "[fiducial parameters]";
    PLOGI << "   ppc0: " << m_params.ppc0();
    PLOGI << "   rho0: " << m_params.larmor0() << " ["
          << m_params.larmor0() / meshblock.minCellSize() << " cells]";
    PLOGI << "   c_omp0: " << m_params.skindepth0() << " ["
          << m_params.skindepth0() / meshblock.minCellSize() << " cells]";
    PLOGI << "   sigma0: " << m_params.sigma0();

    if (meshblock.particles.size() > 0) {
      PLOGI << "[particles]";
      int i {0};
      for (auto& prtls : meshblock.particles) {
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
    return meshblock.rangeActiveCells();
  }

  template <Dimension D, SimulationType S>
  auto Simulation<D, S>::rangeAllCells() -> range_t<D> {
    return meshblock.rangeAllCells();
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::WriteOutput(const unsigned long&) {
    WaitAndSynchronize();
    // auto output_format   = m_params.outputFormat();
    // auto output_interval = m_params.outputInterval();
    // if (output_format == "disabled") {
    //   return;
    // } else if (output_format == "csv") {
    //   if ((tstep % output_interval == 0) && (output_interval > 0)) {
    //     // csv::writeField(
    //     //   "ex1-" + zeropad(std::to_string(tstep), 5) + ".csv", meshblock, em::ex1);
    //     // csv::writeField(
    //     //   "jx1-" + zeropad(std::to_string(tstep), 5) + ".csv", meshblock, cur::jx1);
    //   }
    // } else {
    //   NTTHostError("unrecognized output format");
    // }
    // PLOGD << "Output written.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::Finalize() {
    WaitAndSynchronize();
    PLOGD << "Simulation finalized.";
  }

  template <Dimension D, SimulationType S>
  void Simulation<D, S>::SynchronizeHostDevice() {
    WaitAndSynchronize();
    meshblock.SynchronizeHostDevice();
    for (auto& species : meshblock.particles) {
      species.SynchronizeHostDevice();
    }
    PLOGD << "... host-device synchronized";
  }

} // namespace ntt

#ifdef PIC_SIMTYPE
template class ntt::Simulation<ntt::Dim1, ntt::TypePIC>;
template class ntt::Simulation<ntt::Dim2, ntt::TypePIC>;
template class ntt::Simulation<ntt::Dim3, ntt::TypePIC>;
#elif defined(GRPIC_SIMTYPE)
template class ntt::Simulation<ntt::Dim2, ntt::SimulationType::GRPIC>;
template class ntt::Simulation<ntt::Dim3, ntt::SimulationType::GRPIC>;
#endif

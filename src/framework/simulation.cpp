#include "simulation.h"

#include "wrapper.h"

#include "fields.h"
#include "metric.h"
#include "utils.h"

#include <plog/Log.h>
#include <toml/toml.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace ntt {
  auto stringifySimulationEngine(const SimulationEngine& sim) -> std::string {
    switch (sim) {
    case SANDBOXEngine:
      return "Sandbox";
    case PICEngine:
      return "PIC";
    case GRPICEngine:
      return "GRPIC";
    default:
      return "N/A";
    }
  }
  auto stringifyBoundaryCondition(const BoundaryCondition& bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::ABSORB:
      return "Absorbing";
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
    case ParticlePusher::NONE:
      return "None";
    default:
      return "N/A";
    }
  }

  template <Dimension D, SimulationEngine S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_params { inputdata, D },
      meshblock { m_params.resolution(),
                  m_params.extent(),
                  m_params.metricParameters(),
                  m_params.species() },
      writer {},
      random_pool { constant::RandomSeed } {
    meshblock.random_pool_ptr = &random_pool;
    meshblock.boundaries      = m_params.boundaries();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Initialize() {
    NTTLog();
    // find timestep and effective cell size
    meshblock.setMinCellSize(meshblock.metric.dx_min);
    meshblock.setTimestep(m_params.cfl() * meshblock.minCellSize());

    // initialize writer
    writer.Initialize(m_params, meshblock);

    WaitAndSynchronize();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Verify() {
    NTTLog();
    meshblock.Verify();
    WaitAndSynchronize();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::PrintDetails() {
    std::string bc { "   boundary conditions: { " };
    for (auto& b : m_params.boundaries()) {
      bc += stringifyBoundaryCondition(b) + " x ";
    }
    bc.erase(bc.size() - 3);
    bc += " }";

    std::string res { "   resolution: { " };
    for (auto& r : m_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";

    std::string ext { "   extent: " };
    for (auto i { 0 }; i < (int)(m_params.extent().size()); i += 2) {
      ext += "{" + std::to_string(m_params.extent()[i]) + ", "
             + std::to_string(m_params.extent()[i + 1]) + "} ";
    }

    std::string cell { "   cell size: " };
    cell += std::to_string(meshblock.minCellSize());

    PLOGN_(InfoFile)
      << "[Simulation details]\n"
      << "   title: " << m_params.title() << "\n"
      << "   engine: " << stringifySimulationEngine(S) << "\n"
      << "   total runtime: " << m_params.totalRuntime() << "\n"
      << "   dt: " << meshblock.timestep() << " ["
      << static_cast<int>(m_params.totalRuntime() / meshblock.timestep()) << " steps]\n"
      << "[domain]\n"
      << "   dimension: " << static_cast<short>(D) << "D\n"
      << "   metric: " << (meshblock.metric.label) << "\n"
      << bc << "\n"
      << res << "\n"
      << ext << "\n"
      << cell << "\n"
      << "[fiducial parameters]\n"
      << "   ppc0: " << m_params.ppc0() << "\n"
      << "   rho0: " << m_params.larmor0() << " ["
      << m_params.larmor0() / meshblock.minCellSize() << " cells]\n"
      << "   c_omp0: " << m_params.skindepth0() << " ["
      << m_params.skindepth0() / meshblock.minCellSize() << " cells]\n"
      << "   omp0 * dt: " << m_params.skindepth0() * meshblock.timestep() << "\n"
      << "   sigma0: " << m_params.sigma0();

    if (meshblock.particles.size() > 0) {
      PLOGN_(InfoFile) << "[particles]";
      int i { 0 };
      for (auto& prtls : meshblock.particles) {
        PLOGN_(InfoFile)
          << "   [species #" << i + 1 << "]\n"
          << "      label: " << prtls.label() << "\n"
          << "      mass: " << prtls.mass() << "\n"
          << "      charge: " << prtls.charge() << "\n"
          << "      pusher: " << stringifyParticlePusher(prtls.pusher()) << "\n"
          << "      maxnpart: " << prtls.maxnpart() << " (" << prtls.npart() << ")";
        ++i;
      }
    } else {
      PLOGN_(InfoFile) << "[no particles]";
    }
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::rangeActiveCells() -> range_t<D> {
    return meshblock.rangeActiveCells();
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::rangeAllCells() -> range_t<D> {
    return meshblock.rangeAllCells();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Finalize() {
    WaitAndSynchronize();
    NTTLog();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::PrintDiagnostics(std::ostream&              os,
                                          const std::vector<double>& fractions) {
    for (std::size_t i { 0 }; i < meshblock.particles.size(); ++i) {
      auto& species { meshblock.particles[i] };
      os << "species #" << i << ": " << species.npart() << " ("
         << (double)(species.npart()) * 100 / (double)(species.maxnpart()) << "%";
      if (fractions.size() == meshblock.particles.size()) {
        auto fraction = fractions[i];
        os << ", " << fraction * 100 << "% dead)\n";
      } else {
        os << ")\n";
      }
    }
  }

}    // namespace ntt
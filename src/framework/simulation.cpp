#include "simulation.h"

#include "wrapper.h"

#include "io/output.h"
#include "utils/progressbar.h"
#include "utils/timer.h"
#include "utils/utils.h"

#include <plog/Log.h>
#include <toml.hpp>

#include <string>

namespace ntt {
  template <Dimension D, SimulationEngine S>
  Simulation<D, S>::Simulation(const toml::value& inputdata)
    : m_params { inputdata, D },
      m_metadomain { m_params.resolution(),
                     m_params.extent(),
                     m_params.domaindecomposition(),
                     m_params.metricParameters(),
                     m_params.boundaries() },
      meshblock { m_metadomain.localDomain()->ncells(),
                  m_metadomain.localDomain()->extent(),
                  m_params.metricParameters(),
                  m_params.species() },
      writer {},
      random_pool { constant::RandomSeed } {
    meshblock.random_pool_ptr = &random_pool;
    meshblock.boundaries      = m_metadomain.localDomain()->boundaries();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::Initialize() {
    NTTLog();
    // find timestep and effective cell size
    meshblock.setMinCellSize(meshblock.metric.dx_min);
    if (m_params.dt() <= ZERO) {
      meshblock.setTimestep(m_params.cfl() * meshblock.minCellSize());
    } else {
      meshblock.setTimestep(m_params.dt());
    }

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
    std::string bc { "" };
    for (auto& boundaries_xi : m_params.boundaries()) {
      bc += "{";
      for (auto& boundaries : boundaries_xi) {
        bc += stringizeBoundaryCondition(boundaries) + ", ";
      }
      bc.erase(bc.size() - 2);
      bc += "} ";
    }
    bc.erase(bc.size() - 1);

    std::string res { "{ " };
    for (auto& r : m_params.resolution()) {
      res += std::to_string(r) + " x ";
    }
    res.erase(res.size() - 3);
    res += " }";

    std::string ext { "" };
    for (auto i { 0 }; i < (int)(m_params.extent().size()); i += 2) {
      ext += "{" + std::to_string(m_params.extent()[i]) + ", "
             + std::to_string(m_params.extent()[i + 1]) + "} ";
    }

    std::string cell { "" };
    cell += std::to_string(meshblock.minCellSize());

    PLOGN_(InfoFile)
      << "============================================================\n"
      << "Entity v" << ENTITY_VERSION << "\n"
      << "============================================================\n\n"
      << "[Simulation parameters]\n"
      << std::setw(42) << std::setfill('.') << std::left << "  title:" << m_params.title()
      << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  engine:" << stringizeSimulationEngine(S) << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  timestep:" << meshblock.timestep() << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  CFL:" << meshblock.timestep() / meshblock.minCellSize() << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  total runtime:" << m_params.totalRuntime() << " ["
      << static_cast<int>(m_params.totalRuntime() / meshblock.timestep()) << " steps]\n"
      << "[domain]\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  dimension:" << static_cast<short>(D) << "D\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  metric:" << (meshblock.metric.label) << "\n"
      << std::setw(42) << std::setfill('.') << std::left << "  boundary conditions:" << bc
      << "\n"
      << std::setw(42) << std::setfill('.') << std::left << "  resolution:" << res << "\n"
      << std::setw(42) << std::setfill('.') << std::left << "  extent:" << ext << "\n"
      << std::setw(42) << std::setfill('.') << std::left << "  cell size:" << cell << "\n"
      << "[fiducial parameters]\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  particles per cell [ppc0]:" << m_params.ppc0() << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  Larmor radius [rho0]:" << m_params.larmor0() << " ["
      << m_params.larmor0() / meshblock.minCellSize() << " cells]\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  Larmor frequency [omegaB0 * dt]:" << meshblock.timestep() / m_params.larmor0()
      << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  skin depth [d0]:" << m_params.skindepth0() << " ["
      << m_params.skindepth0() / meshblock.minCellSize() << " cells]\n"
      << std::setw(42) << std::setfill('.') << std::left << "  plasma frequency [omp0 * dt]:"
      << (ONE / m_params.skindepth0()) * meshblock.timestep() << "\n"
      << std::setw(42) << std::setfill('.') << std::left
      << "  magnetization [sigma0]:" << m_params.sigma0();

    if (meshblock.particles.size() > 0) {
      PLOGN_(InfoFile) << "[particles]";
      int i { 0 };
      for (auto& prtls : meshblock.particles) {
        PLOGN_(InfoFile)
          << "  [species #" << i + 1 << "]\n"
          << std::setw(42) << std::setfill('.') << std::left << "    label: " << prtls.label()
          << "\n"
          << std::setw(42) << std::setfill('.') << std::left << "    mass: " << prtls.mass()
          << "\n"
          << std::setw(42) << std::setfill('.') << std::left
          << "    charge: " << prtls.charge() << "\n"
          << std::setw(42) << std::setfill('.') << std::left
          << "    pusher: " << stringizeParticlePusher(prtls.pusher()) << "\n"
          << std::setw(42) << std::setfill('.') << std::left
          << "    maxnpart: " << prtls.maxnpart() << " (active: " << prtls.npart() << ")";
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
    writer.Finalize();
    WaitAndSynchronize();
    NTTLog();
  }

  template <Dimension D, SimulationEngine S>
  void Simulation<D, S>::PrintDiagnostics(const std::size_t&         step,
                                          const real_t&              time,
                                          const std::vector<double>& fractions,
                                          const timer::Timers&       timers,
                                          std::vector<long double>&  tstep_durations,
                                          const DiagFlags            diag_flags,
                                          std::ostream&              os) {
    tstep_durations.push_back(timers.get("Total"));
    if (step % m_params.diagInterval() == 0) {
      const auto title { "time = " + std::to_string(time)
                         + " : step = " + std::to_string(step) };
      if (diag_flags & DiagFlags_Timers) {
        timers.printAll(title);
      } else {
        os << title << std::endl;
      }
      if (diag_flags & DiagFlags_Species) {
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
      if (diag_flags & DiagFlags_Progress) {
        os << std::setw(46) << std::setfill('-') << "" << std::endl;
        ProgressBar(tstep_durations, time, m_params.totalRuntime(), os);
      }
      os << std::setw(46) << std::setfill('=') << "" << std::endl;
    }
  }

}    // namespace ntt
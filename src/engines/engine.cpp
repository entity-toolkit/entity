#include "engines/engine.h"

#include "enums.h"
#include "global.h"

#include "utils/formatting.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include <string>

namespace ntt {

  template <SimEngine::type S, class M>
  void Engine<S, M>::print_report() const {

    // ╔══════════════════════════════════════════════════════════════╗
    // ║                         Entity Simulation                    ║
    // ╚══════════════════════════════════════════════════════════════╝

    // Simulation Parameters
    // ├─ Environment
    // │  └─ Earth-like Planet
    // ├─ Entity Count
    // │  └─ 150
    // ├─ Energy Levels
    // │  └─ Initial: High
    // ├─ Evolution Dynamics
    // │  ├─ Mutation Rate
    // │  │  └─ 0.05
    // │  └─ Selection Pressure
    // │     └─ Moderate
    // └─ Simulation Settings
    //    ├─ Duration
    //    │  └─ 10000 Generations
    //    └─ Climate
    //       └─ Variability: Dynamic

    // Initializing simulation with the above parameters...
    // Stand by, booting up entities...

    // const auto version = "Entity v" + std::string(ENTITY_VERSION);
    // const auto hash    = std::string(ENTITY_GIT_HASH);
    // // clang-format off
    // const auto report = fmt::format(
    //   "╔%s╗\n"
    //   "║ %s%s║\n"
    //   "║ hash: %s%s║\n"
    //   "╚%s╝\n"
    //   "\n"
    //   "Configuration\n"
    //   "├─ Name:\n"
    //   "│   └─ %s\n"
    //   "├─ Engine:\n"
    //   "│   └─ %s\n"
    //   "├─ Metric:\n"
    //   "│   └─ %dD %s\n"
    //   "├─ Timestep:\n"
    //   "│   └─ %.3e\n"
    //   "Metadomain\n"
    //   "├─ Resolution:\n"
    //   "│   └─ %s\n"
    //   "├─ Extent:\n"
    //   "│   └─ %s\n"
    //   "├─ Boundary conditions:\n"
    //   "│   ├─ Fields:\n"
    //   "│   │   └─ %s\n"
    //   "│   └─ Particles:\n"
    //   "│       └─ %s\n"
    //   // "├─ Domain decomposition:\n"
    //   // "│   └─ %s [%d total]\n"
    //   , 
    //   fmt::repeat("═", 58).c_str(), 
    //   version.c_str(), fmt::repeat(" ", 57 - version.size()).c_str(),
    //   hash.c_str(), fmt::repeat(" ", 51 - hash.size()).c_str(),
    //   fmt::repeat("═", 58).c_str(),
    //   m_params.get<std::string>("simulation.name").c_str(),
    //   SimEngine(S).to_string(),
    //   M::Dim, Metric(M::MetricType).to_string(),
    //   m_params.get<real_t>("algorithms.timestep.dt"),
    //   m_params.stringize<std::size_t>("grid.resolution").c_str(),
    //   m_params.stringize<boundaries_t<real_t>>("grid.extent").c_str(),
    //   m_params.stringize<boundaries_t<FldsBC>>("grid.boundaries.fields").c_str(),
    //   m_params.stringize<boundaries_t<PrtlBC>>("grid.boundaries.particles").c_str()
    // );
    // // clang-format on

    // info::Print(report);

    //           PLOGN_(InfoFile)
    //         << "============================================================\n"
    //         << "Entity v" << ENTITY_VERSION << "\n"
    //         << "============================================================\n\n"
    //         << "[Simulation parameters]\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  title:" << m_params.title() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  engine:" << stringizeSimulationEngine(S) << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  timestep:" << meshblock.timestep() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  CFL:" << meshblock.timestep() / meshblock.minCellSize() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  total runtime:" << m_params.totalRuntime() << " ["
    //         << static_cast<int>(m_params.totalRuntime() / meshblock.timestep())
    //         << " steps]\n"
    //         << "[domain]\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  dimension:" << static_cast<short>(D) << "D\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  metric:" << (meshblock.metric.label) << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  boundary conditions:" << bc << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  resolution:" << res << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left << "  extent:" << ext
    //         << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  cell size:" << cell << "\n"
    //         << "[fiducial parameters]\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  particles per cell [ppc0]:" << m_params.ppc0() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  Larmor radius [rho0]:" << m_params.larmor0() << " ["
    //         << m_params.larmor0() / meshblock.minCellSize() << " cells]\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  Larmor frequency [omegaB0 * dt]:"
    //         << meshblock.timestep() / m_params.larmor0() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  skin depth [d0]:" << m_params.skindepth0() << " ["
    //         << m_params.skindepth0() / meshblock.minCellSize() << " cells]\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  plasma frequency [omp0 * dt]:"
    //         << (ONE / m_params.skindepth0()) * meshblock.timestep() << "\n"
    //         << std::setw(42) << std::setfill('.') << std::left
    //         << "  magnetization [sigma0]:" << m_params.sigma0();

    //       if (meshblock.particles.size() > 0) {
    //         PLOGN_(InfoFile) << "[particles]";
    //         int i { 0 };
    //         for (auto& species : meshblock.particles) {
    //           PLOGN_(InfoFile) << "  [species #" << i + 1 << "]\n"
    //                            << std::setw(42) << std::setfill('.') << std::left
    //                            << "    label: " << species.label() << "\n"
    //                            << std::setw(42) << std::setfill('.') << std::left
    //                            << "    mass: " << species.mass() << "\n"
    //                            << std::setw(42) << std::setfill('.') << std::left
    //                            << "    charge: " << species.charge() << "\n"
    //                            << std::setw(42) << std::setfill('.') << std::left
    //                            << "    pusher: "
    //                            << stringizeParticlePusher(species.pusher());
    //           if (species.cooling() != Cooling::NONE) {
    //             PLOGN_(InfoFile)
    //               << std::setw(42) << std::setfill('.') << std::left
    //               << "    cooling: " << stringizeCooling(species.cooling());
    //           }
    //           PLOGN_(InfoFile) << std::setw(42) << std::setfill('.') << std::left
    //                            << "    maxnpart: " << species.maxnpart()
    //                            << " (active: " << species.npart() << ")";
    //           ++i;
    //         }
    //       } else {
    //         PLOGN_(InfoFile) << "[no particles]";
    //       }
    //     }
    // #ifdef MPI_ENABLED
    //     {
    //       int rank, size;
    //       MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //       MPI_Comm_size(MPI_COMM_WORLD, &size);
    //       if (rank == 0) {
    //         PLOGN_(InfoFile) << "[metadomain]";
    //       }
    //       for (auto r { 0 }; r < size; ++r) {
    //         if (r == rank) {
    //           PLOGN_(InfoFile) << "  [domain #" << rank << "]";
    //           auto        extent = metadomain()->localDomain()->extent();
    //           auto        ncells = metadomain()->localDomain()->ncells();
    //           std::string res { "{ " };
    //           for (auto& r : ncells) {
    //             res += std::to_string(r) + " x ";
    //           }
    //           res.erase(res.size() - 3);
    //           res += " }";

    //           std::string ext { "" };
    //           for (auto i { 0 }; i < (int)(extent.size()); i += 2) {
    //             ext += "{" + std::to_string(extent[i]) + ", " +
    //                    std::to_string(extent[i + 1]) + "} ";
    //           }
    //           PLOGN_(InfoFile) << std::setw(42) << std::setfill('.') << std::left
    //                            << "    resolution:" << res << "\n"
    //                            << std::setw(42) << std::setfill('.') << std::left
    //                            << "    extent:" << ext;
    //         }
    //         MPI_Barrier(MPI_COMM_WORLD);
    //       }
    //     }
    // #endif
  }

  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template class Engine<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;

} // namespace ntt
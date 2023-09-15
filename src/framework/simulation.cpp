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
  Simulation<D, S>::Simulation(const toml::value& inputdata) :
    m_params { inputdata, D },
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
#ifdef MPI_ENABLED
    random_pool { constant::RandomSeed + m_metadomain.localDomain()->mpiRank() }
#else
    random_pool { constant::RandomSeed }
#endif
  {
    meshblock.random_pool_ptr = &random_pool;
    meshblock.boundaries      = m_metadomain.localDomain()->boundaries();

    // find timestep, effective cell size, fiducial cell volume
    // synchronize with other blocks
    meshblock.metric.set_dxMin(m_metadomain.smallestCellSize());
    m_params.setV0(m_metadomain.fiducialCellVolume());
    if (m_params.dt() <= ZERO) {
      meshblock.setTimestep(m_params.cfl() * meshblock.minCellSize());
    } else {
      meshblock.setTimestep(m_params.dt());
    }
    NTTHostErrorIf(meshblock.timestep() <= ZERO, "Timestep is zero or negative. Check CFL condition and/or min cell size.");

    // initialize writer
    writer.Initialize(m_params, m_metadomain, meshblock);

    WaitAndSynchronize();
  }

  template <Dimension D, SimulationEngine S>
  Simulation<D, S>::~Simulation() {
    writer.Finalize();
    WaitAndSynchronize();
    NTTLog();
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::Verify() -> void {
    NTTLog();
    meshblock.Verify();
    meshblock.CheckNaNs("Initial check",
                        CheckNaN_Fields | CheckNaN_Particles | CheckNaN_Currents);
    meshblock.CheckOutOfBounds("Initial check");
    WaitAndSynchronize();
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::ParticlesBoundaryConditions() -> void {
    for (auto& species : meshblock.particles) {
      species.BoundaryConditions(meshblock);
    }
    NTTLog();
  }

  template <Dimension D, SimulationEngine S>
  auto Simulation<D, S>::PrintDetails() -> void {
    auto skip_details = false;
#if defined(MPI_ENABLED)
    skip_details = (metadomain()->localDomain()->mpiRank() != 0);
#endif
    if (!skip_details) {
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
        ext += "{" + std::to_string(m_params.extent()[i]) + ", " +
               std::to_string(m_params.extent()[i + 1]) + "} ";
      }

      std::string cell { "" };
      cell += std::to_string(meshblock.minCellSize());

      PLOGN_(InfoFile)
        << "============================================================\n"
        << "Entity v" << ENTITY_VERSION << "\n"
        << "============================================================\n\n"
        << "[Simulation parameters]\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  title:" << m_params.title() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  engine:" << stringizeSimulationEngine(S) << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  timestep:" << meshblock.timestep() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  CFL:" << meshblock.timestep() / meshblock.minCellSize() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  total runtime:" << m_params.totalRuntime() << " ["
        << static_cast<int>(m_params.totalRuntime() / meshblock.timestep())
        << " steps]\n"
        << "[domain]\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  dimension:" << static_cast<short>(D) << "D\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  metric:" << (meshblock.metric.label) << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  boundary conditions:" << bc << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  resolution:" << res << "\n"
        << std::setw(42) << std::setfill('.') << std::left << "  extent:" << ext
        << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  cell size:" << cell << "\n"
        << "[fiducial parameters]\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  particles per cell [ppc0]:" << m_params.ppc0() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  Larmor radius [rho0]:" << m_params.larmor0() << " ["
        << m_params.larmor0() / meshblock.minCellSize() << " cells]\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  Larmor frequency [omegaB0 * dt]:"
        << meshblock.timestep() / m_params.larmor0() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  skin depth [d0]:" << m_params.skindepth0() << " ["
        << m_params.skindepth0() / meshblock.minCellSize() << " cells]\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  plasma frequency [omp0 * dt]:"
        << (ONE / m_params.skindepth0()) * meshblock.timestep() << "\n"
        << std::setw(42) << std::setfill('.') << std::left
        << "  magnetization [sigma0]:" << m_params.sigma0();

      if (meshblock.particles.size() > 0) {
        PLOGN_(InfoFile) << "[particles]";
        int i { 0 };
        for (auto& species : meshblock.particles) {
          PLOGN_(InfoFile) << "  [species #" << i + 1 << "]\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    label: " << species.label() << "\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    mass: " << species.mass() << "\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    charge: " << species.charge() << "\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    pusher: "
                           << stringizeParticlePusher(species.pusher()) << "\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    maxnpart: " << species.maxnpart()
                           << " (active: " << species.npart() << ")";
          ++i;
        }
      } else {
        PLOGN_(InfoFile) << "[no particles]";
      }
    }
#ifdef MPI_ENABLED
    {
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      if (rank == 0) {
        PLOGN_(InfoFile) << "[metadomain]";
      }
      for (auto r { 0 }; r < size; ++r) {
        if (r == rank) {
          PLOGN_(InfoFile) << "  [domain #" << rank << "]";
          auto        extent = metadomain()->localDomain()->extent();
          auto        ncells = metadomain()->localDomain()->ncells();
          std::string res { "{ " };
          for (auto& r : ncells) {
            res += std::to_string(r) + " x ";
          }
          res.erase(res.size() - 3);
          res += " }";

          std::string ext { "" };
          for (auto i { 0 }; i < (int)(extent.size()); i += 2) {
            ext += "{" + std::to_string(extent[i]) + ", " +
                   std::to_string(extent[i + 1]) + "} ";
          }
          PLOGN_(InfoFile) << std::setw(42) << std::setfill('.') << std::left
                           << "    resolution:" << res << "\n"
                           << std::setw(42) << std::setfill('.') << std::left
                           << "    extent:" << ext;
        }
        MPI_Barrier(MPI_COMM_WORLD);
      }
    }
#endif
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
  auto Simulation<D, S>::PrintDiagnostics(const std::size_t&   step,
                                          const real_t&        time,
                                          const timer::Timers& timers,
                                          std::vector<long double>& tstep_durations,
                                          const DiagFlags diag_flags,
                                          std::ostream&   os) -> void {
    if (tstep_durations.size() > m_params.diagMaxnForPbar()) {
      tstep_durations.erase(tstep_durations.begin());
    }
    tstep_durations.push_back(timers.get("Total"));
    if (step % m_params.diagInterval() == 0) {
      auto& mblock = this->meshblock;
      const auto title { fmt::format("Time = %f : step = %d : Δt = %f", time, step, mblock.timestep()) };
      timers.printAll(title,
                      (diag_flags & DiagFlags_Timers)
                        ? timer::TimerFlags_Default
                        : timer::TimerFlags_PrintTitle,
                      os);
      if (diag_flags & DiagFlags_Species) {
        auto header = fmt::format("%s %27s", "[SPECIES]", "[TOT]");
#if defined(MPI_ENABLED)
        header += fmt::format("%17s %s", "[MIN (%) :", "MAX (%)]");
#endif
        PrintOnce(
          [](std::ostream& os, std::string header) {
            os << header << std::endl;
          },
          os,
          header);
        for (const auto& species : meshblock.particles) {
          species.PrintParticleCounts(os);
        }
      }
      if (diag_flags & DiagFlags_Progress) {
        PrintOnce(
          [](std::ostream& os) {
            os << std::setw(65) << std::setfill('-') << "" << std::endl;
          },
          os);
        ProgressBar(tstep_durations, time, m_params.totalRuntime(), os);
      }
      PrintOnce(
        [](std::ostream& os) {
          os << std::setw(65) << std::setfill('=') << "" << std::endl;
        },
        os);
    }
  }

} // namespace ntt
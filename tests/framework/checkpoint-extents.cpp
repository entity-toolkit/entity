/**
 * Tests that per-subdomain extents and global mesh extents are correctly saved
 * to and restored from a checkpoint, as needed by the moving window feature.
 */

#include "enums.h"
#include "global.h"

#include "utils/comparators.h"
#include "utils/error.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

#include "metrics/minkowski.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>
#include <toml11/toml.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

const std::string CHECKPOINT_DIR = "test_checkpoint_extents";

void cleanup() {
  std::filesystem::remove_all(CHECKPOINT_DIR);
}

auto extents_match(const boundaries_t<real_t>& a,
                   const boundaries_t<real_t>& b) -> bool {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t d { 0 }; d < a.size(); ++d) {
    if (not cmp::AlmostEqual(a[d].first, b[d].first) or
        not cmp::AlmostEqual(a[d].second, b[d].second)) {
      return false;
    }
  }
  return true;
}

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);

  try {
    using M = Minkowski<Dim::_1D>;

    const std::vector<ncells_t> res { 64 };
    const boundaries_t<real_t>  init_extent { { static_cast<real_t>(0.0),
                                                static_cast<real_t>(10.0) } };
    const boundaries_t<FldsBC>  fldsbc {
      { FldsBC::PERIODIC, FldsBC::PERIODIC }
    };
    const boundaries_t<PrtlBC> prtlbc {
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
    };
    const std::vector<int> decomp { -1 };

#if !defined(MPI_ENABLED)
    const unsigned int ndomains { 1 };
    adios2::ADIOS      adios;
#else
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const unsigned int ndomains { static_cast<unsigned int>(mpi_size) };
    adios2::ADIOS      adios { MPI_COMM_WORLD };
#endif

    const auto checkpoint_step = timestep_t { 2 };
    const int  n_shift         = 4;

    // reference extents captured after the window shift
    boundaries_t<real_t>              expected_global_extent;
    std::vector<boundaries_t<real_t>> expected_subdomain_extents(ndomains);

    {
      Metadomain<SimEngine::SRPIC, M> md {
        ndomains, decomp, res, init_extent, fldsbc, prtlbc, {}, {}
      };

      SimulationParams params;
      params.set("checkpoint.write_path", std::string { CHECKPOINT_DIR });
      params.set("checkpoint.interval", timestep_t { 1 });
      params.set("checkpoint.interval_time", simtime_t { -1.0 });
      params.set("checkpoint.keep", 5);
      params.set("checkpoint.walltime", std::string { "" });
      params.setRawData(toml::value { toml::table {} });

      md.InitCheckpointWriter(&adios, params);

      // shift the window and capture the resulting extents
      md.ShiftByCells(n_shift, in::x1);
      expected_global_extent = md.mesh().extent();
      for (unsigned int idx { 0 }; idx < ndomains; ++idx) {
        expected_subdomain_extents[idx] = md.subdomain(idx).mesh.extent();
      }

      // write checkpoint at step 2 (finished_step must be > 1 to be saved)
      const auto wrote = md.WriteCheckpoint(
        params,
        checkpoint_step,
        checkpoint_step,
        simtime_t { 0.0 },
        simtime_t { 0.0 });
      raise::ErrorIf(not wrote, "checkpoint was not written", HERE);
    }

    {
      // construct a fresh metadomain from the original (pre-shift) extent
      Metadomain<SimEngine::SRPIC, M> md2 {
        ndomains, decomp, res, init_extent, fldsbc, prtlbc, {}, {}
      };

      // sanity: verify global extent is the original one before reading
      raise::ErrorIf(
        extents_match(md2.mesh().extent(), expected_global_extent),
        "global extent should differ from shifted extent before checkpoint read",
        HERE);

      SimulationParams params2;
      params2.set("checkpoint.read_path", std::string { CHECKPOINT_DIR });
      params2.set("checkpoint.start_step", checkpoint_step);

      md2.ContinueFromCheckpoint(&adios, params2);

      // global mesh extent must match the shifted one
      raise::ErrorIf(
        not extents_match(md2.mesh().extent(), expected_global_extent),
        "global mesh extent mismatch after checkpoint read",
        HERE);

      // per-subdomain extents must also match
      for (unsigned int idx { 0 }; idx < ndomains; ++idx) {
        raise::ErrorIf(
          not extents_match(md2.subdomain(idx).mesh.extent(),
                            expected_subdomain_extents[idx]),
          "subdomain extent mismatch after checkpoint read",
          HERE);
      }
    }

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    cleanup();
    GlobalFinalize();
    return 1;
  }

  cleanup();
  GlobalFinalize();
  return 0;
}

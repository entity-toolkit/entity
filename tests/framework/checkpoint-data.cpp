/**
 * Tests that field array sizes and particle coordinates are correctly preserved
 * through a checkpoint round-trip.
 *
 * In MPI mode the checkpoint is written from a metadomain whose subdomain
 * sizes have been changed via redecomposeFromCheckpoint (simulating load
 * balancing from [32,32] to [40,24] cells), then read back into a uniformly
 * decomposed metadomain ([32,32]), exercising the reconstruction path in
 * ContinueFromCheckpoint.
 */

#include "enums.h"
#include "global.h"

#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "metrics/minkowski.h"

#include "framework/domain/metadomain.h"
#include "framework/parameters/parameters.h"

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

const std::string CHECKPOINT_DIR = "test_checkpoint_data";

void cleanup() {
  std::filesystem::remove_all(CHECKPOINT_DIR);
}

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);

  try {
    using M = Minkowski<Dim::_1D>;

    const std::vector<ncells_t> res { 64 };
    const boundaries_t<real_t>  init_extent {
       { static_cast<real_t>(0.0), static_cast<real_t>(10.0) }
    };
    const boundaries_t<FldsBC> fldsbc {
      { FldsBC::PERIODIC, FldsBC::PERIODIC }
    };
    const boundaries_t<PrtlBC> prtlbc {
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
    };
    const std::vector<int> decomp { -1 };

    const std::vector<ParticleSpecies> species_params {
      ParticleSpecies { static_cast<spidx_t>(1),
                       "e-", 1.0f,
                       -1.0f,
                       static_cast<npart_t>(10),
                       timestep_t { 0 },
                       timestep_t { 0 },
                       ParticlePusher::BORIS,
                       false, RadiativeDrag::NONE,
                       EmissionType::NONE,
                       static_cast<unsigned short>(0),
                       static_cast<unsigned short>(0) }
    };

#if !defined(MPI_ENABLED)
    const unsigned int          ndomains { 1 };
    adios2::ADIOS               adios;
    // non-MPI: single domain, sizes always match (equal-sizes path)
    const std::vector<ncells_t> saved_ncells_per_dom { 64 };
#else
    int mpi_size, mpi_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    const unsigned int ndomains { static_cast<unsigned int>(mpi_size) };
    adios2::ADIOS      adios { MPI_COMM_WORLD };
    raise::ErrorIf(mpi_size != 2, "this test requires exactly 2 MPI ranks", HERE);
    // asymmetric split: 40+24=64; exercises redecomposeFromCheckpoint
    const std::vector<ncells_t> saved_ncells_per_dom { 40, 24 };
    const real_t                x_split = init_extent[0].second *
                           static_cast<real_t>(saved_ncells_per_dom[0]) /
                           static_cast<real_t>(64);
#endif

    const auto    checkpoint_step = timestep_t { 2 };
    const int     prtl_i1_0 { 3 };
    const real_t  prtl_dx1_0 { static_cast<real_t>(0.25) };
    const int     prtl_i1_1 { 7 };
    const real_t  prtl_dx1_1 { static_cast<real_t>(0.75) };
    const npart_t npart_placed { 2 };

    // ── Write phase ───────────────────────────────────────────────────────────
    {
      Metadomain<SimEngine::SRPIC, M> md { ndomains, decomp, res, init_extent,
                                           fldsbc,   prtlbc, {},  species_params };

#if defined(MPI_ENABLED)
      // Simulate load balancing: [32, 32] → [40, 24]
      md.redecomposeFromCheckpoint(
        { { saved_ncells_per_dom[0] }, { saved_ncells_per_dom[1] } },
        { { { init_extent[0].first, x_split } },
          { { x_split, init_extent[0].second } } });
#endif

      auto*      local = md.subdomain_ptr(md.l_subdomain_indices()[0]);
      const auto n_all = local->mesh.n_all()[0];

      // set em::ex1(i) = real_t(i) over all cells including ghost cells
      {
        auto em_h = Kokkos::create_mirror_view(local->fields.em);
        for (std::size_t i { 0 }; i < n_all; ++i) {
          em_h(i, em::ex1) = static_cast<real_t>(i);
        }
        Kokkos::deep_copy(local->fields.em, em_h);
      }

      // place 2 particles at known positions (safe within the 24-cell domain)
      auto& sp = local->species[0];
      sp.set_npart(npart_placed);
      {
        auto i1_h  = Kokkos::create_mirror_view(sp.i1);
        auto dx1_h = Kokkos::create_mirror_view(sp.dx1);
        i1_h(0)    = prtl_i1_0;
        dx1_h(0)   = static_cast<prtldx_t>(prtl_dx1_0);
        i1_h(1)    = prtl_i1_1;
        dx1_h(1)   = static_cast<prtldx_t>(prtl_dx1_1);
        Kokkos::deep_copy(sp.i1, i1_h);
        Kokkos::deep_copy(sp.dx1, dx1_h);
      }

      SimulationParams params;
      params.set("checkpoint.write_path", std::string { CHECKPOINT_DIR });
      params.set("checkpoint.interval", timestep_t { 1 });
      params.set("checkpoint.interval_time", simtime_t { -1.0 });
      params.set("checkpoint.keep", 5);
      params.set("checkpoint.walltime", std::string { "" });
      params.setRawData(toml::value { toml::table {} });

      md.InitCheckpointWriter(&adios, params);

      const auto wrote = md.WriteCheckpoint(params,
                                            checkpoint_step,
                                            checkpoint_step,
                                            simtime_t { 0.0 },
                                            simtime_t { 0.0 });
      raise::ErrorIf(not wrote, "checkpoint was not written", HERE);
    }

    // ── Read phase ────────────────────────────────────────────────────────────
    {
      // fresh metadomain with the original (uniform) decomposition
      // MPI: [32, 32] — mismatch with saved [40, 24] → reconstruction triggered
      Metadomain<SimEngine::SRPIC, M> md2 { ndomains,    decomp,        res,
                                            init_extent, fldsbc,        prtlbc,
                                            {},          species_params };

      SimulationParams params2;
      params2.set("checkpoint.read_path", std::string { CHECKPOINT_DIR });
      params2.set("checkpoint.start_step", checkpoint_step);

      md2.ContinueFromCheckpoint(&adios, params2);

      auto*      local      = md2.subdomain_ptr(md2.l_subdomain_indices()[0]);
      const auto lidx       = md2.l_subdomain_indices()[0];
      const auto exp_ncells = saved_ncells_per_dom[lidx];
      const auto exp_n_all  = exp_ncells + 2 * N_GHOSTS;

      // 1. field array size reflects saved (possibly reconstructed) ncells
      raise::ErrorIf(local->fields.em.extent(0) != exp_n_all,
                     fmt::format("field em extent: got %lu, expected %lu",
                                 local->fields.em.extent(0),
                                 exp_n_all),
                     HERE);

      // 2. field values survive the round-trip
      {
        auto em_h = Kokkos::create_mirror_view(local->fields.em);
        Kokkos::deep_copy(em_h, local->fields.em);
        for (std::size_t i { 0 }; i < exp_n_all; ++i) {
          raise::ErrorIf(
            not cmp::AlmostEqual(em_h(i, em::ex1), static_cast<real_t>(i)),
            fmt::format("em::ex1 mismatch at i=%lu: got %f, expected %f",
                        i,
                        static_cast<double>(em_h(i, em::ex1)),
                        static_cast<double>(i)),
            HERE);
        }
      }

      // 3. particle count and coordinates survive the round-trip
      auto& sp = local->species[0];
      raise::ErrorIf(sp.npart() != npart_placed,
                     fmt::format("particle count: got %lu, expected %lu",
                                 sp.npart(),
                                 npart_placed),
                     HERE);
      {
        auto i1_h  = Kokkos::create_mirror_view(sp.i1);
        auto dx1_h = Kokkos::create_mirror_view(sp.dx1);
        Kokkos::deep_copy(i1_h, sp.i1);
        Kokkos::deep_copy(dx1_h, sp.dx1);

        raise::ErrorIf(
          i1_h(0) != prtl_i1_0,
          fmt::format("particle 0 i1: got %d, expected %d", i1_h(0), prtl_i1_0),
          HERE);
        raise::ErrorIf(
          not cmp::AlmostEqual(static_cast<real_t>(dx1_h(0)), prtl_dx1_0),
          "particle 0 dx1 mismatch",
          HERE);
        raise::ErrorIf(
          i1_h(1) != prtl_i1_1,
          fmt::format("particle 1 i1: got %d, expected %d", i1_h(1), prtl_i1_1),
          HERE);
        raise::ErrorIf(
          not cmp::AlmostEqual(static_cast<real_t>(dx1_h(1)), prtl_dx1_1),
          "particle 1 dx1 mismatch",
          HERE);
      }
    }

  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    CallOnce([&] {
      cleanup();
    });
    GlobalFinalize();
    return 1;
  }

  CallOnce([&] {
    cleanup();
  });
  GlobalFinalize();
  return 0;
}

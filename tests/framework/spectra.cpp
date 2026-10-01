/**
 * Tests the spatially binned spectra output.
 *
 * Particles with known cell positions and energies are placed on a 16x12
 * Minkowski grid, binned into 4x3 spatial bins x 4 energy bins, written out
 * via `Metadomain::WriteSpectra`, and read back from the `.bp` file.
 *
 * In MPI mode the particles are distributed over the two subdomains, which
 * exercises the reduction of the per-rank histograms in `writeSpectrumSpatial`.
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

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;
using namespace metric;

const std::string SIMNAME = "test_spectra";

namespace {
  constexpr ncells_t    NX1 = 16, NX2 = 12;
  constexpr std::size_t NB1 = 4, NB2 = 3, NBE = 4;
  constexpr real_t      E_MIN = 0.0, E_MAX = 4.0;

  struct TestPrtl {
    ncells_t    gi1, gi2; // global cell index
    real_t      gamma;
    real_t      weight;
    bool        alive;
    std::size_t b1, b2, be; // expected bin
  };

  // dx = 0.5 in both directions, so no particle sits on a bin edge
  const std::vector<TestPrtl> PRTLS {
    {  0,  0,   1.5, 1.0,  true, 0, 0, 0 },
    {  2,  3,   1.5, 2.0,  true, 0, 0, 0 }, // accumulates with the previous one
    {  6,  1,   2.5, 1.0,  true, 1, 0, 1 },
    {  9,  7,   3.5, 1.0,  true, 2, 1, 2 },
    { 15, 11,   4.5, 1.0,  true, 3, 2, 3 },
    {  5,  9,   1.5, 3.0,  true, 1, 2, 0 },
    { 13,  5, 100.0, 1.0,  true, 3, 1, 3 }, // above e_max -> last energy bin
    { 11, 11,   2.5, 5.0, false, 0, 0, 0 }, // dead -> not counted
  };

  void cleanup() {
    std::filesystem::remove_all(SIMNAME);
  }

  auto read_block(adios2::Engine&           reader,
                  adios2::Variable<real_t>& var,
                  std::vector<real_t>&      data) -> adios2::Dims {
    for (const auto& blk : reader.BlocksInfo(var, reader.CurrentStep())) {
      std::size_t ntot { 1 };
      for (const auto& c : blk.Count) {
        ntot *= c;
      }
      if (ntot == 0) {
        continue;
      }
      var.SetBlockSelection(blk.BlockID);
      data.resize(ntot);
      reader.Get(var, data.data(), adios2::Mode::Sync);
      return blk.Count;
    }
    raise::Error("no non-empty block written for " + var.Name(), HERE);
    return {};
  }
} // namespace

auto main(int argc, char* argv[]) -> int {
  GlobalInitialize(argc, argv);

  try {
    using M = Minkowski<Dim::_2D>;

    const std::vector<ncells_t> res { NX1, NX2 };
    const boundaries_t<real_t>  extent {
       { static_cast<real_t>(0.0), static_cast<real_t>(NX1) },
       { static_cast<real_t>(0.0), static_cast<real_t>(NX2) }
    };
    const boundaries_t<FldsBC> fldsbc {
      { FldsBC::PERIODIC, FldsBC::PERIODIC },
      { FldsBC::PERIODIC, FldsBC::PERIODIC }
    };
    const boundaries_t<PrtlBC> prtlbc {
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC },
      { PrtlBC::PERIODIC, PrtlBC::PERIODIC }
    };
    const std::vector<int> decomp { -1, -1 };

    const std::vector<ParticleSpecies> species_params {
      ParticleSpecies { static_cast<spidx_t>(1),
                       "e-", 1.0f,
                       -1.0f,
                       static_cast<npart_t>(100),
                       timestep_t { 0 },
                       timestep_t { 0 },
                       ParticlePusher::BORIS,
                       false, RadiativeDrag::NONE,
                       EmissionType::NONE,
                       static_cast<unsigned short>(0),
                       static_cast<unsigned short>(0) }
    };

#if !defined(MPI_ENABLED)
    const unsigned int ndomains { 1 };
    adios2::ADIOS      adios;
#else
    int mpi_size;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    raise::ErrorIf(mpi_size != 2, "this test requires exactly 2 MPI ranks", HERE);
    const unsigned int ndomains { static_cast<unsigned int>(mpi_size) };
    adios2::ADIOS      adios { MPI_COMM_WORLD };
#endif

    const auto out_step = timestep_t { 5 };
    const auto out_time = simtime_t { 1.25 };

    Metadomain<SimEngine::SRPIC, M> md { ndomains, decomp, res, extent,
                                         fldsbc,   prtlbc, {},  species_params };

    auto*      local = md.subdomain_ptr(md.l_subdomain_indices()[0]);
    const auto off   = local->offset_ncells();
    const auto nloc  = local->mesh.n_active();

    // place the particles that fall into this subdomain
    {
      auto& sp = local->species[0];

      auto i1_h  = Kokkos::create_mirror_view(sp.i1);
      auto i2_h  = Kokkos::create_mirror_view(sp.i2);
      auto dx1_h = Kokkos::create_mirror_view(sp.dx1);
      auto dx2_h = Kokkos::create_mirror_view(sp.dx2);
      auto ux1_h = Kokkos::create_mirror_view(sp.ux1);
      auto w_h   = Kokkos::create_mirror_view(sp.weight);
      auto tag_h = Kokkos::create_mirror_view(sp.tag);

      npart_t np { 0 };
      for (const auto& p : PRTLS) {
        if (p.gi1 < off[0] or p.gi1 >= off[0] + nloc[0] or p.gi2 < off[1] or
            p.gi2 >= off[1] + nloc[1]) {
          continue;
        }
        i1_h(np)  = static_cast<int>(p.gi1 - off[0]);
        i2_h(np)  = static_cast<int>(p.gi2 - off[1]);
        dx1_h(np) = static_cast<prtldx_t>(0.5);
        dx2_h(np) = static_cast<prtldx_t>(0.5);
        ux1_h(np) = math::sqrt(p.gamma * p.gamma - ONE);
        w_h(np)   = p.weight;
        tag_h(np) = p.alive ? ParticleTag::alive : ParticleTag::dead;
        ++np;
      }
      sp.set_npart(np);

      Kokkos::deep_copy(sp.i1, i1_h);
      Kokkos::deep_copy(sp.i2, i2_h);
      Kokkos::deep_copy(sp.dx1, dx1_h);
      Kokkos::deep_copy(sp.dx2, dx2_h);
      Kokkos::deep_copy(sp.ux1, ux1_h);
      Kokkos::deep_copy(sp.weight, w_h);
      Kokkos::deep_copy(sp.tag, tag_h);
    }

    SimulationParams params;
    params.set("simulation.name", SIMNAME);
    params.set("output.format", std::string { "BPFile" });
    params.set("output.debug.ghosts", false);
    params.set("output.fields.quantities", std::vector<std::string> {});
    params.set("output.fields.custom", std::vector<std::string> {});
    params.set("output.fields.downsampling", std::vector<unsigned int> { 1, 1 });
    params.set("output.particles.species", std::vector<spidx_t> {});
    params.set("output.spectra.e_min", E_MIN);
    params.set("output.spectra.e_max", E_MAX);
    params.set("output.spectra.log_bins", false);
    params.set("output.spectra.num_energy_bins", NBE);
    params.set("output.spectra.num_spatial_bins",
               std::vector<std::size_t> { NB1, NB2 });
    for (const auto& type : { "fields", "particles", "spectra" }) {
      params.set("output." + std::string(type) + ".interval", timestep_t { 1 });
      params.set("output." + std::string(type) + ".interval_time",
                 simtime_t { -1.0 });
    }
    params.setRawData(toml::value { toml::table {} });

    md.InitWriter(&adios, params);
    md.WriteSpectra(params, local, out_step, out_time);

    adios.FlushAll();

    // ── read back ─────────────────────────────────────────────────────────────
    {
      adios2::IO io = adios.DeclareIO("read-spectra");
      io.SetEngine("BPFile");

      namespace fs          = std::filesystem;
      adios2::Engine reader = io.Open(
        fs::path(SIMNAME) / fs::path("spectra") /
          fs::path(fmt::format("spectra.%08lu.bp", out_step)),
        adios2::Mode::Read);
      raise::ErrorIf(reader.BeginStep() != adios2::StepStatus::OK,
                     "no step in the spectra file",
                     HERE);

      {
        auto var = io.InquireVariable<real_t>("sN_1");
        raise::ErrorIf(not var, "sN_1 not found", HERE);
        std::vector<real_t> dn;
        const auto          shape = read_block(reader, var, dn);

        raise::ErrorIf(
          shape != adios2::Dims({ NB1, NB2, NBE }),
          fmt::format(
            "sN_1 shape is %s, expected %s",
            fmt::formatVector(shape).c_str(),
            fmt::formatVector(std::vector<std::size_t> { NB1, NB2, NBE }).c_str()),
          HERE);

        std::vector<real_t> expected(NB1 * NB2 * NBE, ZERO);
        for (const auto& p : PRTLS) {
          if (p.alive) {
            expected[(p.b1 * NB2 + p.b2) * NBE + p.be] += p.weight;
          }
        }
        for (auto i { 0u }; i < expected.size(); ++i) {
          raise::ErrorIf(not cmp::AlmostEqual(dn[i], expected[i]),
                         fmt::format("sN_1[%u] = %f, expected %f",
                                     i,
                                     static_cast<double>(dn[i]),
                                     static_cast<double>(expected[i])),
                         HERE);
        }
      }

      // bin edges: energy in [e_min, e_max], spatial in physical units
      const std::vector<std::pair<std::string, std::vector<real_t>>> bins {
        {  "sEbn",   { 0.0, 1.0, 2.0, 3.0, 4.0 } },
        { "sX1bn", { 0.0, 4.0, 8.0, 12.0, 16.0 } },
        { "sX2bn",       { 0.0, 4.0, 8.0, 12.0 } }
      };
      for (const auto& [name, expected] : bins) {
        auto var = io.InquireVariable<real_t>(name);
        raise::ErrorIf(not var, name + " not found", HERE);
        std::vector<real_t> edges;
        const auto          shape = read_block(reader, var, edges);
        raise::ErrorIf(shape != adios2::Dims({ expected.size() }),
                       fmt::format("%s has %lu edges, expected %lu",
                                   name.c_str(),
                                   shape.empty() ? 0ul : shape[0],
                                   expected.size()),
                       HERE);
        for (auto i { 0u }; i < expected.size(); ++i) {
          raise::ErrorIf(not cmp::AlmostEqual(edges[i], expected[i]),
                         fmt::format("%s[%u] = %f, expected %f",
                                     name.c_str(),
                                     i,
                                     static_cast<double>(edges[i]),
                                     static_cast<double>(expected[i])),
                         HERE);
        }
      }

      reader.EndStep();
      reader.Close();
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

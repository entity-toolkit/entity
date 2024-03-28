#ifdef MPI_ENABLED
  #include "wrapper.h"

  #if defined(SANDBOX_ENGINE)

    #include "sandbox.h"
template <ntt::Dimension D>
using SimEngine = ntt::SANDBOX<D>;

  #elif defined(PIC_ENGINE)

    #include "pic.h"
template <ntt::Dimension D>
using SimEngine = ntt::PIC<D>;

  #elif defined(GRPIC_ENGINE)

    #include "grpic.h"
template <ntt::Dimension D>
using SimEngine = ntt::GRPIC<D>;

  #endif

  #include "sim_params.h"

  #include "communications/decomposition.h"
  #include "communications/metadomain.h"
  #include "meshblock/meshblock.h"
  #include "utilities/qmath.h"

  #include "utilities/injector.hpp"

  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
  #include <mpi.h>
  #include <toml.hpp>

  #include <cstdio>
  #include <iomanip>
  #include <iostream>
  #include <stdexcept>
  #include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, output, algorithm;
    toml::table particles, species_1, species_2;
    const auto  simname  = "Writer-" + std::string(SIMULATION_METRIC);
    simulation["title"]  = simname;
    domain["resolution"] = toml::array { 64, 64 };

    particles["n_species"] = 2;
    species_1["mass"]      = 0.0;
    species_1["charge"]    = 0.0;
    species_1["maxnpart"]  = 1e2;
    species_2["mass"]      = 0.0;
    species_2["charge"]    = 0.0;
    species_2["maxnpart"]  = 1e2;

  #ifdef MINKOWSKI_METRIC
    domain["extent"]     = toml::array { -1.0, 1.0, -1.0, 1.0 };
    domain["boundaries"] = toml::array { toml::array { "PERIODIC" },
                                         toml::array { "PERIODIC" } };
  #else
    domain["extent"]     = toml::array { 0.8, 20.0 };
    domain["boundaries"] = toml::array {
      toml::array { "OPEN", "ABSORB" },
      toml::array { "AXIS" }
    };
    domain["qsph_r0"] = 0.0;
    domain["qsph_h"]  = 0.4;
    domain["spin"]    = 0.5;
  #endif

    units["ppc0"]       = 1.0;
    units["larmor0"]    = 0.1;
    units["skindepth0"] = 1.0;

    // output["fields"]      = toml::array { "E", "B" };
    output["particles"]   = toml::array { "X", "U" };
    output["prtl_stride"] = 1;
    output["format"]      = "HDF5";
    output["as_is"]       = true;
    output["ghosts"]      = true;

    auto inputdata = toml::table {
      {"simulation", simulation},
      {    "domain",     domain},
      {     "units",      units},
      {    "output",     output},
      { "particles",  particles},
      { "species_1",  species_1},
      { "species_2",  species_2}
    };

    // write
    {
      SimEngine<ntt::Dim2> sim(inputdata);
      auto&                mblock { sim.meshblock };
      // allocate fields
      mblock.em   = ntt::ndfield_t<ntt::Dim2, 6> { "em",
                                                   mblock.Ni1() + 2 * N_GHOSTS,
                                                   mblock.Ni2() + 2 * N_GHOSTS };
      mblock.bckp = ntt::ndfield_t<ntt::Dim2, 6> { "bckp",
                                                   mblock.Ni1() + 2 * N_GHOSTS,
                                                   mblock.Ni2() + 2 * N_GHOSTS };

      // allocate particles
      for (auto& specie : mblock.particles) {
        specie.i1     = ntt::array_t<int*> { specie.label() + "_i1",
                                             specie.maxnpart() };
        specie.i2     = ntt::array_t<int*> { specie.label() + "_i2",
                                             specie.maxnpart() };
        specie.dx1    = ntt::array_t<prtldx_t*> { specie.label() + "_dx1",
                                                  specie.maxnpart() };
        specie.dx2    = ntt::array_t<prtldx_t*> { specie.label() + "_dx2",
                                                  specie.maxnpart() };
        specie.ux1    = ntt::array_t<real_t*> { specie.label() + "_ux1",
                                                specie.maxnpart() };
        specie.ux2    = ntt::array_t<real_t*> { specie.label() + "_ux2",
                                                specie.maxnpart() };
        specie.ux3    = ntt::array_t<real_t*> { specie.label() + "_ux3",
                                                specie.maxnpart() };
        specie.weight = ntt::array_t<real_t*> { specie.label() + "_w",
                                                specie.maxnpart() };
  #ifndef MINKOWSKI_METRIC
        specie.phi = ntt::array_t<real_t*> { specie.label() + "_phi",
                                             specie.maxnpart() };
  #endif
        specie.tag = ntt::array_t<short*> { specie.label() + "_tag",
                                            specie.maxnpart() };
      }

      {
        // fill dummy fields
        auto tag = (real_t)sim.metadomain()->mpiRank();
        Kokkos::deep_copy(mblock.em, (real_t)(-100.0));
        Kokkos::parallel_for(
          "FillWithDummies-Flds",
          mblock.rangeActiveCells(),
          Lambda(ntt::index_t i1, ntt::index_t i2) {
            mblock.em(i1, i2, ntt::em::ex1) = tag;
            mblock.em(i1, i2, ntt::em::ex2) = tag + 0.1;
            mblock.em(i1, i2, ntt::em::ex3) = tag + 0.2;
            mblock.em(i1, i2, ntt::em::bx1) = tag + 0.3;
            mblock.em(i1, i2, ntt::em::bx2) = tag + 0.4;
            mblock.em(i1, i2, ntt::em::bx3) = tag + 0.5;
          });
      }

      {
        if (sim.metadomain()->mpiRank() == 0) {
          auto& specie1 = mblock.particles[0];
          auto& specie2 = mblock.particles[1];
          specie1.setNpart(2);
          specie2.setNpart(2);
          Kokkos::parallel_for(
            "FillWithDummies-Prtls",
            specie1.rangeActiveParticles(),
            Lambda(ntt::index_t p) {
              specie1.tag(p) = ntt::ParticleTag::alive;
              specie1.i1(p)  = 1 + p;
              specie1.i2(p)  = 1 + 5 * p;
              specie1.dx1(p) = 0.5;
              specie1.dx2(p) = 0.5;
              specie1.ux1(p) = 1.0 + (real_t)(p * 0.5);
              specie1.ux2(p) = 1.0 + (real_t)(p * 4.5);
              specie1.ux3(p) = 1.0 + (real_t)(p * 0.5);

              specie2.tag(p) = ntt::ParticleTag::alive;
              specie2.i1(p)  = 1 + 3 * p;
              specie2.i2(p)  = 1 + 3 * p;
              specie2.dx1(p) = 0.5;
              specie2.dx2(p) = 0.5;
              specie2.ux1(p) = 0.5 + (real_t)(p * 0.5);
              specie2.ux2(p) = 0.2 + (real_t)(p * 0.5);
              specie2.ux3(p) = -0.1 + (real_t)(p * 0.5);
            });
        }
      }
      {
        // advance the fake simulation
        const auto nsteps = 100;
        for (auto i { 0 }; i < nsteps; ++i) {
          sim.Communicate(ntt::Comm_E | ntt::Comm_B);
          sim.ParticlesPush();
          sim.ParticlesBoundaryConditions();
          sim.Communicate(ntt::Comm_Prtl);
          sim.writer.WriteAll(*sim.params(),
                              *sim.metadomain(),
                              mblock,
                              (real_t)i,
                              (std::size_t)i);
          printf("step: %d, rank: %d, npart1: %ld, npart2: %ld\n",
                 i,
                 sim.metadomain()->mpiRank(),
                 mblock.particles[0].npart(),
                 mblock.particles[1].npart());
        }
      }
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
#endif
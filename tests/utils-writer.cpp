#include "wrapper.h"

#include "sandbox.h"
#include "sim_params.h"

#include "communications/decomposition.h"
#include "communications/metadomain.h"
#include "meshblock/meshblock.h"
#include "utilities/qmath.h"

#include "utilities/injector.hpp"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>
#include <toml.hpp>

#ifdef MPI_ENABLED
  #include <mpi.h>
#endif

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, output;
    toml::table particles, species_1, species_2, species_3;
    const auto  simname  = "Writer-" + std::string(SIMULATION_METRIC);
    simulation["title"]  = simname;
    domain["resolution"] = toml::array { 250, 400 };

    particles["n_species"] = 3;
    species_1["mass"]      = 1.0;
    species_1["charge"]    = -1.0;
    species_1["maxnpart"]  = 1e6;
    species_2["mass"]      = 1.0;
    species_2["charge"]    = 1.0;
    species_2["maxnpart"]  = 1e6;
    species_3["mass"]      = 0.0;
    species_3["charge"]    = 0.0;
    species_3["maxnpart"]  = 1e6;

#ifdef MINKOWSKI_METRIC
    domain["extent"]     = toml::array { -50.0, 50.0, -20.0, 140.0 };
    domain["boundaries"] = toml::array { toml::array { "PERIODIC" },
                                         toml::array { "PERIODIC" } };
#else
    domain["extent"]     = toml::array { 1.0, 150.0 };
    domain["boundaries"] = toml::array {
      toml::array { "OPEN", "ABSORB" },
      toml::array { "AXIS" }
    };
    domain["qsph_r0"] = 0.0;
    domain["qsph_h"]  = 0.4;
    domain["spin"]    = 0.9;
#endif

    units["ppc0"]       = 1.0;
    units["larmor0"]    = 1.0;
    units["skindepth0"] = 1.0;

    output["fields"]      = toml::array { "E", "B" };
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
      { "species_2",  species_2},
      { "species_3",  species_3}
    };

    // write
    {
      ntt::SANDBOX<ntt::Dim2> sim(inputdata);
      auto&                   mblock = sim.meshblock;
      // allocate fields
      mblock.em                      = ntt::ndfield_t<ntt::Dim2, 6> { "em",
                                                                      mblock.Ni1() + 2 * N_GHOSTS,
                                                                      mblock.Ni2() + 2 * N_GHOSTS };
      mblock.bckp                    = ntt::ndfield_t<ntt::Dim2, 6> { "bckp",
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
#ifdef MPI_ENABLED
        auto tag = (real_t)sim.metadomain()->mpiRank();
#else
        auto tag = ZERO;
#endif
        Kokkos::deep_copy(mblock.em, (real_t)(-100.0));
        Kokkos::parallel_for(
          "FillWithDummies",
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
        ntt::InjectInVolume<ntt::Dim2, ntt::SANDBOXEngine>(*sim.params(),
                                                           mblock,
                                                           { 1, 2 },
                                                           2.0);
      }
      sim.Communicate(ntt::Comm_E | ntt::Comm_B);
      sim.writer.WriteAll(*sim.params(), *sim.metadomain(), mblock, ZERO, 0);
      for (auto& specie : mblock.particles) {
        specie.setNpart((std::size_t)(specie.npart() / 2));
      }
      sim.writer.WriteAll(*sim.params(), *sim.metadomain(), mblock, ZERO, 0);
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
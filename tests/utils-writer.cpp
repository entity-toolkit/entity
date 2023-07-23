#include "wrapper.h"

#include "sandbox.h"

#include "communications/decomposition.h"
#include "communications/metadomain.h"
#include "utils/qmath.h"

#include <toml.hpp>

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, output;
    simulation["title"]  = "WriterTest";
    domain["resolution"] = toml::array { 2500, 4000 };

#ifdef MINKOWSKI_METRIC
    domain["extent"] = toml::array { -50.0, 50.0, -20.0, 140.0 };
    domain["boundaries"]
      = toml::array { toml::array { "PERIODIC" }, toml::array { "PERIODIC" } };
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

    output["fields"]    = toml::array { "E", "B" };
    output["format"]    = "HDF5";
    output["as_is"]     = true;
    output["ghosts"]    = true;

    auto inputdata      = toml::table {
           {"simulation", simulation},
           {    "domain",     domain},
           {     "units",      units},
           {    "output",     output}
    };

    ntt::SANDBOX<ntt::Dim2> sim(inputdata);
    auto&                   mblock = sim.meshblock;
    mblock.em                      = ntt::ndfield_t<ntt::Dim2, 6> { "em",
                                                                    mblock.Ni1() + 2 * N_GHOSTS,
                                                                    mblock.Ni2() + 2 * N_GHOSTS };
    mblock.bckp                    = ntt::ndfield_t<ntt::Dim2, 6> { "bckp",
                                                                    mblock.Ni1() + 2 * N_GHOSTS,
                                                                    mblock.Ni2() + 2 * N_GHOSTS };

    {
#ifdef MPI_ENABLED
      auto tag = (real_t)sim.metadomain()->mpiRank();
#else
      auto tag = ZERO;
#endif
      Kokkos::deep_copy(mblock.em, -tag);
      Kokkos::parallel_for(
        "FillWithDummies", mblock.rangeActiveCells(), Lambda(ntt::index_t i1, ntt::index_t i2) {
          mblock.em(i1, i2, ntt::em::ex1) = tag;
          mblock.em(i1, i2, ntt::em::ex2) = tag + 0.1;
          mblock.em(i1, i2, ntt::em::ex3) = tag + 0.2;
          mblock.em(i1, i2, ntt::em::bx1) = tag + 0.3;
          mblock.em(i1, i2, ntt::em::bx2) = tag + 0.4;
          mblock.em(i1, i2, ntt::em::bx3) = tag + 0.5;
        });
    }
    sim.writer.WriteAll(*sim.params(), mblock, ZERO, 0);

  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
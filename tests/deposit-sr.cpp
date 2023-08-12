#include "wrapper.h"

#include "pic.h"
template <ntt::Dimension D>
using SimEngine = ntt::PIC<D>;

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

auto measure_Eflux(ntt::Meshblock<ntt::Dim2, ntt::PICEngine>& mblock, std::size_t i1_measure)
  -> real_t {
  real_t int_E { ZERO };
  Kokkos::parallel_reduce(
    "Integrate_E",
    ntt::CreateRangePolicy<ntt::Dim1>({ N_GHOSTS }, { mblock.Ni2() + N_GHOSTS }),
    Lambda(ntt::index_t i2, real_t & l_int_e) {
      const auto ex1 = INV_4
                       * (mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::ex1)
                          + mblock.em(i1_measure + N_GHOSTS, i2 + 1, ntt::em::ex1)
                          + mblock.em(i1_measure + N_GHOSTS - 1, i2, ntt::em::ex1)
                          + mblock.em(i1_measure + N_GHOSTS - 1, i2 + 1, ntt::em::ex1));
      const auto ex2 = mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::ex2);
      const auto ex3 = INV_2
                       * (mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::ex3)
                          + mblock.em(i1_measure + N_GHOSTS, i2 + 1, ntt::em::ex3));
      ntt::vec_t<ntt::Dim3>   e_hat { ZERO };
      ntt::coord_t<ntt::Dim2> xc_cu { (real_t)i1_measure, (real_t)(i2 - N_GHOSTS) + HALF };
      mblock.metric.v3_Cntrv2Hat(xc_cu, { ex1, ex2, ex3 }, e_hat);

      ntt::coord_t<ntt::Dim2> x1_ph { ZERO }, x2_ph { ZERO };
      mblock.metric.x_Code2Phys({ (real_t)(i1_measure), (real_t)(i2 - N_GHOSTS) }, x1_ph);
      mblock.metric.x_Code2Phys({ (real_t)(i1_measure), (real_t)(i2 - N_GHOSTS) + ONE }, x2_ph);

      const auto r { HALF * (x1_ph[0] + x2_ph[0]) };
      const auto theta { HALF * (x1_ph[1] + x2_ph[1]) };
      const auto dtheta { x2_ph[1] - x1_ph[1] };

      l_int_e += e_hat[0] * r * math::sin(theta) * dtheta;
    },
    int_E);
  return int_E;
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, problem, algorithm, output;
    toml::table particles, species_1, species_2;
    const auto  simname          = "Deposit-" + std::string(SIMULATION_METRIC);
    simulation["title"]          = simname;
    domain["resolution"]         = toml::array { 256, 256 };

    algorithm["CFL"]             = 0.5;
    algorithm["current_filters"] = 4;

    particles["n_species"]       = 2;
    species_1["mass"]            = 1.0;
    species_1["charge"]          = -1.0;
    species_1["maxnpart"]        = 1e2;
    species_2["mass"]            = 1.0;
    species_2["charge"]          = 1.0;
    species_2["maxnpart"]        = 1e2;

    domain["extent"]             = toml::array { 1.0, 10.0 };
    domain["boundaries"]         = toml::array {
      toml::array { "CUSTOM", "ABSORB" },
       toml::array { "AXIS" }
    };
    domain["sph_rabsorb"]   = 9.5;
    domain["qsph_r0"]       = 0.0;
    domain["qsph_h"]        = 0.5;

    problem["x1"]           = std::vector<double> { 5.000, 2.0, 8.0, 4.0, 7.5 };
    problem["x2"]           = std::vector<double> { 1.570796, 0.7854, 2.3562, 0.1, 3.041593 };
    problem["x3"]           = std::vector<double> { 0.0, 0.0, 0.0, 0.0, 0.0 };
    problem["ux1"]          = std::vector<double> { 2.0, 2.0, 3.0, 3.0, 2.0 };
    problem["ux2"]          = std::vector<double> { 4.0, 7.0, 4.0, -5.0, 9.0 };

    units["ppc0"]           = 4.0;
    units["larmor0"]        = 2e-3;
    units["skindepth0"]     = 0.1;

    output["format"]        = "HDF5";
    output["fields"]        = std::vector<std::string> { "E", "B" };
    output["particles"]     = std::vector<std::string> { "X", "U" };
    output["interval_time"] = 0.1;

    auto inputdata          = toml::table {
               {"simulation", simulation},
               {    "domain",     domain},
               {     "units",      units},
               { "particles",  particles},
               { "algorithm",  algorithm},
               {   "problem",    problem},
               { "species_1",  species_1},
               { "species_2",  species_2},
               {    "output",     output}
    };

    SimEngine<ntt::Dim2> sim(inputdata);

    {
      auto& mblock = sim.meshblock;
      const real_t            r1 = 11.3;
      // real_t                  flux_1_mid, flux_2_mid;
      // const real_t            tiny { 1e-6 };

      // std::size_t             i1_1, i1_2;
      // ntt::coord_t<ntt::Dim2> xcu { ZERO };
      // mblock.metric.x_Phys2Code({ r1, 0.0 }, xcu);
      // i1_1 = (std::size_t)xcu[0];
      // mblock.metric.x_Phys2Code({ r2, 0.0 }, xcu);
      // i1_2 = (std::size_t)xcu[0];

      // auto file = std::ofstream("Eflux" + simname + ".csv");
      // file << "time,Eflux_1,Eflux_2\n";

      sim.ResetSimulation();
      sim.InitialStep();
      while (sim.time() < 20.0) {
        sim.StepForward(ntt::DiagFlags_None);
        // if ((sim.time() < 8.0) && (sim.time() > 7.9)) {
        //   flux_1_mid = measure_Eflux(mblock, i1_1);
        // }
        // if ((sim.time() < 16.0) && (sim.time() > 15.9)) {
        //   flux_2_mid = measure_Eflux(mblock, i1_2);
        // }
        // const auto flux1 = measure_Eflux(mblock, i1_1);
        // const auto flux2 = measure_Eflux(mblock, i1_2);
        // file << sim.time() << "," << flux1 << "," << flux2 << "\n";

        // if (sim.time() >= 8.0 && ntt::CloseToZero(flux_1_mid, tiny)) {
        //   throw std::runtime_error("int E_2 is zero when there are charges: "
        //                            + std::to_string(flux_1_mid));
        // }
        // if (sim.time() >= 16.0 && ntt::CloseToZero(flux_2_mid, tiny)) {
        //   throw std::runtime_error("int E_1 is zero when there are charges: "
        //                            + std::to_string(flux_2_mid));
        // }
      }
      // file.close();
      // const auto flux_1 = measure_Eflux(mblock, i1_1);
      // const auto flux_2 = measure_Eflux(mblock, i1_2);
      // if (!ntt::CloseToZero(flux_1, tiny) || !ntt::CloseToZero(flux_2, tiny)) {
      //   throw std::runtime_error("int E is not zero: " + std::to_string(flux_1) + ", "
      //                            + std::to_string(flux_2));
      // }
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();
    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
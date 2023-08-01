#include "wrapper.h"

#include "grpic.h"
template <ntt::Dimension D>
using SimEngine = ntt::GRPIC<D>;

#include <Kokkos_Core.hpp>
#include <toml.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

auto measure_Dflux(ntt::Meshblock<ntt::Dim2, ntt::GRPICEngine>& mblock, std::size_t i1_measure)
  -> real_t {
  real_t int_D { ZERO };
  Kokkos::parallel_reduce(
    "Integrate_D",
    ntt::CreateRangePolicy<ntt::Dim1>({ N_GHOSTS }, { 256 + N_GHOSTS }),
    Lambda(ntt::index_t i2, real_t & l_int_d) {
      const auto dx1 = INV_4
                       * (mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::dx1)
                          + mblock.em(i1_measure + N_GHOSTS, i2 + 1, ntt::em::dx1)
                          + mblock.em(i1_measure + N_GHOSTS - 1, i2, ntt::em::dx1)
                          + mblock.em(i1_measure + N_GHOSTS - 1, i2 + 1, ntt::em::dx1));
      const auto dx2 = mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::dx2);
      const auto dx3 = INV_2
                       * (mblock.em(i1_measure + N_GHOSTS, i2, ntt::em::dx3)
                          + mblock.em(i1_measure + N_GHOSTS, i2 + 1, ntt::em::dx3));
      ntt::vec_t<ntt::Dim3>   d_hat { ZERO };
      ntt::coord_t<ntt::Dim2> xc_cu { (real_t)i1_measure, (real_t)(i2 - N_GHOSTS) + HALF };
      mblock.metric.v3_Cntrv2PhysCntrv(xc_cu, { dx1, dx2, dx3 }, d_hat);

      ntt::coord_t<ntt::Dim2> x1_ph { ZERO }, x2_ph { ZERO };
      mblock.metric.x_Code2Phys({ (real_t)(i1_measure), (real_t)(i2 - N_GHOSTS) }, x1_ph);
      mblock.metric.x_Code2Phys({ (real_t)(i1_measure), (real_t)(i2 - N_GHOSTS) + ONE }, x2_ph);

      const auto r { HALF * (x1_ph[0] + x2_ph[0]) };
      const auto theta { HALF * (x1_ph[1] + x2_ph[1]) };
      const auto dtheta { x2_ph[1] - x1_ph[1] };

      l_int_d += d_hat[0] * r * math::sin(theta) * dtheta;
    },
    int_D);
  return int_D;
}

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    toml::table simulation, domain, units, problem, algorithm;
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

    domain["extent"]             = toml::array { 0.8, 20.0 };
    domain["boundaries"]         = toml::array {
      toml::array { "OPEN", "ABSORB" },
       toml::array { "AXIS" }
    };
    domain["sph_rabsorb"] = 18.0;
    domain["qsph_r0"]     = 0.0;
    domain["qsph_h"]      = 0.5;
    domain["spin"]        = 0.9;

    problem["x1"]         = std::vector<double> { 5.000, 15.0, 18.0, 4.0, 12.0 };
    problem["x2"]         = std::vector<double> { 1.570796, 0.7854, 2.3562, 0.1, 3.041593 };
    problem["x3"]         = std::vector<double> { 0.0, 0.0, 0.0, 0.0, 0.0 };
    problem["ux1"]        = std::vector<double> { 2.0, 2.0, 3.0, 3.0, 2.0 };
    problem["ux2"]        = std::vector<double> { 4.0, 7.0, 4.0, -5.0, 9.0 };

    units["ppc0"]         = 4.0;
    units["larmor0"]      = 2e-3;
    units["skindepth0"]   = 0.1;

    auto inputdata        = toml::table {
             {"simulation", simulation},
             {    "domain",     domain},
             {     "units",      units},
             { "particles",  particles},
             { "algorithm",  algorithm},
             {   "problem",    problem},
             { "species_1",  species_1},
             { "species_2",  species_2}
    };

    SimEngine<ntt::Dim2> sim(inputdata);

    {
      auto&                   mblock = sim.meshblock;
      const real_t            r1 = 8, r2 = 15.5;
      const real_t            tiny { 1e-9 };

      std::size_t             i1_1, i1_2;
      ntt::coord_t<ntt::Dim2> xcu { ZERO };
      mblock.metric.x_Phys2Code({ r1, 0.0 }, xcu);
      i1_1 = (std::size_t)xcu[0];
      mblock.metric.x_Phys2Code({ r2, 0.0 }, xcu);
      i1_2 = (std::size_t)xcu[0];

      real_t flux1_prev, flux2_prev;
      bool   fluxes_set = false;

      sim.ResetSimulation();
      sim.InitialStep();
      while (sim.time() < 15.0) {
        sim.StepForward(ntt::DiagFlags_None);
        if (sim.time() > 12.0) {
          const auto flux_1 = measure_Dflux(mblock, i1_1);
          const auto flux_2 = measure_Dflux(mblock, i1_2);
          if (fluxes_set
              && (!ntt::AlmostEqual(flux1_prev, flux_1)
                  || !ntt::AlmostEqual(flux2_prev, flux_2))) {
            throw std::runtime_error("int D is not constant (t > 12): "
                                     + std::to_string(flux_1) + ", " + std::to_string(flux_2));
          }
          fluxes_set = true;
          flux1_prev = flux_1;
          flux2_prev = flux_2;
        }
        if (sim.time() < 3.5) {
          const auto flux_1 = measure_Dflux(mblock, i1_1);
          if (!ntt::CloseToZero(flux_1, tiny)) {
            printf("flux: %e\n", flux_1);
            throw std::runtime_error("int D at r1 is not zero (t < 4): "
                                     + std::to_string(flux_1));
          }
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
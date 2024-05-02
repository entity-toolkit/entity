#include "framework/parameters.h"

#include "defaults.h"
#include "enums.h"

#include "utils/comparators.h"
#include "utils/error.h"

#include "framework/containers/species.h"

#include <stdio.h>
#include <toml.hpp>

#include <iostream>
#include <stdexcept>

using namespace toml::literals::toml_literals;
const auto mink_1d = u8R"(
[simulation]
  name = "mink1d"
  engine = "srpic"
  runtime = 1.0

[grid]
  resolution = [256]
  extent = [[-1.0, 1.0]]

  [grid.metric]
    metric = "minkowski"

  [grid.boundaries]
    fields = [["PERIODIC"]]
    particles = [["ABSORB", "ABSORB"]]

    [grid.boundaries.absorb]
      coeff = 10.0
      ds = 0.025

[scales]
  larmor0 = 0.1
  skindepth0 = 0.01

[algorithms]
  current_filters = 4

  [algorithms.timestep]
    CFL = 0.45

[particles]
  ppc0 = 10.0
  sort_interval = 100

  [[particles.species]]
    label = "e-"
    mass = 1.0
    charge = -1.0
    maxnpart = 1e2
    pusher = "boris"
    n_payloads = 3

  [[particles.species]]
    label = "p+"
    mass = 1.0
    charge = 200.0
    maxnpart = 1e2
    pusher = "vay"

[setup]
  myfloat = 1e-2
  myint   = 123
  mybool  = true
  myarr   = [1.0, 2.0, 3.0]
  mystr   = "hi"

[output]
  fields = ["Rho", "J", "B"]
  particles = ["X", "U"]
  format = "hdf5"
  mom_smooth = 2
  fields_stride = 1
  prtl_stride = 100
  interval_time = 0.01
)"_toml;

const auto sph_2d = u8R"(
[simulation]
  name = "sph2d"
  engine = "srpic"
  runtime = 100.0

[grid]
  resolution = [64, 64]
  extent = [[1.0, 20.0]]

  [grid.metric]
    metric = "spherical"

  [grid.boundaries]
    fields = [["ATMOSPHERE", "ABSORB"]]
    particles = [["ATMOSPHERE", "ABSORB"]]

    [grid.boundaries.absorb]
      coeff = 10.0

    [grid.boundaries.atmosphere]
      temperature = 0.1
      density = 1.0
      height = 0.1
      species = [1, 2]
      ds = 0.2

[scales]
  larmor0 = 0.01
  skindepth0 = 0.01

[algorithms]
  current_filters = 8

  [algorithms.timestep]
    CFL = 0.9

  [algorithms.gca]
    e_ovr_b_max = 0.95
    larmor_max = 0.025

  [algorithms.synchrotron]
    gamma_rad = 50.0

[particles]
  ppc0 = 25.0
  use_weights = true
  sort_interval = 50


  [[particles.species]]
    label = "e-"
    mass = 1.0
    charge = -1.0
    maxnpart = 1e2
    pusher = "boris,gca"
    n_payloads = 3
    cooling = "synchrotron"

  [[particles.species]]
    label = "e+"
    mass = 1.0
    charge = 1.0
    maxnpart = 1e2
    pusher = "boris,gca"
    cooling = "synchrotron"

  [[particles.species]]
    label = "ph"
    mass = 0.0
    charge = 0.0
    maxnpart = 1e2
    
[setup]

)"_toml;

const auto qks_2d = u8R"(
[simulation]
  name = "qks2d"
  engine = "grpic"
  runtime = 1000.0

[grid]
  resolution = [128, 64]
  extent = [[0.8, 100.0]]

  [grid.metric]
    metric = "qkerr_schild"
    qsph_h = 0.25
    ks_a = 0.99

  [grid.boundaries]
    fields = [["ABSORB"]]
    particles = [["ABSORB"]]

[scales]
  larmor0 = 0.001
  skindepth0 = 0.1

[algorithms]
  current_filters = 8

  [algorithms.timestep]
    CFL = 0.5

  [algorithms.gr]
    pusher_eps = 1e-6
    pusher_niter = 5

[particles]
  ppc0 = 4.0
  sort_interval = 100

  [[particles.species]]
    label = "e-"
    mass = 1.0
    charge = -1.0
    maxnpart = 1e2

  [[particles.species]]
    label = "e+"
    mass = 1.0
    charge = 1.0
    maxnpart = 1e2

[setup]
)"_toml;

template <typename T>
void assert_equal(const T& a, const T& b, const std::string& msg) {
  bool eq = false;
  if constexpr (std::is_floating_point_v<T>) {
    eq = cmp::AlmostEqual(a, b);
    if (!eq) {
      printf("%.12e != %.12e\n", a, b);
    }
  } else if constexpr (std::is_class_v<T> &&
                       not std::is_same_v<std::string, std::decay_t<T>>) {
    static_assert(std::is_member_function_pointer_v<decltype(&T::to_string)>);
    eq = (a == b);
    if (!eq) {
      std::cout << T(a).to_string() << " != " << T(b).to_string() << std::endl;
    }
  } else {
    eq = (a == b);
    if (!eq) {
      std::cout << a << " != " << b << std::endl;
    }
  }
  raise::ErrorIf(!eq, msg, HERE);
}

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);

  try {
    using namespace ntt;

    {
      const auto params_mink_1d = SimulationParams(mink_1d);

      assert_equal<Metric>(params_mink_1d.get<Metric>("grid.metric.metric"),
                           Metric::Minkowski,
                           "grid.metric.metric");
      //  engine
      assert_equal<SimEngine>(
        params_mink_1d.get<SimEngine>("simulation.engine"),
        SimEngine::SRPIC,
        "simulation.engine");

      assert_equal(params_mink_1d.get<real_t>("scales.dx0"),
                   (real_t)0.0078125,
                   "scales.dx0");
      assert_equal(params_mink_1d.get<real_t>("scales.V0"),
                   (real_t)0.0078125,
                   "scales.V0");
      boundaries_t<FldsBC> fbc = {
        {FldsBC::PERIODIC, FldsBC::PERIODIC}
      };
      assert_equal<FldsBC>(
        params_mink_1d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[0].first,
        fbc[0].first,
        "grid.boundaries.fields[0].first");
      assert_equal<FldsBC>(
        params_mink_1d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[0].second,
        fbc[0].second,
        "grid.boundaries.fields[0].second");
      assert_equal(
        params_mink_1d.get<boundaries_t<FldsBC>>("grid.boundaries.fields").size(),
        fbc.size(),
        "grid.boundaries.fields.size()");

      assert_equal(params_mink_1d.get<std::size_t>("particles.nspec"),
                   (std::size_t)2,
                   "particles.nspec");

      const auto species = params_mink_1d.get<std::vector<ParticleSpecies>>(
        "particles.species");
      assert_equal<std::string>(species[0].label(), "e-", "species[0].label");
      assert_equal(species[0].mass(), 1.0f, "species[0].mass");
      assert_equal(species[0].charge(), -1.0f, "species[0].charge");
      assert_equal<std::size_t>(species[0].maxnpart(), 100, "species[0].maxnpart");
      assert_equal<PrtlPusher>(species[0].pusher(),
                               PrtlPusher::BORIS,
                               "species[0].pusher");
      assert_equal<unsigned short>(species[0].npld(), 3, "species[0].npld");

      assert_equal<std::string>(species[1].label(), "p+", "species[1].label");
      assert_equal(species[1].mass(), 1.0f, "species[1].mass");
      assert_equal(species[1].charge(), 200.0f, "species[1].charge");
      assert_equal<std::size_t>(species[1].maxnpart(), 100, "species[1].maxnpart");
      assert_equal<PrtlPusher>(species[1].pusher(),
                               PrtlPusher::VAY,
                               "species[1].pusher");
      assert_equal<unsigned short>(species[1].npld(), 0, "species[1].npld");

      assert_equal<real_t>(params_mink_1d.get<real_t>("setup.myfloat"),
                           (real_t)(1e-2),
                           "setup.myfloat");
      assert_equal<int>(params_mink_1d.get<int>("setup.myint"),
                        (int)(123),
                        "setup.myint");
      assert_equal<bool>(params_mink_1d.get<bool>("setup.mybool"),
                         true,
                         "setup.mybool");
      const auto myarr = params_mink_1d.get<std::vector<real_t>>("setup.myarr");
      assert_equal<real_t>(myarr[0], 1.0, "setup.myarr[0]");
      assert_equal<real_t>(myarr[1], 2.0, "setup.myarr[1]");
      assert_equal<real_t>(myarr[2], 3.0, "setup.myarr[2]");
      assert_equal<std::string>(params_mink_1d.get<std::string>("setup.mystr"),
                                "hi",
                                "setup.mystr");
    }

    {
      const auto params_sph_2d = SimulationParams(sph_2d);

      assert_equal<Metric>(params_sph_2d.get<Metric>("grid.metric.metric"),
                           Metric::Spherical,
                           "grid.metric.metric");

      assert_equal<SimEngine>(params_sph_2d.get<SimEngine>("simulation.engine"),
                              SimEngine::SRPIC,
                              "simulation.engine");

      boundaries_t<FldsBC> fbc = {
        {FldsBC::ATMOSPHERE, FldsBC::ABSORB},
        {      FldsBC::AXIS,   FldsBC::AXIS}
      };

      assert_equal<real_t>(params_sph_2d.get<real_t>("scales.B0"),
                           (real_t)(100.0),
                           "scales.B0");
      assert_equal<real_t>(params_sph_2d.get<real_t>("scales.sigma0"),
                           (real_t)(1.0),
                           "scales.sigma0");
      assert_equal<real_t>(params_sph_2d.get<real_t>("scales.omegaB0"),
                           (real_t)(100.0),
                           "scales.omegaB0");

      assert_equal<FldsBC>(
        params_sph_2d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[0].first,
        fbc[0].first,
        "grid.boundaries.fields[0].first");
      assert_equal<FldsBC>(
        params_sph_2d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[0].second,
        fbc[0].second,
        "grid.boundaries.fields[0].second");

      assert_equal<FldsBC>(
        params_sph_2d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[1].first,
        fbc[1].first,
        "grid.boundaries.fields[0].first");
      assert_equal<FldsBC>(
        params_sph_2d.get<boundaries_t<FldsBC>>("grid.boundaries.fields")[1].second,
        fbc[1].second,
        "grid.boundaries.fields[0].second");
      assert_equal(
        params_sph_2d.get<boundaries_t<FldsBC>>("grid.boundaries.fields").size(),
        fbc.size(),
        "grid.boundaries.fields.size()");

      // absorb coeffs
      assert_equal<real_t>(
        params_sph_2d.get<real_t>("grid.boundaries.absorb.ds"),
        (real_t)(defaults::bc::absorb::ds_frac * 19.0),
        "grid.boundaries.absorb.ds");

      assert_equal<real_t>(
        params_sph_2d.get<real_t>("grid.boundaries.absorb.coeff"),
        (real_t)10.0,
        "grid.boundaries.absorb.coeff");

      assert_equal(params_sph_2d.get<bool>("particles.use_weights"),
                   true,
                   "particles.use_weights");

      assert_equal(params_sph_2d.get<std::size_t>("particles.nspec"),
                   (std::size_t)3,
                   "particles.nspec");

      assert_equal(params_sph_2d.get<real_t>("algorithms.gca.e_ovr_b_max"),
                   (real_t)0.95,
                   "algorithms.gca.e_ovr_b_max");
      assert_equal(params_sph_2d.get<real_t>("algorithms.gca.larmor_max"),
                   (real_t)0.025,
                   "algorithms.gca.larmor_max");

      assert_equal(
        params_sph_2d.get<real_t>("algorithms.synchrotron.gamma_rad"),
        (real_t)50.0,
        "algorithms.synchrotron.gamma_rad");

      const auto species = params_sph_2d.get<std::vector<ParticleSpecies>>(
        "particles.species");
      assert_equal<std::string>(species[0].label(), "e-", "species[0].label");
      assert_equal(species[0].mass(), 1.0f, "species[0].mass");
      assert_equal(species[0].charge(), -1.0f, "species[0].charge");
      assert_equal<std::size_t>(species[0].maxnpart(), 100, "species[0].maxnpart");
      assert_equal<PrtlPusher>(species[0].pusher(),
                               PrtlPusher::BORIS,
                               "species[0].pusher");
      assert_equal<PrtlPusher>(species[0].use_gca(), true, "species[0].use_gca");
      assert_equal<unsigned short>(species[0].npld(), 3, "species[0].npld");
      assert_equal<Cooling>(species[0].cooling(),
                            Cooling::SYNCHROTRON,
                            "species[0].cooling");

      assert_equal<std::string>(species[1].label(), "e+", "species[1].label");
      assert_equal(species[1].mass(), 1.0f, "species[1].mass");
      assert_equal(species[1].charge(), 1.0f, "species[1].charge");
      assert_equal<std::size_t>(species[1].maxnpart(), 100, "species[1].maxnpart");
      assert_equal<PrtlPusher>(species[1].pusher(),
                               PrtlPusher::BORIS,
                               "species[1].pusher");
      assert_equal<PrtlPusher>(species[1].use_gca(), true, "species[1].use_gca");
      assert_equal<unsigned short>(species[1].npld(), 0, "species[1].npld");
      assert_equal<Cooling>(species[1].cooling(),
                            Cooling::SYNCHROTRON,
                            "species[1].cooling");

      assert_equal<std::string>(species[2].label(), "ph", "species[2].label");
      assert_equal(species[2].mass(), 0.0f, "species[2].mass");
      assert_equal(species[2].charge(), 0.0f, "species[2].charge");
      assert_equal<std::size_t>(species[2].maxnpart(), 100, "species[2].maxnpart");
      assert_equal<PrtlPusher>(species[2].pusher(),
                               PrtlPusher::PHOTON,
                               "species[2].pusher");
      assert_equal<unsigned short>(species[2].npld(), 0, "species[2].npld");
    }

    {
      const auto params_qks_2d = SimulationParams(qks_2d);

      assert_equal<Metric>(params_qks_2d.get<Metric>("grid.metric.metric"),
                           Metric::QKerr_Schild,
                           "grid.metric.metric");

      assert_equal<SimEngine>(params_qks_2d.get<SimEngine>("simulation.engine"),
                              SimEngine::GRPIC,
                              "simulation.engine");

      assert_equal<Coord>(params_qks_2d.get<Coord>("grid.metric.coord"),
                          Coord::Qsph,
                          "grid.metric.coord");

      assert_equal<real_t>(params_qks_2d.get<real_t>("grid.metric.qsph_r0"),
                           (real_t)(0.0),
                           "grid.metric.qsph_r0");
      assert_equal<real_t>(params_qks_2d.get<real_t>("grid.metric.qsph_h"),
                           (real_t)(0.25),
                           "grid.metric.qsph_h");

      assert_equal<real_t>(params_qks_2d.get<real_t>("grid.metric.ks_a"),
                           (real_t)(0.99),
                           "grid.metric.ks_a");
      assert_equal<real_t>(params_qks_2d.get<real_t>("grid.metric.ks_rh"),
                           (real_t)(1.1410673598),
                           "grid.metric.ks_rh");

      const auto expect = std::map<std::string, real_t> {
        {"r0",  0.0},
        { "h", 0.25},
        { "a", 0.99}
      };
      auto read = params_qks_2d.get<std::map<std::string, real_t>>(
        "grid.metric.params");
      for (const auto& [key, val] : expect) {
        assert_equal<real_t>(read.at(key), val, "grid.metric.params");
      }

      // algorithms gr
      assert_equal<real_t>(
        params_qks_2d.get<real_t>("algorithms.gr.pusher_eps"),
        (real_t)(1e-6),
        "algorithms.gr.pusher_eps");
      assert_equal<unsigned short>(
        params_qks_2d.get<unsigned short>("algorithms.gr.pusher_niter"),
        (unsigned short)(5),
        "algorithms.gr.pusher_niter");

      boundaries_t<PrtlBC> pbc = {
        {PrtlBC::HORIZON, PrtlBC::ABSORB},
        {   PrtlBC::AXIS,   PrtlBC::AXIS}
      };

      assert_equal<real_t>(params_qks_2d.get<real_t>("scales.B0"),
                           (real_t)(1000.0),
                           "scales.B0");
      assert_equal<real_t>(params_qks_2d.get<real_t>("scales.sigma0"),
                           (real_t)(10000.0),
                           "scales.sigma0");
      assert_equal<real_t>(params_qks_2d.get<real_t>("scales.omegaB0"),
                           (real_t)(1000.0),
                           "scales.omegaB0");

      assert_equal<PrtlBC>(
        params_qks_2d.get<boundaries_t<PrtlBC>>("grid.boundaries.particles")[0].first,
        pbc[0].first,
        "grid.boundaries.fields[0].first");
      assert_equal<PrtlBC>(
        params_qks_2d.get<boundaries_t<PrtlBC>>("grid.boundaries.particles")[0].second,
        pbc[0].second,
        "grid.boundaries.particles[0].second");

      assert_equal<PrtlBC>(
        params_qks_2d.get<boundaries_t<PrtlBC>>("grid.boundaries.particles")[1].first,
        pbc[1].first,
        "grid.boundaries.particles[0].first");
      assert_equal<PrtlBC>(
        params_qks_2d.get<boundaries_t<PrtlBC>>("grid.boundaries.particles")[1].second,
        pbc[1].second,
        "grid.boundaries.particles[0].second");
      assert_equal(
        params_qks_2d.get<boundaries_t<PrtlBC>>("grid.boundaries.particles").size(),
        pbc.size(),
        "grid.boundaries.particles.size()");

      // absorb coeffs
      assert_equal<real_t>(
        params_qks_2d.get<real_t>("grid.boundaries.absorb.ds"),
        (real_t)(defaults::bc::absorb::ds_frac * (100.0 - 0.8)),
        "grid.boundaries.absorb.ds");

      assert_equal<real_t>(
        params_qks_2d.get<real_t>("grid.boundaries.absorb.coeff"),
        defaults::bc::absorb::coeff,
        "grid.boundaries.absorb.coeff");

      assert_equal(params_qks_2d.get<std::size_t>("particles.nspec"),
                   (std::size_t)2,
                   "particles.nspec");

      const auto species = params_qks_2d.get<std::vector<ParticleSpecies>>(
        "particles.species");
      assert_equal<std::string>(species[0].label(), "e-", "species[0].label");
      assert_equal(species[0].mass(), 1.0f, "species[0].mass");
      assert_equal(species[0].charge(), -1.0f, "species[0].charge");
      assert_equal<std::size_t>(species[0].maxnpart(), 100, "species[0].maxnpart");
      assert_equal<PrtlPusher>(species[0].pusher(),
                               PrtlPusher::BORIS,
                               "species[0].pusher");
      assert_equal<unsigned short>(species[0].npld(), 0, "species[0].npld");

      assert_equal<std::string>(species[1].label(), "e+", "species[1].label");
      assert_equal(species[1].mass(), 1.0f, "species[1].mass");
      assert_equal(species[1].charge(), 1.0f, "species[1].charge");
      assert_equal<std::size_t>(species[1].maxnpart(), 100, "species[1].maxnpart");
      assert_equal<PrtlPusher>(species[1].pusher(),
                               PrtlPusher::BORIS,
                               "species[1].pusher");
      assert_equal<unsigned short>(species[1].npld(), 0, "species[1].npld");
    }

  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }

  Kokkos::finalize();

  return 0;
}

// const auto mink_1d = R"(
// [simulation]
//   name = ""
//   engine = ""
//   runtime = ""

// [grid]
//   resolution = ""
//   extent = ""

//   [grid.metric]
//     metric = ""
//     qsph_r0 = ""
//     qsph_h = ""
//     ks_a = ""

//   [grid.boundaries]
//     fields = ""
//     particles = ""
//     absorb_d = ""
//     absorb_coeff = ""

// [scales]
//   larmor0 = ""
//   skindepth0 = ""

// [algorithms]
//   current_filters = ""

//   [algorithms.toggles]
//     fieldsolver = ""
//     deposit = ""

//   [algorithms.timestep]
//     CFL = ""
//     correction = ""

//   [algorithms.gr]
//     pusher_eps = ""
//     pusher_niter = ""

//   [algorithms.gca]
//     e_ovr_b_max = ""
//     larmor_max = ""

//   [algorithms.synchrotron]
//     gamma_rad = ""

// [particles]
//   ppc0 = ""
//   use_weights = ""
//   sort_interval = ""

//   [[particles.species]]
//     label = ""
//     mass = ""
//     charge = ""
//     maxnpart = ""
//     pusher = ""
//     n_payloads = ""
//     cooling = ""
// [setup]

// [output]
//   fields = ""
//   particles = ""
//   format = ""
//   mom_smooth = ""
//   fields_stride = ""
//   prtl_stride = ""
//   interval = ""
//   interval_time = ""

//   [output.debug]
//     as_is = ""
//     ghosts = ""

// [diagnostics]
//   interval = ""
//   log_level = ""
//   blocking_timers = ""
// )"_toml;

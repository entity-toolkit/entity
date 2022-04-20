// // test external libraries
// #include "test_extern_toml.h"
// #include "test_extern_kokkos.h"

// #include <acutest.h>

// void testSuccess() {}

// TEST_LIST
//     = {{"extern/toml", testExternToml},
//        {"extern/kokkos", testExternKokkos},
//        {"success", testSuccess},
//        {nullptr, nullptr}};

#include "global.h"
#include "cargs.h"
#include "input.h"

#include <toml/toml.hpp>

#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>
#include <vector>
#include <stdexcept>

using plog_t = plog::ColorConsoleAppender<plog::NTTFormatter>;

void initLogger(plog_t* console_appender);

auto main(int argc, char* argv[]) -> int {
  plog_t console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    // auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    // // auto outputpath = cl_args.getArgument("-output", ntt::DEF_output_path);
    // auto  inputdata = toml::parse(static_cast<std::string>(inputfilename));
    // short res = static_cast<short>(ntt::readFromInput<std::vector<int>>(inputdata, "domain", "resolution").size());

    // if (res == 1) {
    //   // ntt::SIMULATION_CONTAINER<ntt::Dimension::ONE_D> sim(inputdata);
    //   // sim.process();
    // } else if (res == 2) {
    //   ntt::SIMULATION_CONTAINER<ntt::Dimension::TWO_D> sim(inputdata);
    //   sim.process();
    // } else if (res == 3) {
    //   ntt::SIMULATION_CONTAINER<ntt::Dimension::THREE_D> sim(inputdata);
    //   sim.process();
    // } else {
    //   NTTError("wrong dimension specified");
    // }
  }
  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}

void initLogger(plog_t* console_appender) {
  plog::Severity max_severity;
#ifdef DEBUG
  max_severity = plog::verbose;
#else
  max_severity = plog::info;
#endif
  plog::init(max_severity, console_appender);
}

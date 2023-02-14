#include "wrapper.h"

#include "cargs.h"
#include "input.h"

#ifdef PIC_ENGINE
#  include "pic.h"
#  define SIMULATION_CONTAINER PIC
#elif defined(GRPIC_ENGINE)
#  include "grpic.h"
#  define SIMULATION_CONTAINER GRPIC
#endif

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml/toml.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

// Logging is done via `plog` library...
// ... Use the following commands:
//  `PLOGI << ...` for general info
//  `PLOGF << ...` for fatal error messages (development)
//  `PLOGD << ...` for debug messages (development)
//  `PLOGE << ...` for simple error messages
//  `PLOGW << ...` for warnings

auto main(int argc, char* argv[]) -> int {
  plog::Severity max_severity;
#ifdef DEBUG
  max_severity = plog::verbose;
#else
  max_severity = plog::info;
#endif
  plog::ColorConsoleAppender<plog::NTTFormatter> consoleAppender;
  plog::RollingFileAppender<plog::TxtFormatter>  fileAppender("entity.log", 1048576, 3);
  plog::init(max_severity, &consoleAppender);
  plog::init<ntt::LogFile>(plog::verbose, &fileAppender);

  Kokkos::initialize();
  try {
    PLOGI << "Kokkos initialized";
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    // auto outputpath = cl_args.getArgument("-output", ntt::DEF_output_path);
    auto inputdata     = toml::parse(static_cast<std::string>(inputfilename));
    PLOGI << "input file parsed";
    short res = static_cast<short>(
      ntt::readFromInput<std::vector<int>>(inputdata, "domain", "resolution").size());
    if (res == 1) {
      ntt::SIMULATION_CONTAINER<ntt::Dim1> sim(inputdata);
      PLOGI << "1D simulation initialized";
      sim.Run();
    } else if (res == 2) {
      ntt::SIMULATION_CONTAINER<ntt::Dim2> sim(inputdata);
      PLOGI << "2D simulation initialized";
      sim.Run();
    } else if (res == 3) {
      ntt::SIMULATION_CONTAINER<ntt::Dim3> sim(inputdata);
      PLOGI << "3D simulation initialized";
      sim.Run();
    } else {
      NTTHostError("wrong dimension specified");
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}
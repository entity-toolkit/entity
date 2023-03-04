#include "wrapper.h"

#include "cargs.h"
#include "input.h"

/**
 * Engine specific instantiations
 */
#if defined(SANDBOX_ENGINE)
#  include "sandbox.h"
#  define SIMULATION_CONTAINER SANDBOX
#elif defined(PIC_ENGINE)
#  include "pic.h"
#  define SIMULATION_CONTAINER PIC
#elif defined(GRPIC_ENGINE)
#  include "grpic.h"
#  define SIMULATION_CONTAINER GRPIC
#endif

#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml/toml.hpp>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto inputdata     = toml::parse(static_cast<std::string>(inputfilename));

    auto sim_title     = ntt::readFromInput<std::string>(
      inputdata, "simulation", "title", ntt::defaults::title);
    auto logfile_name  = sim_title + ".log";
    auto infofile_name = sim_title + ".info";
    std::remove(logfile_name.c_str());
    std::remove(infofile_name.c_str());
    plog::RollingFileAppender<plog::TxtFormatter>     logfileAppender(logfile_name.c_str());
    plog::RollingFileAppender<plog::Nt2InfoFormatter> infofileAppender(infofile_name.c_str());
    plog::init<ntt::LogFile>(plog::verbose, &logfileAppender);
    plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);

    short res = static_cast<short>(
      ntt::readFromInput<std::vector<int>>(inputdata, "domain", "resolution").size());
    if (res == 1) {
      ntt::SIMULATION_CONTAINER<ntt::Dim1> sim(inputdata);
      NTTLog();
      sim.Run();
    } else if (res == 2) {
      ntt::SIMULATION_CONTAINER<ntt::Dim2> sim(inputdata);
      NTTLog();
      sim.Run();
    } else if (res == 3) {
      ntt::SIMULATION_CONTAINER<ntt::Dim3> sim(inputdata);
      NTTLog();
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
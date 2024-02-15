#include "wrapper.h"

#include "io/cargs.h"
#include "io/input.h"

/**
 * Engine specific instantiations
 */
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

#include <plog/Appenders/RollingFileAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>
#include <toml.hpp>

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input",
                                             ntt::defaults::input_filename);
    auto inputdata     = toml::parse(static_cast<std::string>(inputfilename));
    auto log_level     = ntt::readFromInput<std::string>(inputdata,
                                                     "diagnostics",
                                                     "log_level",
                                                     ntt::defaults::log_level);
    plog::Severity log_level_enum { plog::info };
    if (log_level == "DEBUG") {
      log_level_enum = plog::verbose;
    } else if (log_level == "INFO") {
      log_level_enum = plog::info;
    } else if (log_level == "WARNING") {
      log_level_enum = plog::warning;
    } else if (log_level == "ERROR") {
      log_level_enum = plog::error;
    }

    auto sim_title     = ntt::readFromInput<std::string>(inputdata,
                                                     "simulation",
                                                     "title",
                                                     ntt::defaults::title);
    auto logfile_name  = sim_title + ".log";
    auto infofile_name = sim_title + ".info";
    std::remove(logfile_name.c_str());
    std::remove(infofile_name.c_str());
    plog::RollingFileAppender<plog::TxtFormatter> logfileAppender(
      logfile_name.c_str());
    plog::RollingFileAppender<plog::Nt2InfoFormatter> infofileAppender(
      infofile_name.c_str());
    plog::init<ntt::LogFile>(log_level_enum, &logfileAppender);
    plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);

    short res = static_cast<short>(
      ntt::readFromInput<std::vector<int>>(inputdata, "domain", "resolution").size());
    if (res == 1) {
#ifndef GRPIC_ENGINE
      SimEngine<ntt::Dim1> sim(inputdata);
      NTTLog();
      sim.Run();
#else
      NTTHostError("GRPIC engine does not support 1D");
#endif
    } else if (res == 2) {
      SimEngine<ntt::Dim2> sim(inputdata);
      NTTLog();
      sim.Run();
    } else if (res == 3) {
      SimEngine<ntt::Dim3> sim(inputdata);
      NTTLog();
      sim.Run();
    } else {
      NTTHostError("wrong dimension specified");
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();

    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}
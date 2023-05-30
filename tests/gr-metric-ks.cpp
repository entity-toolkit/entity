#include "wrapper.h"

#include "io/cargs.h"
// #include "grpic.h"
#include "io/input.h"

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

auto main() -> int {
  Kokkos::initialize();
  try {
    using namespace toml::literals::toml_literals;
    const auto inputdata      = R"(
      [simulation]
      title   = "gr-metric-ks"

      [domain]
      a           = 0.0
      resolution  = [512, 256]
      extent      = [1.0, 20.0]
      boundaries  = [["OPEN", "ABSORB"], ["AXIS"]]
    )"_toml;

    // const auto inputdata      = toml::parse(input);
    const auto log_level_enum = plog::verbose;
    const auto sim_title = ntt::readFromInput<std::string>(inputdata, "simulation", "title");
    const auto logfile_name  = sim_title + ".log";
    const auto infofile_name = sim_title + ".info";
    std::remove(logfile_name.c_str());
    std::remove(infofile_name.c_str());
    plog::RollingFileAppender<plog::TxtFormatter>     logfileAppender(logfile_name.c_str());
    plog::RollingFileAppender<plog::Nt2InfoFormatter> infofileAppender(infofile_name.c_str());
    plog::init<ntt::LogFile>(log_level_enum, &logfileAppender);
    plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);

    // ntt::GRPIC<ntt::Dim2> sim(inputdata);
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}
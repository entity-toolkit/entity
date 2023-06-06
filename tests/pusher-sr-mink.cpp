#include "wrapper.h"

#include METRIC_HEADER
#include "sim_params.h"

#include "io/input.h"
#include "meshblock/meshblock.h"
#include "utils/qmath.h"

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
  Kokkos::initialize(argc, argv);
  try {
    using namespace toml::literals::toml_literals;
    const auto inputdata = R"(
      [domain]
      resolution  = [256, 128, 128]
      extent      = [-10.0, 10.0, -5.0, 5.0, 0.0, 10.0]
      boundaries  = [["PERIODIC"], ["PERIODIC"], ["PERIODIC"]]
    )"_toml;

    const auto resolution
      = ntt::readFromInput<std::vector<unsigned int>>(inputdata, "domain", "resolution");
    const auto extent = ntt::readFromInput<std::vector<real_t>>(inputdata, "domain", "extent");

    auto       params = ntt::SimulationParams(inputdata, ntt::Dim3);

    // const auto log_level_enum = plog::verbose;
    // const auto sim_title = ntt::readFromInput<std::string>(inputdata, "simulation",
    // "title"); const auto logfile_name  = sim_title + ".log"; const auto infofile_name =
    // sim_title + ".info"; std::remove(logfile_name.c_str());
    // std::remove(infofile_name.c_str());
    // plog::RollingFileAppender<plog::TxtFormatter>     logfileAppender(logfile_name.c_str());
    // plog::RollingFileAppender<plog::Nt2InfoFormatter>
    // infofileAppender(infofile_name.c_str()); plog::init<ntt::LogFile>(log_level_enum,
    // &logfileAppender); plog::init<ntt::InfoFile>(plog::verbose, &infofileAppender);

    auto       mblock = ntt::Meshblock<ntt::Dim3, ntt::PICEngine>(
      params.resolution(), params.extent(), params.metricParameters(), params.species());

    {
      bool correct;

      (!correct) ? throw std::logic_error("Metric is incorrect") : (void)0;
    }
  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();
    return -1;
  }
  Kokkos::finalize();

  return 0;
}
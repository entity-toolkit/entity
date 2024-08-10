#include "framework/simulation.h"

#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "utils/cargs.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/plog.h"
#include "utils/toml.h"

#include "framework/parameters.h"

#include <string>

namespace ntt {

  Simulation::Simulation(int argc, char* argv[]) {
    GlobalInitialize(argc, argv);

    cargs::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    const auto inputfname = static_cast<std::string>(
      cl_args.getArgument("-input", defaults::input_filename));
    const auto outputdir = static_cast<std::string>(
      cl_args.getArgument("-output", defaults::output_path));

    raw_params = toml::parse(inputfname);
    const auto sim_name = toml::find<std::string>(raw_params, "simulation", "name");
    logger::initPlog<files::LogFile, files::InfoFile, files::ErrFile>(sim_name);

    m_requested_engine = SimEngine::pick(
      fmt::toLower(toml::find<std::string>(raw_params, "simulation", "engine")).c_str());
    m_requested_metric = Metric::pick(
      fmt::toLower(toml::find<std::string>(raw_params, "grid", "metric", "metric"))
        .c_str());

    const auto res = toml::find<std::vector<std::size_t>>(raw_params,
                                                          "grid",
                                                          "resolution");
    raise::ErrorIf(res.size() < 1 || res.size() > 3,
                   "invalid `grid.resolution`",
                   HERE);
    m_requested_dimension = static_cast<Dimension>(res.size());
  }

  Simulation::~Simulation() {
    GlobalFinalize();
  }

} // namespace ntt

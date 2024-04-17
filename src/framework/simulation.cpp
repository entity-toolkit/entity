#include "framework/simulation.h"

#include "defaults.h"
#include "global.h"

#include "utils/cargs.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/plog.h"

#include "framework/parameters.h"

#include <toml.hpp>

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

    const auto inputdata = toml::parse(inputfname);
    const auto sim_name = toml::find<std::string>(inputdata, "simulation", "name");
    logger::initPlog<files::LogFile, files::InfoFile, files::ErrFile>(sim_name);

    params = SimulationParams(inputdata);
  }

  Simulation::~Simulation() {
    GlobalFinalize();
  }

} // namespace ntt
#include "framework/simulation.h"

#include "defaults.h"
#include "enums.h"
#include "global.h"

#include "utils/cargs.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/plog.h"
#include "utils/toml.h"

#include <filesystem>
#include <string>

namespace ntt {

  Simulation::Simulation(int argc, char* argv[]) {
    cargs::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    const auto inputfname = static_cast<std::string>(
      cl_args.getArgument("-input", defaults::input_filename));

    const bool is_resuming = (cl_args.isSpecified("-continue") or
                              cl_args.isSpecified("-restart") or
                              cl_args.isSpecified("-resume") or
                              cl_args.isSpecified("-checkpoint"));
    GlobalInitialize(argc, argv);

    const auto raw_params = toml::parse(inputfname);
    const auto sim_name = toml::find<std::string>(raw_params, "simulation", "name");
    const auto log_level = toml::find_or<std::string>(raw_params,
                                                      "diagnostics",
                                                      "log_level",
                                                      defaults::diag::log_level);
    logger::initPlog<files::LogFile, files::InfoFile, files::ErrFile>(sim_name,
                                                                      log_level);

    m_requested_engine = SimEngine::pick(
      fmt::toLower(toml::find<std::string>(raw_params, "simulation", "engine")).c_str());
    m_requested_metric = Metric::pick(
      fmt::toLower(toml::find<std::string>(raw_params, "grid", "metric", "metric"))
        .c_str());

    const auto res = toml::find<std::vector<ncells_t>>(raw_params,
                                                       "grid",
                                                       "resolution");
    raise::ErrorIf(res.size() < 1 || res.size() > 3,
                   "invalid `grid.resolution`",
                   HERE);
    m_requested_dimension = static_cast<Dimension>(res.size());

    m_params.setRawData(raw_params);
    timestep_t checkpoint_step = 0;

    if (is_resuming) {
      logger::Checkpoint("Reading params from a checkpoint", HERE);
      const auto checkpoint_write_path = toml::find_or<std::string>(
        raw_params,
        "checkpoint",
        "write_path",
        fmt::format(defaults::checkpoint::write_path.c_str(), sim_name.c_str()));
      const path_t checkpoint_read_path = toml::find_or<std::string>(
        raw_params,
        "checkpoint",
        "read_path",
        checkpoint_write_path);
      if (not std::filesystem::exists(checkpoint_read_path)) {
        raise::Fatal("No checkpoints found", HERE);
      }
      for (const auto& entry :
           std::filesystem::directory_iterator(checkpoint_read_path)) {
        const auto fname = entry.path().filename().string();
        if (fname.find("step-") == 0) {
          const timestep_t step = std::stoi(fname.substr(5, fname.size() - 5 - 3));
          if (step > checkpoint_step) {
            checkpoint_step = step;
          }
        }
      }
      path_t checkpoint_metafname = checkpoint_read_path /
                                    fmt::format("meta-%08lu.toml", checkpoint_step);
      if (not std::filesystem::exists(checkpoint_metafname)) {
        raise::Fatal(
          fmt::format("metainformation for %lu not found", checkpoint_step),
          HERE);
        checkpoint_metafname = inputfname;
      }
      logger::Checkpoint(fmt::format("Using %08lu", checkpoint_step), HERE);
      const auto raw_checkpoint_params = toml::parse(checkpoint_metafname);
      const auto start_time = toml::find<simtime_t>(raw_checkpoint_params,
                                                    "metadata",
                                                    "time");
      m_params.setImmutableParams(raw_checkpoint_params);
      m_params.setMutableParams(raw_params);
      m_params.setCheckpointParams(true, checkpoint_step, start_time);
      m_params.setSetupParams(raw_checkpoint_params);
    } else {
      logger::Checkpoint("Defining new params", HERE);
      m_params.setImmutableParams(raw_params);
      m_params.setMutableParams(raw_params);
      m_params.setCheckpointParams(false, 0, 0.0);
      m_params.setSetupParams(raw_params);
    }
    m_params.checkPromises();
  }

  Simulation::~Simulation() {
    GlobalFinalize();
  }

} // namespace ntt

#include "global.h"
#include "sim_params.h"
#include "cargs.h"

#include <toml/toml.hpp>

#include <string>

namespace ntt {
SimulationParams::SimulationParams(int argc, char *argv[]) {
  CommandLineArguments cl_args;
  cl_args.readCommandLineArguments(argc, argv);
  m_inputfilename = cl_args.getArgument("-input", DEF_input_filename);
  m_outputpath = cl_args.getArgument("-output", DEF_output_path);
  m_inputdata = toml::parse(static_cast<std::string>(m_inputfilename));
}
}

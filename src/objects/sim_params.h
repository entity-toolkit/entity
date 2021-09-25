#ifndef OBJECTS_SIM_PARAMS_H
#define OBJECTS_SIM_PARAMS_H

#include <toml/toml.hpp>

#include <string_view>

namespace ntt {

class SimulationParams {
private:
  std::string_view m_inputfilename;
  std::string_view m_outputpath;
  toml::value m_inputdata;
public:
  SimulationParams(int argc, char *argv[]);
  ~SimulationParams();
};

}

#endif

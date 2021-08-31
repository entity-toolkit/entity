#ifndef SIMULATION_SIM_H
#define SIMULATION_SIM_H

#include "global.h"
#include "arrays.h"

#include <toml/toml.hpp>

#include <vector>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace ntt {
class Simulation {
protected:
  std::string m_title;

  const Dimension m_dimension;
  const CoordinateSystem m_coord_system;
  const SimulationType m_simulation_type;

  bool m_initialized { false };

  std::string_view m_inputfilename;
  std::string_view m_outputpath;
  toml::value m_inputdata;

  std::vector<int> m_resolution;
  std::vector<real_t> m_dimensions;
  real_t m_runtime;
  real_t m_timestep;

public:
  Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type);
  ~Simulation() = default;
  [[nodiscard]] auto get_title() const -> std::string { return m_title; }
  [[nodiscard]] auto get_precision() const -> std::size_t { return sizeof(real_t); }
  [[nodiscard]] auto get_dimension() const -> Dimension { return m_dimension; }
  [[nodiscard]] auto get_coord_system() const -> CoordinateSystem { return m_coord_system; }
  [[nodiscard]] auto get_simulation_type() const -> SimulationType { return m_simulation_type; }

  void parseInput(int argc, char *argv[]);

  template <typename T> auto readFromInput(const std::string &blockname, const std::string &variable) -> T;
  template <typename T> auto readFromInput(const std::string &blockname, const std::string &variable, const T &defval) -> T;

  void run() {};

  void initialize();

  void printDetails(std::ostream& os);
  void printDetails();
  // virtual void restart() = 0;
  // virtual void mainloop() = 0;
  // virtual void finalize() = 0;
};
} // namespace ntt

#endif

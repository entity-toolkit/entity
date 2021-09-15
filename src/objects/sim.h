#ifndef OBJECTS_SIM_H
#define OBJECTS_SIM_H

#include "global.h"
#include "domain.h"

#include <toml/toml.hpp>

#include <vector>
#include <cstddef>
#include <string>
#include <string_view>

namespace ntt {
class AbstractSimulation {
protected:
  bool m_initialized{false};
  bool m_inputparsed{false};

public:
  AbstractSimulation() = default;
  ~AbstractSimulation() = default;

  [[nodiscard]] auto is_initialized() const -> bool { return m_initialized; }
  [[nodiscard]] auto is_inputparsed() const -> bool { return m_inputparsed; }

  virtual void parseInput(int argc, char *argv[]) = 0;
  virtual void printDetails(std::ostream &) = 0;
  virtual void printDetails() = 0;
  virtual void initialize() = 0;
  virtual void verify() = 0;
  virtual void mainloop() = 0;
  virtual void finalize() = 0;
};

class Simulation : public AbstractSimulation {
protected:
  std::string m_title;
  const SimulationType m_simulation_type;

  std::string_view m_inputfilename;
  std::string_view m_outputpath;
  toml::value m_inputdata;

  Domain m_domain;

  real_t m_runtime;
  real_t m_timestep;

public:
  Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type);
  ~Simulation() = default;

  [[nodiscard]] auto get_title() const -> std::string { return m_title; }
  [[nodiscard]] auto get_precision() const -> std::size_t { return sizeof(real_t); }
  [[nodiscard]] auto get_simulation_type() const -> SimulationType { return m_simulation_type; }
  [[nodiscard]] auto get_dimension() const -> Dimension { return m_domain.m_dimension; }
  [[nodiscard]] auto get_coord_system() const -> CoordinateSystem { return m_domain.m_coord_system; }
  [[nodiscard]] auto get_resolution() const -> std::vector<int> { return m_domain.m_resolution; }
  [[nodiscard]] auto get_extent() const -> std::vector<real_t> { return m_domain.m_extent; }

  void parseInput(int argc, char *argv[]) override;
  void printDetails(std::ostream &os) override;
  void printDetails() override;

  void initialize() override;
  void verify() override;
  void mainloop() override;
  void finalize() override;

  [[nodiscard]] auto getSizeInBytes() -> std::size_t;
};

} // namespace ntt

#endif

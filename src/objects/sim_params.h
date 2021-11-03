#ifndef OBJECTS_SIM_PARAMS_H
#define OBJECTS_SIM_PARAMS_H

#include "global.h"
#include "particles.h"

#include <toml/toml.hpp>

#include <vector>
#include <string_view>

namespace ntt {

class SimulationParams {
  std::string_view m_inputfilename;
  std::string_view m_outputpath;

  SimulationType m_simtype {UNDEFINED_SIM};

  std::string m_title;
  real_t m_cfl, m_timestep, m_min_cell_size;
  real_t m_correction;
  real_t m_runtime;

  // independent params
  real_t m_ppc0;
  real_t m_larmor0;
  real_t m_skindepth0;

  // dependent params
  real_t m_sigma0;
  real_t m_charge0;
  real_t m_B0;

  std::vector<ParticleSpecies> m_species;
  ParticleShape m_prtl_shape;

  CoordinateSystem m_coord_system {UNDEFINED_COORD};
  std::vector<real_t> m_extent;
  std::vector<std::size_t> m_resolution;
  std::vector<BoundaryCondition> m_boundaries;

public:
  toml::value m_inputdata;

  SimulationParams(const toml::value& inputdata, Dimension dim);
  ~SimulationParams() = default;

  template <Dimension D>
  friend class Simulation;
  friend class Simulation1D;
  friend class Simulation2D;
  friend class Simulation3D;

  template <Dimension D>
  friend struct ProblemGenerator;

  void printDetails();
  void verify();

  [[nodiscard]] auto get_min_timestep() -> real_t;
  [[nodiscard]] auto get_extent() const -> const std::vector<real_t>& { return m_extent; }
  [[nodiscard]] auto get_resolution() const -> const std::vector<std::size_t>& {
    return m_resolution;
  }
  [[nodiscard]] auto get_timestep() const -> real_t { return m_timestep; }
};

} // namespace ntt

#endif

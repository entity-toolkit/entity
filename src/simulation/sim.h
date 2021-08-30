#ifndef SIMULATION_SIM_H
#define SIMULATION_SIM_H

#include "global.h"
#include "arrays.h"
#include "input.h"

#include <toml/toml.hpp>

#include <cstddef>
#include <string>

namespace ntt {
class Simulation {
protected:
  std::string m_title;
  const Dimension m_dimension;
  const CoordinateSystem m_coord_system;
  const SimulationType m_simulation_type;
  toml::value m_inputdata;

public:
  Simulation(Dimension dim, CoordinateSystem coord_sys, SimulationType sim_type);
  ~Simulation() = default;
  void set_title(const std::string &title) { m_title = title; }
  [[nodiscard]] auto get_title() const -> std::string { return m_title; }
  [[nodiscard]] auto get_precision() const -> std::size_t {
    return sizeof(real_t);
  }
  [[nodiscard]] auto get_dimension() const -> Dimension { return m_dimension; }
  [[nodiscard]] auto get_coord_system() const -> CoordinateSystem {
    return m_coord_system;
  }

  void parseInput(int argc, char *argv[]);

  template<typename T>
  T readFromInput(const std::string &blockname, const std::string &variable);

  virtual void initialize() = 0;
  virtual void restart() = 0;
  virtual void mainloop() = 0;
  virtual void finalize() = 0;
};

class PICSimulation1D : public Simulation {
protected:
  // arrays::OneDArray<real_t> ex1, ex2, ex3;
  // arrays::OneDArray<real_t> bx1, bx2, bx3;
  // particle system
public:
  PICSimulation1D(CoordinateSystem coord_sys) : Simulation{ONE_D, coord_sys, PIC_SIM} {}
  ~PICSimulation1D() = default;

  void initialize() override {}
  void restart() override {}
  void mainloop() override {}
  void finalize() override {}
};
} // namespace ntt

#endif

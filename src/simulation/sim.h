#ifndef SIMULATION_SIM_H
#define SIMULATION_SIM_H

#include "global.h"
#include "arrays.h"
#include "domain.h"

#include <toml/toml.hpp>

#include <vector>
#include <cstddef>
#include <string>
#include <stdexcept>

namespace ntt {
class AbstractSimulation {
protected:
  bool m_initialized{false};
  bool m_inputparsed{false};

public:
  AbstractSimulation() = default;
  ~AbstractSimulation() = default;

  [[nodiscard]] auto is_initialized() const -> bool { return m_initialized; }

  virtual void parseInput(int argc, char *argv[]) = 0;
  virtual void printDetails(std::ostream &) = 0;
  virtual void printDetails() = 0;
  virtual void initialize() = 0;
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

  // const Dimension m_dimension;
  // const CoordinateSystem m_coord_system;
  // std::vector<int> m_resolution;
  // std::vector<real_t> m_size;
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

  template <typename T>
  [[nodiscard]] auto readFromInput(const std::string &blockname, const std::string &variable) -> T;
  template <typename T>
  [[nodiscard]] auto readFromInput(const std::string &blockname, const std::string &variable, const T &defval) -> T;

  void initialize() override;
  void mainloop() override;
  void finalize() override;
};

class PICSimulation : public Simulation {
protected:
  ParticlePusher m_pusher{UNDEFINED_PUSHER};

public:
  PICSimulation(Dimension dim, CoordinateSystem coord_sys, ParticlePusher pusher)
      : Simulation{dim, coord_sys, PIC_SIM}, m_pusher(pusher){};
  PICSimulation(Dimension dim, CoordinateSystem coord_sys) : Simulation{dim, coord_sys, PIC_SIM} {};
  ~PICSimulation() = default;
  void printDetails(std::ostream &os) override;
  void mainloop() override;
};

class PICSimulation1D : public PICSimulation {
protected:
  arrays::OneDArray<real_t> ex1, ex2, ex3;
  arrays::OneDArray<real_t> bx1, bx2, bx3;

public:
  PICSimulation1D(ParticlePusher pusher) : PICSimulation{ONE_D, CARTESIAN_COORD, pusher} {};
  PICSimulation1D() : PICSimulation{ONE_D, CARTESIAN_COORD} {};
  ~PICSimulation1D() = default;
  void initialize() override;
  void finalize() override;
};

class PICSimulation2D : public PICSimulation {
protected:
  arrays::TwoDArray<real_t> ex1, ex2, ex3;
  arrays::TwoDArray<real_t> bx1, bx2, bx3;

public:
  PICSimulation2D(CoordinateSystem coord_sys, ParticlePusher pusher) : PICSimulation{TWO_D, coord_sys, pusher} {};
  PICSimulation2D(CoordinateSystem coord_sys) : PICSimulation{TWO_D, coord_sys} {};
  ~PICSimulation2D() = default;
  void initialize() override;
  void finalize() override;
};

class PICSimulation3D : public PICSimulation {
protected:
  arrays::ThreeDArray<real_t> ex1, ex2, ex3;
  arrays::ThreeDArray<real_t> bx1, bx2, bx3;

public:
  PICSimulation3D(CoordinateSystem coord_sys, ParticlePusher pusher) : PICSimulation{THREE_D, coord_sys, pusher} {};
  PICSimulation3D(CoordinateSystem coord_sys) : PICSimulation{THREE_D, coord_sys} {};
  ~PICSimulation3D() = default;
  void initialize() override;
  void finalize() override;
};

} // namespace ntt

#endif

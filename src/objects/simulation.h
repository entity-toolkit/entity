#ifndef OBJECTS_SIMULATION_H
#define OBJECTS_SIMULATION_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"

#include <toml/toml.hpp>

#include <string>
#include <string_view>

namespace ntt {

// TODO!: there has to be a better way (e.g. `using`, or `typedef`)
// template <template <typename T = std::nullptr_t> class D>
template<Dimension D>
class Simulation {
protected:
  Dimension m_dim{D};

  SimulationParams m_sim_params;
  // ProblemGenerator m_pGen;

public:
  Simulation(const toml::value& inputdata);
  ~Simulation() = default;
  void setIO(std::string_view infname, std::string_view outdirname);
  void initialize();
  void verify();
  void printDetails();
  void finalize();

  void step_forward(const real_t&);
  void mainloop();

  virtual void faradayHalfsubstep(const real_t& time) { UNUSED(time); }
  virtual void depositSubstep(const real_t& time) { UNUSED(time); }
  virtual void ampereSubstep(const real_t& time) { UNUSED(time); }
  virtual void addCurrentsSubstep(const real_t& time) { UNUSED(time); }
  virtual void resetCurrentsSubstep(const real_t& time) { UNUSED(time); }

  virtual void fieldBoundaryConditions(const real_t& time) { UNUSED(time); }

  [[nodiscard]] auto get_params() const -> const SimulationParams& { return m_sim_params; }
};

class Simulation1D : public Simulation<ONE_D> {
  Meshblock1D m_meshblock;
public:
  Simulation1D(const toml::value& inputdata);
  void fieldBoundaryConditions(const real_t& time) override;
  void faradayHalfsubstep(const real_t& time) override;
  void depositSubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  [[nodiscard]] auto get_meshblock() const -> const Meshblock1D& { return m_meshblock; }
};

class Simulation2D : public Simulation<TWO_D> {
  Meshblock2D m_meshblock;
public:
  Simulation2D(const toml::value& inputdata);
  void fieldBoundaryConditions(const real_t& time) override;
  void faradayHalfsubstep(const real_t& time) override;
  void depositSubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  [[nodiscard]] auto get_meshblock() const -> const Meshblock2D& { return m_meshblock; }
};

class Simulation3D : public Simulation<THREE_D> {
  Meshblock3D m_meshblock;
public:
  Simulation3D(const toml::value& inputdata);
  void fieldBoundaryConditions(const real_t& time) override;
  void faradayHalfsubstep(const real_t& time) override;
  void depositSubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  [[nodiscard]] auto get_meshblock() const -> const Meshblock3D& { return m_meshblock; }
};

} // namespace ntt

#endif

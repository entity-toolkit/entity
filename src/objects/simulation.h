#ifndef OBJECTS_SIMULATION_H
#define OBJECTS_SIMULATION_H

#include "global.h"
#include "sim_params.h"
#include "meshblock.h"
#include "pgen.h"

#include <toml/toml.hpp>

#include <string>
#include <string_view>

namespace ntt {

template <Dimension D>
class Simulation {
protected:
  Dimension m_dim{D};

  SimulationParams m_sim_params;
  ProblemGenerator<D> m_pGen;
  Meshblock<D> m_meshblock;

public:
  Simulation(const toml::value& inputdata);
  ~Simulation() = default;
  void setIO(std::string_view infname, std::string_view outdirname);
  void userInitialize();
  void verify();
  void printDetails();
  void finalize();

  void step_forward(const real_t&);
  void mainloop();
  void run(std::string_view, std::string_view);

  // fields
  virtual void faradayHalfsubstep(const real_t&) {}
  virtual void ampereSubstep(const real_t&) {}
  virtual void addCurrentsSubstep(const real_t&) {}
  virtual void resetCurrentsSubstep(const real_t&) {}
  // particles
  virtual void pushParticlesSubstep(const real_t&) {}
  virtual void depositSubstep(const real_t&) {}
  // boundaries
  virtual void fieldBoundaryConditions(const real_t&) {}
  virtual void particleBoundaryConditions(const real_t&) {}

  [[nodiscard]] auto get_params() const -> const SimulationParams& { return m_sim_params; }
  [[nodiscard]] auto get_meshblock() const -> const Meshblock<D>& { return m_meshblock; }
};

struct Simulation1D : public Simulation<ONE_D> {
  Simulation1D(const toml::value& inputdata) : Simulation<ONE_D>{inputdata} {}

  void faradayHalfsubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  void pushParticlesSubstep(const real_t&) override;
  void depositSubstep(const real_t& time) override;

  void fieldBoundaryConditions(const real_t& time) override;
  void particleBoundaryConditions(const real_t& time) override;
};

struct Simulation2D : public Simulation<TWO_D> {
  Simulation2D(const toml::value& inputdata) : Simulation<TWO_D>{inputdata} {}

  void faradayHalfsubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  void pushParticlesSubstep(const real_t&) override;
  void depositSubstep(const real_t& time) override;

  void fieldBoundaryConditions(const real_t& time) override;
  void particleBoundaryConditions(const real_t& time) override;
};

struct Simulation3D : public Simulation<THREE_D> {
  Simulation3D(const toml::value& inputdata) : Simulation<THREE_D>{inputdata} {}

  void faradayHalfsubstep(const real_t& time) override;
  void ampereSubstep(const real_t& time) override;
  void addCurrentsSubstep(const real_t& time) override;
  void resetCurrentsSubstep(const real_t& time) override;

  void pushParticlesSubstep(const real_t&) override;
  void depositSubstep(const real_t& time) override;

  void fieldBoundaryConditions(const real_t& time) override;
  void particleBoundaryConditions(const real_t& time) override;
};

} // namespace ntt

#endif

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
    Dimension m_dim {D};

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

    void step(const real_t&, const short&);
    void mainloop();
    void run(std::string_view, std::string_view);

    // fields
    void faradaySubstep(const real_t&, const real_t&);
    void ampereSubstep(const real_t&, const real_t&);
    void addCurrentsSubstep(const real_t&);
    void resetCurrentsSubstep(const real_t&);
    // particles
    void pushParticlesSubstep(const real_t&);
    void depositSubstep(const real_t&);
    // boundaries
    void fieldBoundaryConditions(const real_t&);
    void particleBoundaryConditions(const real_t&);

    [[nodiscard]] auto get_params() const -> const SimulationParams& { return m_sim_params; }
    [[nodiscard]] auto get_meshblock() const -> const Meshblock<D>& { return m_meshblock; }
  };

} // namespace ntt

#endif

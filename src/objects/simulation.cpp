#include "global.h"
#include "simulation.h"
#include "sim_params.h"
#include "meshblock.h"

#include <plog/Log.h>

#include <stdexcept>

namespace ntt {

template <template <typename T> class D>
Simulation<D>::Simulation(const toml::value& inputdata)
    : m_dim{},
      m_sim_params{inputdata, m_dim.dim},
      m_meshblock{m_sim_params.m_resolution, m_sim_params.m_species},
      m_pGen{m_sim_params} {
  // TODO: meshblock extent can be different from global one
  m_meshblock.set_extent(m_sim_params.m_extent);
  m_meshblock.set_coord_system(m_sim_params.m_coord_system);
}

template <template <typename T> class D>
void Simulation<D>::initialize() {
  m_pGen.userInitFields(m_sim_params, m_meshblock);
  fieldBoundaryConditions(0.0);
  PLOGD << "Simulation initialized.";
}

template <template <typename T> class D>
void Simulation<D>::verify() {
  if (m_sim_params.m_simtype == UNDEFINED_SIM) {
    throw std::logic_error("ERROR: simulation type unspecified.");
  }
  if (m_sim_params.m_coord_system == UNDEFINED_COORD) {
    throw std::logic_error("ERROR: coordinate system unspecified.");
  }
  for (auto& b : m_sim_params.m_boundaries) {
    if (b == UNDEFINED_BC) { throw std::logic_error("ERROR: boundary conditions unspecified."); }
  }
  if (m_meshblock.m_coord_system == CARTESIAN_COORD) {
    // uniform cartesian grid
    if ((m_dim.dim == 2) && (m_meshblock.get_dx1() != m_meshblock.get_dx2())) {
      throw std::logic_error("ERROR: unequal cell size on a cartesian grid.");
    } else if ((m_dim.dim == 3)
               && ((m_meshblock.get_dx1() != m_meshblock.get_dx2())
                   || (m_meshblock.get_dx2() != m_meshblock.get_dx3()))) {
      throw std::logic_error("ERROR: unequal cell size on a cartesian grid.");
    }
    if (m_meshblock.get_dx1() * 0.5 <= m_sim_params.m_timestep) {
      throw std::logic_error("ERROR: timestep is too large (CFL not satisfied).");
    }
  } else {
    throw std::logic_error("ERROR: only cartesian coordinate system is available.");
  }
  for (auto& p : m_meshblock.particles) {
    if (p.get_pusher() == UNDEFINED_PUSHER) {
      throw std::logic_error("ERROR: undefined particle pusher.");
    }
  }
  // TODO: maybe some other tests
  PLOGD << "Simulation prerun check passed.";
}
template <template <typename T> class D>
void Simulation<D>::setIO(std::string_view infname, std::string_view outdirname) {
  m_sim_params.m_outputpath = outdirname;
  m_sim_params.m_inputfilename = infname;
}

template <template <typename T> class D>
void Simulation<D>::printDetails() {
  PLOGI << "[Simulation details]";
  PLOGI << "   title: " << m_sim_params.m_title;
  PLOGI << "   type: " << stringifySimulationType(m_sim_params.m_simtype);
  PLOGI << "   total runtime: " << m_sim_params.m_runtime;
  PLOGI << "   dt: " << m_sim_params.m_timestep << " ["
        << static_cast<int>(m_sim_params.m_runtime / m_sim_params.m_timestep) << " steps]";

  PLOGI << "[domain]";
  PLOGI << "   dimension: " << m_dim.dim << "D";
  PLOGI << "   coordinate system: "
        << stringifyCoordinateSystem(m_sim_params.m_coord_system, m_dim.dim);

  std::string bc{"   boundary conditions: { "};
  for (auto& b : m_sim_params.m_boundaries) {
    bc += stringifyBoundaryCondition(b) + " x ";
  }
  bc.erase(bc.size() - 3);
  bc += " }";
  PLOGI << bc;

  std::string res{"   resolution: { "};
  for (auto& r : m_sim_params.m_resolution) {
    res += std::to_string(r) + " x ";
  }
  res.erase(res.size() - 3);
  res += " }";
  PLOGI << res;

  std::string ext{"   extent: "};
  for (std::size_t i{0}; i < m_sim_params.m_extent.size(); i += 2) {
    ext += "{" + std::to_string(m_sim_params.m_extent[i]) + ", "
         + std::to_string(m_sim_params.m_extent[i + 1]) + "} ";
  }
  PLOGI << ext;

  std::string cell{"   cell size: "};
  real_t effective_dx{0.0};
  if (m_sim_params.m_coord_system == CARTESIAN_COORD) {
    cell += "{" + std::to_string(m_meshblock.get_dx1()) + "}";
    effective_dx = m_meshblock.get_dx1();
  }
  PLOGI << cell;

  PLOGI << "[fiducial parameters]";
  PLOGI << "   ppc0: " << m_sim_params.m_ppc0;
  PLOGI << "   rho0: " << m_sim_params.m_larmor0 << " [" << m_sim_params.m_larmor0 / effective_dx
        << " dx]";
  PLOGI << "   c_omp0: " << m_sim_params.m_skindepth0 << " ["
        << m_sim_params.m_skindepth0 / effective_dx << " dx]";
  PLOGI << "   sigma0: " << m_sim_params.m_sigma0;
  PLOGI << "   q0: " << m_sim_params.m_charge0;
  PLOGI << "   B0: " << m_sim_params.m_B0;

  PLOGI << "[particles]";
  for (std::size_t i{0}; i < m_meshblock.particles.size(); ++i) {
    PLOGI << "   [species #" << i + 1 << "]";
    PLOGI << "      label: " << m_meshblock.particles[i].get_label();
    PLOGI << "      mass: " << m_meshblock.particles[i].get_mass();
    PLOGI << "      charge: " << m_meshblock.particles[i].get_charge();
    PLOGI << "      pusher: " << stringifyParticlePusher(m_meshblock.particles[i].get_pusher());
    PLOGI << "      maxnpart: " << m_meshblock.particles[i].get_maxnpart() << " ("
          << m_meshblock.particles[i].get_npart() << ")";
  }
}

template <template <typename T> class D>
void Simulation<D>::finalize() {
  PLOGD << "Simulation finalized.";
}

template class ntt::Simulation<ntt::One_D>;
template class ntt::Simulation<ntt::Two_D>;
template class ntt::Simulation<ntt::Three_D>;

} // namespace ntt

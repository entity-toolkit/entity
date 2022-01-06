#include "global.h"
#include "simulation.h"

#include "faraday_cartesian.hpp"
#include "ampere_cartesian.hpp"

#include "faraday_curvilinear.hpp"
#include "ampere_curvilinear.hpp"
#include "ampere_ax_poles.hpp"

#include "add_currents.hpp"

#include <plog/Log.h>

namespace ntt {

  // # # # # # # # # # # # # # # # #
  // solve dB/dt
  // # # # # # # # # # # # # # # # #
  template <>
  void Simulation<ONE_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "1D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("faraday", mblock.loopActiveCells(), FaradayCartesian<ONE_D>(mblock, coeff / dx));
    } else {
      throw std::logic_error("# Error: wrong coordinate system for 1D.");
    }
  }

  template <>
  void Simulation<TWO_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "2D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("faraday", mblock.loopActiveCells(), FaradayCartesian<TWO_D>(mblock, coeff / dx));
    } else if ((m_sim_params.m_coord_system == "spherical") || (m_sim_params.m_coord_system == "qspherical")) {
      Kokkos::parallel_for(
          "faraday",
          mblock.loopActiveCells(),
          FaradayCurvilinear<TWO_D>(mblock, coeff));
    } else {
      throw std::logic_error("# Error: 2D faraday for the coordinate system not implemented.");
    }
  }

  template <>
  void Simulation<THREE_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "3D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("faraday", mblock.loopActiveCells(), FaradayCartesian<THREE_D>(mblock, coeff / dx));
    } else {
      throw std::logic_error("# Error: 3D faraday for the coordinate system not implemented.");
    }
  }

  // # # # # # # # # # # # # # # # #
  // solve dE/dt
  // # # # # # # # # # # # # # # # #
  template <>
  void Simulation<ONE_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "1D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("ampere", mblock.loopActiveCells(), AmpereCartesian<ONE_D>(mblock, coeff / dx));
    } else {
      throw std::logic_error("# Error: wrong coordinate system for 1D.");
    }
  }

  template <>
  void Simulation<TWO_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "2D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("ampere", mblock.loopActiveCells(), AmpereCartesian<TWO_D>(mblock, coeff / dx));
      } else if ((m_sim_params.m_coord_system == "spherical") || (m_sim_params.m_coord_system == "qspherical")) {
        Kokkos::parallel_for(
            "ampere",
            mblock.loopCells(0, 0, 1, 0),
            AmpereCurvilinear<TWO_D>(mblock, coeff));
        // evolve E1, E2 near polar axes
        Kokkos::parallel_for(
            "ampere_pole",
            NTT1DRange(mblock.i_min, mblock.i_max),
            AmpereAxisymmetricPoles<TWO_D>(mblock, coeff));
    } else {
      throw std::logic_error("# Error: 2D ampere for the coordinate system not implemented.");
    }
  }

  template <>
  void Simulation<THREE_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "3D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    if (mblock.grid->label == "cartesian") {
      const auto dx {(mblock.grid->x1_max - mblock.grid->x1_min) / (real_t)(mblock.grid->Nx1)};
      Kokkos::parallel_for("ampere", mblock.loopActiveCells(), AmpereCartesian<THREE_D>(mblock, coeff / dx));
    } else {
      throw std::logic_error("# Error: 3D ampere for the coordinate system not implemented.");
    }
  }

  // # # # # # # # # # # # # # # # #
  // add currents to E
  // # # # # # # # # # # # # # # # #
  template <Dimension D>
  void Simulation<D>::addCurrentsSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "D add current";
    Kokkos::parallel_for("addcurrs", mblock.loopActiveCells(), AddCurrents<D>(mblock));
  }

  // # # # # # # # # # # # # # # # #
  // reset currents to zero
  // # # # # # # # # # # # # # # # #
  template <Dimension D>
  void Simulation<D>::resetCurrentsSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "D reset current";
    Kokkos::deep_copy(mblock.j_fields, 0.0);
  }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;

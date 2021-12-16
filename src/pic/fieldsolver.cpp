#include "global.h"
#include "simulation.h"

#include "faraday.hpp"
#include "ampere.hpp"
#include "add_currents.hpp"

#include <plog/Log.h>

namespace ntt {

  // solve dB/dt
  template <>
  void Simulation<ONE_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "1D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    RangeND<ONE_D> range_faraday;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_faraday = m_meshblock.loopActiveCells();
    } else {
      throw std::logic_error("# Error: wrong coordinate system for 1D.");
    }
    Kokkos::parallel_for(
        "faraday",
        range_faraday,
        Faraday<ONE_D>(m_meshblock, coeff_x1, ZERO, ZERO));
  }

  template <>
  void Simulation<TWO_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "2D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    const real_t coeff_x2 {coeff / m_meshblock.get_dx2()};
    RangeND<TWO_D> range_faraday;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_faraday = m_meshblock.loopActiveCells();
    } else if (m_sim_params.m_coord_system == "spherical") {
      range_faraday = m_meshblock.loopCells(1, 0, 1, 0);
    } else {
      throw std::logic_error("# Error: 2D faraday for the coordinate system not implemented.");
    }
    Kokkos::parallel_for(
        "faraday",
        range_faraday,
        Faraday<TWO_D>(m_meshblock, coeff_x1, coeff_x2, ZERO));
  }

  template <>
  void Simulation<THREE_D>::faradaySubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "3D faraday";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    const real_t coeff_x2 {coeff / m_meshblock.get_dx2()};
    const real_t coeff_x3 {coeff / m_meshblock.get_dx3()};
    RangeND<THREE_D> range_faraday;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_faraday = m_meshblock.loopActiveCells();
    } else {
      throw std::logic_error("# Error: 3D faraday for the coordinate system not implemented.");
    }
    Kokkos::parallel_for(
        "faraday",
        range_faraday,
        Faraday<THREE_D>(m_meshblock, coeff_x1, coeff_x2, coeff_x3));
  }

  // solve dE/dt
  template <>
  void Simulation<ONE_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "1D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    RangeND<ONE_D> range_ampere;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_ampere = m_meshblock.loopActiveCells();
    } else {
      throw std::logic_error("# Error: wrong coordinate system for 1D.");
    }
    Kokkos::parallel_for(
        "ampere",
        range_ampere,
        Ampere<ONE_D>(m_meshblock, coeff_x1, ZERO, ZERO));
  }

  template <>
  void Simulation<TWO_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "2D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    const real_t coeff_x2 {coeff / m_meshblock.get_dx2()};
    RangeND<TWO_D> range_ampere;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_ampere = m_meshblock.loopActiveCells();
    } else if (m_sim_params.m_coord_system == "spherical") {
      range_ampere = m_meshblock.loopCells(1, 0, 1, 0);
    } else {
      throw std::logic_error("# Error: 2D ampere for the coordinate system not implemented.");
    }
    Kokkos::parallel_for(
        "ampere",
        range_ampere,
        Ampere<TWO_D>(m_meshblock, coeff_x1, coeff_x2, ZERO));
  }

  template <>
  void Simulation<THREE_D>::ampereSubstep(const real_t& time, const real_t& fraction) {
    UNUSED(time);
    PLOGD << "3D ampere";
    const real_t coeff {fraction * m_sim_params.m_correction * m_sim_params.m_timestep};
    const real_t coeff_x1 {coeff / m_meshblock.get_dx1()};
    const real_t coeff_x2 {coeff / m_meshblock.get_dx2()};
    const real_t coeff_x3 {coeff / m_meshblock.get_dx3()};
    RangeND<THREE_D> range_ampere;
    if (m_sim_params.m_coord_system == "cartesian") {
      range_ampere = m_meshblock.loopActiveCells();
    } else {
      throw std::logic_error("# Error: 3D ampere for the coordinate system not implemented.");
    }
    Kokkos::parallel_for(
        "ampere",
        range_ampere,
        Ampere<THREE_D>(m_meshblock, coeff_x1, coeff_x2, coeff_x3));
  }

  // add currents to E
  template <Dimension D>
  void Simulation<D>::addCurrentsSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "D add current";
    Kokkos::parallel_for("addcurrs", m_meshblock.loopActiveCells(), AddCurrents<D>(m_meshblock));
  }

  // reset currents to zero
  template <Dimension D>
  void Simulation<D>::resetCurrentsSubstep(const real_t& time) {
    UNUSED(time);
    PLOGD << D << "D reset current";
    Kokkos::deep_copy(m_meshblock.j_fields, 0.0);
  }

} // namespace ntt

template class ntt::Simulation<ntt::ONE_D>;
template class ntt::Simulation<ntt::TWO_D>;
template class ntt::Simulation<ntt::THREE_D>;

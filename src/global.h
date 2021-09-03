#ifndef GLOBAL_H
#define GLOBAL_H

#include <cstddef>
#include <string_view>

#define UNUSED(x) (void)(x)

namespace ntt {
#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

inline constexpr std::size_t N_GHOSTS{2};

enum SimulationType { UNDEFINED_SIM, PIC_SIM, FORCE_FREE_SIM, MHD_SIM };
enum Dimension { UNDEFINED_D, ONE_D, TWO_D, THREE_D };
enum CoordinateSystem {
  UNDEFINED_COORD,
  CARTESIAN_COORD,
  POLAR_COORD,
  SPHERICAL_COORD,
  LOG_SPHERICAL_COORD,
  CUSTOM_COORD
};
enum BoundaryCondition { UNDEFINED_BC, PERIODIC_BC, OPEN_BC };

enum ParticlePusher { UNDEFINED_PUSHER, BORIS_PUSHER, VAY_PUSHER };

auto stringifySimulationType(SimulationType sim) -> std::string_view;
auto stringifyDimension(Dimension dim) -> std::string_view;
auto stringifyCoordinateSystem(CoordinateSystem coord) -> std::string_view;
auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string_view;

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string_view;

// defaults
constexpr std::string_view DEF_input_filename{"input"};
constexpr std::string_view DEF_output_path{"output"};
} // namespace ntt

#endif

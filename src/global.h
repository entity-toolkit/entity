#ifndef GLOBAL_H
#define GLOBAL_H

#include <cstddef>
#include <string_view>

namespace ntt {
#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

inline constexpr std::size_t N_GHOSTS{2};

enum SimulationType { PIC_SIM, FORCE_FREE_SIM, MHD_SIM };
enum Dimension { ONE_D, TWO_D, THREE_D };
enum CoordinateSystem { CARTESIAN_COORD, POLAR_COORD, SPHERICAL_COORD, LOG_SPHERICAL_COORD, CUSTOM_COORD };

enum ParticlePusher { BORIS_PUSHER, VAY_PUSHER };

// defaults
constexpr std::string_view DEF_input_filename{"input"};
constexpr std::string_view DEF_output_path{"output"};
} // namespace ntt

#endif

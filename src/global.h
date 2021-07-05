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

enum Dimension { ONE_D, TWO_D, THREE_D };
enum CoordinateSystem { CARTESIAN, POLAR, SPHERICAL, LOG_SPHERICAL, CUSTOM };

enum ParticlePusher { BORIS_PUSHER, VAY_PUSHER };

// defaults
constexpr std::string_view DEF_input_filename{"input"};
} // namespace ntt

#endif

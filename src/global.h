#ifndef GLOBAL_H
#define GLOBAL_H

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string_view>

#define UNUSED(x) (void)(x)

#define KL KOKKOS_LAMBDA
#define HostMemSpace Kokkos::HostSpace
#define AccelMemSpace Kokkos::HostSpace

namespace ntt {

#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

using index_t = const std::size_t;

template<typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

inline constexpr std::size_t N_GHOSTS{2};

enum SimulationType { UNDEFINED_SIM, PIC_SIM, FORCE_FREE_SIM, MHD_SIM };

template<typename T>
struct One_D {
  using ndtype_t = T*;
};

template<typename T>
struct Two_D {
  using ndtype_t = T**;
};

template<typename T>
struct Three_D {
  using ndtype_t = T***;
};

enum CoordinateSystem {
  UNDEFINED_COORD,
  CARTESIAN_COORD,
  POLAR_COORD,
  SPHERICAL_COORD,
  LOG_SPHERICAL_COORD,
  CUSTOM_COORD
};
enum BoundaryCondition { UNDEFINED_BC, PERIODIC_BC, OPEN_BC };

enum ParticlePusher { UNDEFINED_PUSHER, BORIS_PUSHER, VAY_PUSHER, PHOTON_PUSHER };

auto stringifySimulationType(SimulationType sim) -> std::string_view;
auto stringifyCoordinateSystem(CoordinateSystem coord) -> std::string_view;
auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string_view;

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string_view;

// defaults
constexpr std::string_view DEF_input_filename{"input"};
constexpr std::string_view DEF_output_path{"output"};

}

#endif

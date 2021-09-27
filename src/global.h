#ifndef GLOBAL_H
#define GLOBAL_H

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>

#define UNUSED(x) (void)(x)

#define Lambda KOKKOS_LAMBDA

#define HostExeSpace Kokkos::OpenMP
#define HostMemSpace Kokkos::HostSpace

#ifndef DGPUACCELERATED
#  define AccelExeSpace Kokkos::OpenMP
#  define AccelMemSpace Kokkos::HostSpace
#else
#  define AccelExeSpace Kokkos::Cuda
#  define AccelMemSpace Kokkos::CudaSpace
#endif

namespace ntt {

#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

using index_t = const std::size_t;

template<typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

using NTTRange = Kokkos::RangePolicy<AccelExeSpace>;

template<typename T>
struct One_D {
  short dim {1};
  using ndtype_t = T*;
};

template<typename T>
struct Two_D {
  short dim {2};
  using ndtype_t = T**;
};

template<typename T>
struct Three_D {
  short dim {3};
  using ndtype_t = T***;
};

inline constexpr std::size_t N_GHOSTS{2};
enum SimulationType { UNDEFINED_SIM, PIC_SIM, FORCE_FREE_SIM, MHD_SIM };

enum CoordinateSystem {
  UNDEFINED_COORD,
  CARTESIAN_COORD,
  POLAR_R_THETA_COORD,
  POLAR_R_PHI_COORD,
  SPHERICAL_COORD,
  LOG_SPHERICAL_COORD,
  CUSTOM_COORD
};
enum BoundaryCondition { UNDEFINED_BC, PERIODIC_BC, OPEN_BC };

enum ParticlePusher { UNDEFINED_PUSHER, BORIS_PUSHER, VAY_PUSHER, PHOTON_PUSHER };

auto stringifySimulationType(SimulationType sim) -> std::string;
auto stringifyCoordinateSystem(CoordinateSystem coord) -> std::string;
auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string;

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string;

// defaults
constexpr std::string_view DEF_input_filename{"input"};
constexpr std::string_view DEF_output_path{"output"};

}

#endif

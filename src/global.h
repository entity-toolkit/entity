#ifndef GLOBAL_H
#define GLOBAL_H

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>

#define UNUSED(x) (void)(x)

#define Lambda KOKKOS_LAMBDA
#define Inline KOKKOS_INLINE_FUNCTION

#if !defined(GPUENABLED) && defined(OMPENABLED)
#  define AccelExeSpace Kokkos::OpenMP
#  define AccelMemSpace Kokkos::HostSpace
#elif defined(GPUENABLED)
#  define AccelExeSpace Kokkos::Cuda
#  define AccelMemSpace Kokkos::CudaSpace
#else
#  define AccelExeSpace Kokkos::Serial
#  define AccelMemSpace Kokkos::HostSpace
#endif

#define HostMemSpace Kokkos::HostSpace
#if defined (OMPENABLED)
#  define HostExeSpace Kokkos::OpenMP
#else
#  define HostExeSpace Kokkos::Serial
#endif

namespace ntt {

#ifdef SINGLE_PRECISION
using real_t = float;
#else
using real_t = double;
#endif

template<typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;
using NTT1DRange = Kokkos::RangePolicy<AccelExeSpace>;
using NTT2DRange = Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>;
using NTT3DRange = Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>;

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
auto stringifyCoordinateSystem(CoordinateSystem coord, short dim) -> std::string;
auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string;

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string;

// defaults
constexpr std::string_view DEF_input_filename{"input"};
constexpr std::string_view DEF_output_path{"output"};

}

#endif

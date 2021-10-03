#ifndef GLOBAL_H
#define GLOBAL_H

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <cstddef>
#include <string>
#include <iomanip>

#define UNUSED(x) (void)(x)

#define Lambda    KOKKOS_LAMBDA
#define Inline    KOKKOS_INLINE_FUNCTION

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
#if defined(OMPENABLED)
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

using range_t = Kokkos::RangePolicy<AccelExeSpace>::member_type;

template <typename T>
using NTTArray = Kokkos::View<T, AccelMemSpace>;

using ntt_1drange_t = Kokkos::RangePolicy<AccelExeSpace>;
using ntt_2drange_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>;
using ntt_3drange_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>;

auto NTT1DRange(const std::vector<long int>&) -> ntt_1drange_t;
auto NTT1DRange(const long int&, const long int&) -> ntt_1drange_t;
auto NTT2DRange(const std::vector<long int>&, const std::vector<long int>&) -> ntt_2drange_t;
auto NTT3DRange(const std::vector<long int>&, const std::vector<long int>&) -> ntt_3drange_t;

template <typename T>
struct One_D {
  short dim{1};
  using ndtype_t = T*;
};

template <typename T>
struct Two_D {
  short dim{2};
  using ndtype_t = T**;
};

template <typename T>
struct Three_D {
  short dim{3};
  using ndtype_t = T***;
};

inline constexpr int N_GHOSTS{2};
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

} // namespace ntt

namespace plog {
class NTTFormatter {
public:
  static auto header() -> util::nstring;
  static auto format(const Record& record) -> util::nstring;
};
} // namespace plog

#endif

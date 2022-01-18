#ifndef GLOBAL_H
#define GLOBAL_H

#include "definitions.h"
#include "defaults.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <vector>
#include <cstddef>
#include <string>
#include <iomanip>

// Kokkos-specific flags
#define Lambda    KOKKOS_LAMBDA
#define Inline    KOKKOS_INLINE_FUNCTION

// Defining Kokkos execution/memory space aliases
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
  // Defining specific code configurations as enum classes
  enum class Dimension { ONE_D = 1, TWO_D = 2, THREE_D = 3 };
  enum class SimulationType { UNDEFINED, PIC, GRPIC, FORCE_FREE, MHD };
  enum class BoundaryCondition { UNDEFINED, PERIODIC, USER, OPEN, COMM };
  enum class ParticlePusher { UNDEFINED, BORIS, VAY, PHOTON };
  
  // Defining stringify functions for enum classes
  auto stringifySimulationType(SimulationType sim) -> std::string;
  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string;
  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string;

  // ND coordinate alias
  template <Dimension D>
  using coord_t = real_t[static_cast<short>(D)];

  // ND vector alias
  template <Dimension D>
  using vec_t = real_t[static_cast<short>(D)];

  // Number of ghost zones to be used (compile-time enforced)
  inline constexpr int N_GHOSTS {2};

  // Defining an array alias of arbitrary type
  template <typename T>
  using NTTArray = Kokkos::View<T, AccelMemSpace>;

  // Defining Kokkos-specific range aliases
  using range_t = Kokkos::RangePolicy<AccelExeSpace>::member_type;

  // D x N dimensional array for storing fields on ND hypercubes
  template <Dimension D, int N>
  using RealFieldND = typename std::conditional<
    D == Dimension::ONE_D,
    NTTArray<real_t* [N]>,
    typename std::conditional<
      D == Dimension::TWO_D,
      NTTArray<real_t** [N]>,
      typename std::conditional<
        D == Dimension::THREE_D, 
        NTTArray<real_t*** [N]>, 
        std::nullptr_t>::type>::type>::type;
  // Defining aliases for `ntt_*drange_t`
  template <Dimension D>
  using RangeND = typename std::conditional<
    D == Dimension::ONE_D,
    Kokkos::RangePolicy<AccelExeSpace>,
    typename std::conditional<
      D == Dimension::TWO_D,
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>,
      typename std::conditional<
        D == Dimension::THREE_D,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>,
        std::nullptr_t>::type>::type>::type;

  /**
   * Function template for generating ND Kokkos range policy.
   *
   * @tparam D Dimension
   * @param i1 array of size D `int`: { min }.
   * @param i2 array of size D `int`: { max }.
   * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the accelerator execution space.
   */
  template <Dimension D>
  auto NTTRange(const int (&i1)[static_cast<short>(D)], const int (&i2)[static_cast<short>(D)])
    -> RangeND<D>;
} // namespace ntt

namespace plog {
  class NTTFormatter {
  public:
    static auto header() -> util::nstring;
    static auto format(const Record& record) -> util::nstring;
  };
} // namespace plog

#endif

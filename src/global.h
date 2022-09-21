#ifndef GLOBAL_H
#define GLOBAL_H

#include "definitions.h"
#include "defaults.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <type_traits>
#include <vector>
#include <cstddef>
#include <string>
#include <iomanip>

// Kokkos-specific flags
#define Lambda KOKKOS_LAMBDA
#define Inline KOKKOS_INLINE_FUNCTION

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

#if defined(GPUENABLED)
#  define NTTError(msg) ({})
#else
#  define NTTError(msg)                                                                       \
    throw std::runtime_error("# ERROR: " msg " : filename: " __FILE__ " : "                   \
                             "line: " LINE_STRING)
#endif

namespace math = Kokkos;

namespace ntt {
  // Defining specific code configurations as enum classes
  enum class Dimension { ONE_D = 1, TWO_D = 2, THREE_D = 3 };
  enum class SimulationType { UNDEFINED, PIC, GRPIC, FORCE_FREE, MHD };
  enum class BoundaryCondition { UNDEFINED, PERIODIC, USER, OPEN, COMM };
  enum class ParticlePusher { UNDEFINED, BORIS, VAY, PHOTON };

  inline constexpr auto Dim1      = Dimension::ONE_D;
  inline constexpr auto Dim2      = Dimension::TWO_D;
  inline constexpr auto Dim3      = Dimension::THREE_D;
  inline constexpr auto TypePIC   = SimulationType::PIC;
  inline constexpr auto TypeGRPIC = SimulationType::GRPIC;

  // Defining stringify functions for enum classes
  auto stringifySimulationType(SimulationType sim) -> std::string;
  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string;
  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string;

  // Number of ghost zones to be used (compile-time enforced)
  inline constexpr int N_GHOSTS = 2;

  /* -------------------------------------------------------------------------- */
  /*                                Type aliases                                */
  /* -------------------------------------------------------------------------- */
  // ND coordinate alias
  template <typename T, Dimension D>
  using tuple_t = T[static_cast<short>(D)];

  // ND coordinate alias
  template <Dimension D>
  using coord_t = tuple_t<real_t, D>;

  // ND vector alias
  template <Dimension D>
  using vec_t = tuple_t<real_t, D>;

  using index_t = const std::size_t;

  // Defining an array alias of arbitrary type
  template <typename T>
  using array_t = Kokkos::View<T, AccelMemSpace>;

  // Defining a scatter view alias of arbitrary type
  template <typename T>
  using scatter_array_t = Kokkos::Experimental::ScatterView<T>;

  // D x N dimensional array for storing fields on ND hypercubes
  template <Dimension D, int N>
  using ndfield_t = typename std::conditional<
    D == Dim1,
    array_t<real_t* [N]>,
    typename std::conditional<
      D == Dim2,
      array_t<real_t** [N]>,
      typename std::conditional<D == Dim3, array_t<real_t*** [N]>, std::nullptr_t>::type>::
      type>::type;

  // D x N dimensional scatter array for storing fields on ND hypercubes
  template <Dimension D, int N>
  using scatter_ndfield_t = typename std::conditional<
    D == Dim1,
    scatter_array_t<real_t* [N]>,
    typename std::conditional<D == Dim2,
                              scatter_array_t<real_t** [N]>,
                              typename std::conditional<D == Dim3,
                                                        scatter_array_t<real_t*** [N]>,
                                                        std::nullptr_t>::type>::type>::type;

  // Defining aliases for `RangePolicy` and `MDRangePolicy`
  template <Dimension D>
  using range_t = typename std::conditional<
    D == Dim1,
    Kokkos::RangePolicy<AccelExeSpace>,
    typename std::conditional<
      D == Dim2,
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>,
      typename std::conditional<D == Dim3,
                                Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>,
                                std::nullptr_t>::type>::type>::type;

  /**
   * @brief Function template for generating ND Kokkos range policy.
   * @overload
   * @tparam D Dimension
   * @param i1 array of size D `int`: { min }.
   * @param i2 array of size D `int`: { max }.
   * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the accelerator execution space.
   */
  template <Dimension D>
  auto CreateRangePolicy(const tuple_t<int, D>&, const tuple_t<int, D>&) -> range_t<D>;

  /**
   * @brief Synchronize CPU/GPU before advancing.
   */
  void WaitAndSynchronize();

} // namespace ntt

namespace plog {
  /**
   * @brief Formatter for logging messages.
   */
  class NTTFormatter {
  public:
    static auto header() -> util::nstring;
    static auto format(const Record& record) -> util::nstring;
  };
} // namespace plog

#endif

#ifndef GLOBAL_H
#define GLOBAL_H

// !TODO: this is a bad practice... wait until CUDA issue is fixed
#pragma nv_diag_suppress 20011

#include "config.h"

#include "definitions.h"
#include "directions.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_Sort.hpp>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

#include <cstddef>
#include <iomanip>
#include <string>
#include <vector>

#include <type_traits>

#define ClassLambda KOKKOS_CLASS_LAMBDA
#define Lambda      KOKKOS_LAMBDA
#define Function    KOKKOS_FUNCTION
#define Inline      KOKKOS_INLINE_FUNCTION

namespace math = Kokkos;

namespace ntt {
  // Array alias of arbitrary type
  template <typename T>
  using array_t = Kokkos::View<T, AccelMemSpace>;

  template <short D>
  using ndarray_t = typename std::conditional<
    D == 1,
    array_t<real_t*>,
    typename std::conditional<
      D == 2,
      array_t<real_t**>,
      typename std::conditional<
        D == 3,
        array_t<real_t***>,
        typename std::conditional<D == 4, array_t<real_t****>, std::nullptr_t>::type>::type>::type>::type;

  // Array mirror alias of arbitrary type
  template <typename T>
  using array_mirror_t = typename array_t<T>::HostMirror;

  // Scatter view alias of arbitrary type
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
      typename std::conditional<D == Dim3, array_t<real_t*** [N]>, std::nullptr_t>::type>::type>::type;

  // D x N dimensional array (host memspace) for storing fields on ND hypercube
  template <Dimension D, int N>
  using ndfield_mirror_t = typename ndfield_t<D, N>::HostMirror;

  // D x N dimensional scatter array for storing fields on ND hypercubes
  template <Dimension D, int N>
  using scatter_ndfield_t = typename std::conditional<
    D == Dim1,
    scatter_array_t<real_t* [N]>,
    typename std::conditional<
      D == Dim2,
      scatter_array_t<real_t** [N]>,
      typename std::conditional<D == Dim3, scatter_array_t<real_t*** [N]>, std::nullptr_t>::type>::type>::type;

  // !TODO: this looks ugly, template it...

  // Defining aliases for `RangePolicy` and `MDRangePolicy` for the device space
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

  // Defining aliases for `RangePolicy` and `MDRangePolicy` for the host space
  template <Dimension D>
  using range_h_t = typename std::conditional<
    D == Dim1,
    Kokkos::RangePolicy<HostExeSpace>,
    typename std::conditional<
      D == Dim2,
      Kokkos::MDRangePolicy<Kokkos::Rank<2>, HostExeSpace>,
      typename std::conditional<D == Dim3,
                                Kokkos::MDRangePolicy<Kokkos::Rank<3>, HostExeSpace>,
                                std::nullptr_t>::type>::type>::type;

  // Random number pool/generator type alias
  using RandomNumberPool_t = Kokkos::Random_XorShift1024_Pool<AccelExeSpace>;
  using RandomGenerator_t  = typename RandomNumberPool_t::generator_type;

  template <typename T>
  Inline auto Random(RandomGenerator_t&) -> T;

  template <>
  Inline auto Random<float>(RandomGenerator_t& gen) -> float {
    return gen.frand();
  }

  template <>
  Inline auto Random<double>(RandomGenerator_t& gen) -> double {
    return gen.drand();
  }

  /**
   * @brief Function template for generating ND Kokkos range policy.
   * @overload
   * @tparam D Dimension
   * @param i1 array of size D `std::size_t`: { min }.
   * @param i2 array of size D `std::size_t`: { max }.
   * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the accelerator execution space.
   */
  template <Dimension D>
  auto CreateRangePolicy(const tuple_t<std::size_t, D>&,
                         const tuple_t<std::size_t, D>&) -> range_t<D>;

  /**
   * @brief Function template for generating ND Kokkos range policy on the host.
   * @overload
   * @tparam D Dimension
   * @param i1 array of size D `std::size_t`: { min }.
   * @param i2 array of size D `std::size_t`: { max }.
   * @returns Kokkos::RangePolicy or Kokkos::MDRangePolicy in the host execution space.
   */
  template <Dimension D>
  auto CreateRangePolicyOnHost(const tuple_t<std::size_t, D>&,
                               const tuple_t<std::size_t, D>&) -> range_h_t<D>;

  /**
   * @brief Synchronize CPU/GPU before advancing.
   * @param debug_only Only synchronize if `DEBUG` is defined (default: always sync).
   */
  void WaitAndSynchronize(bool debug_only = false);

  inline void GlobalInitialize(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
#if defined(MPI_ENABLED)
    MPI_Init(&argc, &argv);
#endif // MPI_ENABLED
  }

  inline void GlobalFinalize() {
#if defined(MPI_ENABLED)
    MPI_Finalize();
#endif // MPI_ENABLED
    Kokkos::finalize();
  }

} // namespace ntt

#endif
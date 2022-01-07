#ifndef GLOBAL_H
#define GLOBAL_H

#include "constants.h"

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
  inline constexpr float ONE {1.0f};
  inline constexpr float ZERO {0.0f};
  inline constexpr float HALF {0.5f};
#else
  using real_t = double;
  inline constexpr double ONE {1.0};
  inline constexpr double ZERO {0.0};
  inline constexpr double HALF {0.5};
#endif

#define SIGN(x)      (((x) < ZERO) ? -ONE : ONE)
#define HEAVISIDE(x) (((x) <= ZERO) ? ZERO : ONE)

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

  enum class Dimension { ONE_D = 1,
                         TWO_D = 2,
                         THREE_D = 3 };

  template <Dimension D, int N>
  using RealFieldND = typename std::conditional<D == Dimension::ONE_D, NTTArray<real_t* [N]>,
                        typename std::conditional<D == Dimension::TWO_D, NTTArray<real_t** [N]>, 
                          typename std::conditional<D == Dimension::THREE_D, NTTArray<real_t*** [N]>, 
                            std::nullptr_t>
                          ::type>
                        ::type>
                      ::type;

  template <Dimension D>
  using RangeND = typename std::conditional<D == Dimension::ONE_D, ntt_1drange_t, 
                    typename std::conditional<D == Dimension::TWO_D, ntt_2drange_t, 
                      typename std::conditional<D == Dimension::THREE_D, ntt_3drange_t, 
                        std::nullptr_t>
                      ::type>
                    ::type>
                  ::type;

  inline constexpr int N_GHOSTS {2};
  enum class SimulationType { UNDEFINED,
                              PIC,
                              FORCE_FREE,
                              MHD };

  enum class BoundaryCondition { UNDEFINED,
                                 PERIODIC,
                                 USER,
                                 OPEN };

  enum class ParticlePusher { UNDEFINED,
                              BORIS,
                              VAY,
                              PHOTON };

  auto stringifySimulationType(SimulationType sim) -> std::string;
  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string;
  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string;

  // defaults
  constexpr std::string_view DEF_input_filename {"input"};
  constexpr std::string_view DEF_output_path {"output"};

} // namespace ntt

namespace plog {
  class NTTFormatter {
  public:
    static auto header() -> util::nstring;
    static auto format(const Record& record) -> util::nstring;
  };
} // namespace plog

#endif

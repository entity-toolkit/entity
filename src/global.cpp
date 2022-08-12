#include "global.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
#include <cassert>
#include <string>
#include <iomanip>

namespace ntt {
  const auto Dim1 = Dimension::ONE_D;
  const auto Dim2 = Dimension::TWO_D;
  const auto Dim3 = Dimension::THREE_D;

  auto NTTWait() -> void { Kokkos::fence(); }

  auto stringifySimulationType(SimulationType sim) -> std::string {
    switch (sim) {
    case SimulationType::PIC:
      return "PIC";
    case SimulationType::GRPIC:
      return "GRPIC";
    case SimulationType::FORCE_FREE:
      return "FF";
    case SimulationType::MHD:
      return "MHD";
    default:
      return "N/A";
    }
  }

  auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string {
    switch (bc) {
    case BoundaryCondition::PERIODIC:
      return "Periodic";
    case BoundaryCondition::OPEN:
      return "Open";
    case BoundaryCondition::USER:
      return "User";
    case BoundaryCondition::COMM:
      return "Communicate";
    default:
      return "N/A";
    }
  }

  auto stringifyParticlePusher(ParticlePusher pusher) -> std::string {
    switch (pusher) {
    case ParticlePusher::BORIS:
      return "Boris";
    case ParticlePusher::VAY:
      return "Vay";
    case ParticlePusher::PHOTON:
      return "Photon";
    default:
      return "N/A";
    }
  }

  template <>
  auto NTTRange<Dim1>(const tuple_t<int, Dim1>& i1, const tuple_t<int, Dim1>& i2)
    -> RangeND<Dim1> {
    auto i1min = static_cast<range_t>(i1[0]);
    auto i1max = static_cast<range_t>(i2[0]);
    return Kokkos::RangePolicy<AccelExeSpace>(i1min, i1max);
  }

  template <>
  auto NTTRange<Dim2>(const tuple_t<int, Dim2>& i1, const tuple_t<int, Dim2>& i2)
    -> RangeND<Dim2> {
    auto i1min = static_cast<range_t>(i1[0]);
    auto i1max = static_cast<range_t>(i2[0]);
    auto i2min = static_cast<range_t>(i1[1]);
    auto i2max = static_cast<range_t>(i2[1]);
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>({i1min, i2min},
                                                                 {i1max, i2max});
  }

  template <>
  auto NTTRange<Dim3>(const tuple_t<int, Dim3>& i1, const tuple_t<int, Dim3>& i2)
    -> RangeND<Dim3> {
    auto i1min = static_cast<range_t>(i1[0]);
    auto i1max = static_cast<range_t>(i2[0]);
    auto i2min = static_cast<range_t>(i1[1]);
    auto i2max = static_cast<range_t>(i2[1]);
    auto i3min = static_cast<range_t>(i1[2]);
    auto i3max = static_cast<range_t>(i2[2]);
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>({i1min, i2min, i3min},
                                                                 {i1max, i2max, i3max});
  }

} // namespace ntt

namespace plog {

  auto NTTFormatter::header() -> util::nstring { return util::nstring(); }
  auto NTTFormatter::format(const Record& record) -> util::nstring {
    util::nostringstream ss;
#ifdef DEBUG
    if (record.getSeverity() == plog::debug) {
      ss << PLOG_NSTR("\n") << record.getFunc() << PLOG_NSTR(" @ ") << record.getLine()
         << PLOG_NSTR("\n");
    }
#endif
    ss << std::setw(9) << std::left << severityToString(record.getSeverity())
       << PLOG_NSTR(": ");
    ss << record.getMessage() << PLOG_NSTR("\n");
    return ss.str();
  }

} // namespace plog

template ntt::RangeND<ntt::Dimension::ONE_D>
ntt::NTTRange<ntt::Dimension::ONE_D>(const ntt::tuple_t<int, ntt::Dimension::ONE_D>&,
                                     const ntt::tuple_t<int, ntt::Dimension::ONE_D>&);
template ntt::RangeND<ntt::Dimension::TWO_D>
ntt::NTTRange<ntt::Dimension::TWO_D>(const ntt::tuple_t<int, ntt::Dimension::TWO_D>&,
                                     const ntt::tuple_t<int, ntt::Dimension::TWO_D>&);
template ntt::RangeND<ntt::Dimension::THREE_D>
ntt::NTTRange<ntt::Dimension::THREE_D>(const ntt::tuple_t<int, ntt::Dimension::THREE_D>&,
                                       const ntt::tuple_t<int, ntt::Dimension::THREE_D>&);
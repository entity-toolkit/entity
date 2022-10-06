#include "global.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
#include <cassert>
#include <string>
#include <iomanip>

namespace ntt {
  auto WaitAndSynchronize() -> void { Kokkos::fence(); }

  auto stringifySimulationType(SimulationType sim) -> std::string {
    switch (sim) {
    case TypePIC:
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
  auto CreateRangePolicy<Dim1>(const tuple_t<int, Dim1>& i1, const tuple_t<int, Dim1>& i2)
    -> range_t<Dim1> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    return Kokkos::RangePolicy<AccelExeSpace>(i1min, i1max);
  }

  template <>
  auto CreateRangePolicy<Dim2>(const tuple_t<int, Dim2>& i1, const tuple_t<int, Dim2>& i2)
    -> range_t<Dim2> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>({i1min, i2min},
                                                                 {i1max, i2max});
  }

  template <>
  auto CreateRangePolicy<Dim3>(const tuple_t<int, Dim3>& i1, const tuple_t<int, Dim3>& i2)
    -> range_t<Dim3> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    index_t i3min = i1[2];
    index_t i3max = i2[2];
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

template ntt::range_t<ntt::Dim1>
ntt::CreateRangePolicy<ntt::Dim1>(const ntt::tuple_t<int, ntt::Dim1>&,
                                  const ntt::tuple_t<int, ntt::Dim1>&);
template ntt::range_t<ntt::Dim2>
ntt::CreateRangePolicy<ntt::Dim2>(const ntt::tuple_t<int, ntt::Dim2>&,
                                  const ntt::tuple_t<int, ntt::Dim2>&);
template ntt::range_t<ntt::Dim3>
ntt::CreateRangePolicy<ntt::Dim3>(const ntt::tuple_t<int, ntt::Dim3>&,
                                  const ntt::tuple_t<int, ntt::Dim3>&);
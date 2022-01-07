#include "global.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
#include <cassert>
#include <string>
#include <iomanip>

namespace ntt {

  auto stringifySimulationType(SimulationType sim) -> std::string {
    switch (sim) {
    case SimulationType::PIC:
      return "PIC";
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

  auto NTT1DRange(const std::vector<long int>& r1, const std::vector<long int>& r2) -> ntt_1drange_t {
    assert(r1.size() == 1);
    assert(r2.size() == 1);
    return Kokkos::RangePolicy<AccelExeSpace>(static_cast<range_t>(r1[0]), static_cast<range_t>(r2[0]));
  }
  auto NTT1DRange(const long int& r1, const long int& r2) -> ntt_1drange_t {
    return Kokkos::RangePolicy<AccelExeSpace>(static_cast<range_t>(r1), static_cast<range_t>(r2));
  }
  auto NTT2DRange(const std::vector<long int>& r1, const std::vector<long int>& r2) -> ntt_2drange_t {
    assert(r1.size() == 2);
    assert(r2.size() == 2);
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>(
      {static_cast<range_t>(r1[0]), static_cast<range_t>(r1[1])},
      {static_cast<range_t>(r2[0]), static_cast<range_t>(r2[1])});
  }
  auto NTT3DRange(const std::vector<long int>& r1, const std::vector<long int>& r2) -> ntt_3drange_t {
    assert(r1.size() == 3);
    assert(r2.size() == 3);
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
      {static_cast<range_t>(r1[0]), static_cast<range_t>(r1[1]), static_cast<range_t>(r1[2])},
      {static_cast<range_t>(r2[0]), static_cast<range_t>(r2[1]), static_cast<range_t>(r2[2])});
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
    ss << std::setw(9) << std::left << severityToString(record.getSeverity()) << PLOG_NSTR(": ");
    ss << record.getMessage() << PLOG_NSTR("\n");
    return ss.str();
  }

} // namespace plog

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
  case PIC_SIM:
    return "PIC";
  case FORCE_FREE_SIM:
    return "FF";
  case MHD_SIM:
    return "MHD";
  default:
    return "N/A";
  }
}

auto stringifyCoordinateSystem(CoordinateSystem coord) -> std::string {
  switch (coord) {
  case CARTESIAN_COORD:
    return "cartesian";
  case SPHERICAL_COORD:
    return "spherical";
  case CYLINDRICAL_COORD:
    return "cylindrical";
  case CARTESIAN_LIKE_COORD:
    return "cartesian-like";
  case SPHERICAL_LIKE_COORD:
    return "spherical-like";
  case CYLINDRICAL_LIKE_COORD:
    return "cylindrical-like";
  default:
    return "N/A";
  }
}

auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string {
  switch (bc) {
  case PERIODIC_BC:
    return "Periodic";
  case OPEN_BC:
    return "Open";
  default:
    return "N/A";
  }
}

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string {
  switch (pusher) {
  case BORIS_PUSHER:
    return "Boris";
  case VAY_PUSHER:
    return "Vay";
  case PHOTON_PUSHER:
    return "Photon";
  default:
    return "N/A";
  }
}

auto NTT1DRange(const std::vector<long int>& r1, const std::vector<long int>& r2) -> ntt_1drange_t {
  assert(r1.size() == 1);
  assert(r2.size() == 1);
  return Kokkos::RangePolicy<AccelExeSpace>(
      static_cast<range_t>(r1[0]), static_cast<range_t>(r2[0]));
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

auto getCoordinateSystem() -> CoordinateSystem {
# ifdef HARDCODE_FLAT_COORDS
  return CARTESIAN_COORD;
# elif HARDCODE_SPHERICAL_COORDS
  return SPHERICAL_COORD;
# elif HARDCODE_CYLINDRICAL_COORDS
  return CYLINDRICAL_COORD;
# elif HARDCODE_CARTESIAN_LIKE_COORDS
  return CARTESIAN_LIKE_COORD;
# elif HARDCODE_SPHERICAL_LIKE_COORDS
  return SPHERICAL_LIKE_COORD;
# elif HARDCODE_CYLINDRICAL_LIKE_COORDS
  return CYLINDRICAL_LIKE_COORD;
# else
  return UNDEFINED_COORD;
# endif
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

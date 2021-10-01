#include "global.h"

#include <plog/Log.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <cstddef>
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

auto stringifyCoordinateSystem(CoordinateSystem coord, short dim) -> std::string {
  switch (coord) {
  case CARTESIAN_COORD:
    return ((dim == 1) ? "X" : ((dim == 2) ? "XY" : "XYZ"));
  case POLAR_R_THETA_COORD:
    return "R_TH";
  case POLAR_R_PHI_COORD:
    return "R_PHI";
  case SPHERICAL_COORD:
    return "R_TH_PHI";
  case LOG_SPHERICAL_COORD:
    return "logR_TH_PHI";
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

} // namespace ntt

namespace plog {

util::nstring NTTFormatter::header() { return util::nstring(); }
util::nstring NTTFormatter::format(const Record& record) {
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

#include "global.h"

#include <string>

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

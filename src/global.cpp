#include "global.h"

#include <string_view>

auto stringifySimulationType(SimulationType sim) -> std::string_view {
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

auto stringifyDimension(Dimension dim) -> std::string_view {
  switch (dim) {
  case ONE_D:
    return "1D";
  case TWO_D:
    return "2D";
  case THREE_D:
    return "3D";
  default:
    return "N/A";
  }
}

auto stringifyCoordinateSystem(CoordinateSystem coord) -> std::string_view {
  switch (coord) {
  case CARTESIAN_COORD:
    return "XYZ";
  case POLAR_COORD:
    return "R_PHI";
  case SPHERICAL_COORD:
    return "R_TH_PHI";
  case LOG_SPHERICAL_COORD:
    return "logR_TH_PHI";
  default:
    return "N/A";
  }
}

auto stringifyBoundaryCondition(BoundaryCondition bc) -> std::string_view {
  switch (bc) {
  case PERIODIC_BC:
    return "Periodic";
  case OPEN_BC:
    return "Open";
  default:
    return "N/A";
  }
}

auto stringifyParticlePusher(ParticlePusher pusher) -> std::string_view {
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

#include "global.h"

#include <string_view>

namespace ntt {
std::string_view stringifySimulationType(SimulationType sim) {
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

std::string_view stringifyDimension(Dimension dim) {
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

std::string_view stringifyCoordinateSystem(CoordinateSystem coord) {
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

std::string_view stringifyParticlePusher(ParticlePusher pusher) {
  switch (pusher) {
    case BORIS_PUSHER:
      return "Boris";
    case VAY_PUSHER:
      return "Vay";
    default:
      return "N/A";
  }
}

}

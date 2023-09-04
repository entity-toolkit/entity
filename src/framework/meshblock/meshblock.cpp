#include "meshblock/meshblock.h"

#include "wrapper.h"

#include "particle_macros.h"
#include "sim_params.h"
#include "species.h"

#include "meshblock/particles.h"
#include "utils/qmath.h"

namespace ntt {
  template <Dimension D, SimulationEngine S>
  Meshblock<D, S>::Meshblock(const std::vector<unsigned int>&    res,
                             const std::vector<real_t>&          ext,
                             const real_t*                       params,
                             const std::vector<ParticleSpecies>& species) :
    Mesh<D>(res, ext, params),
    Fields<D, S>(res) {
    for (auto& part : species) {
      particles.emplace_back(part);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Meshblock<D, S>::Verify() {
    NTTHostErrorIf(std::isnan(minCellSize()), "Minimum cell size evaluated to NaN");

    // verifying that the correct particle arrays are allocated for a given dimension ...
    // ... and a given simulation engine
    for (auto& species : particles) {
      if constexpr (D == Dim1) {
        NTTHostErrorIf(
          (species.i2.extent(0) != 0) || (species.i3.extent(0) != 0) ||
            (species.dx2.extent(0) != 0) || (species.dx3.extent(0) != 0) ||
            (species.i2_prev.extent(0) != 0) || (species.i3_prev.extent(0) != 0) ||
            (species.dx2_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
          "Wrong particle arrays allocated for 1D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) ||
                           (species.dx1_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 1D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 1D mesh MINKOWSKI");
#endif
      } else if constexpr (D == Dim2) {
        NTTHostErrorIf(
          (species.i3.extent(0) != 0) || (species.dx3.extent(0) != 0) ||
            (species.i3_prev.extent(0) != 0) || (species.dx3_prev.extent(0) != 0),
          "Wrong particle arrays allocated for 2D mesh");
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) ||
                           (species.dx1_prev.extent(0) != 0) ||
                           (species.i2_prev.extent(0) != 0) ||
                           (species.dx2_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      } else {
        if constexpr (S == PICEngine) {
          NTTHostErrorIf((species.i1_prev.extent(0) != 0) ||
                           (species.dx1_prev.extent(0) != 0) ||
                           (species.i2_prev.extent(0) != 0) ||
                           (species.dx2_prev.extent(0) != 0) ||
                           (species.i3_prev.extent(0) != 0) ||
                           (species.dx3_prev.extent(0) != 0),
                         "Wrong particle arrays allocated for 2D mesh PIC");
        }
#ifdef MINKOWSKI_METRIC
        NTTHostErrorIf(species.phi.extent(0) != 0,
                       "Wrong particle arrays allocated for 2D mesh MINKOWSKI");
#endif
      }
    }
  }
} // namespace ntt
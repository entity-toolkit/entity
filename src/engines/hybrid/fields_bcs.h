/**
 * @file engines/hybrid/fields_bcs.h
 * @brief Non-periodic field boundary conditions for the HYBRID engine
 * @implements
 *   - ntt::hybrid::PerfectConductorFieldsIn<> -> void
 *   - ntt::hybrid::FieldBoundaries<>          -> void
 * @namespaces:
 *   - ntt::hybrid::
 *
 * PERIODIC boundaries are realized by the metadomain halo exchange
 * (Metadomain::CommunicateFields) and are skipped here. The reflecting-wall
 * (CONDUCTOR) condition is applied to the Yee-staggered evolved fields stored in
 * `em` — edge-E `Ee` in comps 0..2, face-B `Bf` in comps 3..5 — which is exactly
 * the layout the engine-agnostic `kernel::bc::ConductorBoundaries_kernel` expects
 * (E in 0..2, B in 3..5), so that kernel is reused verbatim.
 *
 * @note Hybrid is Cartesian Minkowski, so only the conductor wall is meaningful
 *       (no axis / GR-horizon boundaries).
 * @note LIMITATION: this only fixes the evolved `Bf`/`Ee` in `em`. The predictor
 *       face-B scratch `cur` (Bf*, Bf**) is 3-component and the half-step edge-E
 *       `Ee'`, `Ee''` live in `em0::345`, neither of which the SR conductor kernel
 *       can address; their wall ghosts, and the cell-centered `bckp` ghosts the
 *       pusher gathers from, are left to the halo exchange. For the persistent
 *       CT-evolved B that is sufficient; a hybrid-specific wall kernel for the
 *       intermediate buffers is a TODO.
 */

#ifndef ENGINES_HYBRID_FIELDS_BCS_H
#define ENGINES_HYBRID_FIELDS_BCS_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "arch/kokkos_aliases.h"
#include "utils/error.h"

#include "metrics/minkowski.h"

#include "framework/domain/domain.h"
#include "framework/domain/grid.h"
#include "kernels/fields_bcs.hpp"

#include <Kokkos_Core.hpp>

#include <vector>

namespace ntt {
  namespace hybrid {

    /**
     * @brief Apply the perfect-conductor (reflecting wall) condition on `em`
     *        (Ee in 0..2, Bf in 3..5) along one orthogonal direction.
     */
    template <Dimension D>
    void PerfectConductorFieldsIn(
      const dir::direction_t<D>&                       direction,
      Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
      BCTags                                           tags) {
      const auto sign = direction.get_sign();
      const auto dim  = direction.get_dim();

      std::vector<ncells_t> xi_min, xi_max;
      const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
      for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
        const auto dd = all_dirs[d];
        if (dim == dd) {
          xi_min.push_back(0);
          xi_max.push_back((sign < 0) ? (N_GHOSTS + 1) : N_GHOSTS);
        } else {
          xi_min.push_back(0);
          xi_max.push_back(domain.mesh.n_all(dd));
        }
      }

      range_t<D> range;
      if constexpr (D == Dim::_1D) {
        range = CreateRangePolicy<D>({ xi_min[0] }, { xi_max[0] });
      } else if constexpr (D == Dim::_2D) {
        range = CreateRangePolicy<D>({ xi_min[0], xi_min[1] },
                                     { xi_max[0], xi_max[1] });
      } else {
        range = CreateRangePolicy<D>({ xi_min[0], xi_min[1], xi_min[2] },
                                     { xi_max[0], xi_max[1], xi_max[2] });
      }
      const ncells_t i_edge = (sign > 0) ? domain.mesh.i_max(dim)
                                         : domain.mesh.i_min(dim);

      if (dim == in::x1) {
        if (sign > 0) {
          Kokkos::parallel_for(
            "ConductorFields",
            range,
            kernel::bc::ConductorBoundaries_kernel<D, in::x1, true>(
              domain.fields.em, i_edge, tags));
        } else {
          Kokkos::parallel_for(
            "ConductorFields",
            range,
            kernel::bc::ConductorBoundaries_kernel<D, in::x1, false>(
              domain.fields.em, i_edge, tags));
        }
      } else if (dim == in::x2) {
        if constexpr (D == Dim::_2D || D == Dim::_3D) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<D, in::x2, true>(
                domain.fields.em, i_edge, tags));
          } else {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<D, in::x2, false>(
                domain.fields.em, i_edge, tags));
          }
        } else {
          raise::Error("Invalid dimension for x2 conductor BC", HERE);
        }
      } else { // in::x3
        if constexpr (D == Dim::_3D) {
          if (sign > 0) {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<D, in::x3, true>(
                domain.fields.em, i_edge, tags));
          } else {
            Kokkos::parallel_for(
              "ConductorFields",
              range,
              kernel::bc::ConductorBoundaries_kernel<D, in::x3, false>(
                domain.fields.em, i_edge, tags));
          }
        } else {
          raise::Error("Invalid dimension for x3 conductor BC", HERE);
        }
      }
    }

    /**
     * @brief Apply non-periodic field boundaries to the evolved `em` fields.
     * @param global_grid the global grid (for the configured boundary type)
     * @param tags        which fields to act on (e.g. BC::B, BC::E, or BC::B|BC::E)
     * @note PERIODIC is skipped (handled by CommunicateFields). Only CONDUCTOR is
     *       implemented for hybrid; any other non-periodic type raises.
     */
    template <Dimension D>
    void FieldBoundaries(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>& global_grid,
                         BCTags         tags) {
      for (const auto& direction : dir::Directions<D>::orth) {
        const auto gbc = global_grid.flds_bc_in(direction);
        if (gbc == FldsBC::PERIODIC) {
          continue; // realized by the halo exchange
        } else if (gbc == FldsBC::CONDUCTOR) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CONDUCTOR) {
            PerfectConductorFieldsIn<D>(direction, domain, tags);
          }
        } else {
          raise::Error("hybrid engine: only PERIODIC and CONDUCTOR field "
                       "boundaries are implemented",
                       HERE);
        }
      }
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_FIELDS_BCS_H

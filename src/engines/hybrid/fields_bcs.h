/**
 * @file engines/hybrid/fields_bcs.h
 * @brief Non-periodic (reflecting-wall / perfect-conductor) boundary
 *        conditions for the HYBRID engine, x1 walls only.
 * @implements
 *   - ntt::hybrid::PerfectConductorFieldsIn<> -> void
 *   - ntt::hybrid::FieldBoundaries<>          -> void   (em: Ee and/or Bf)
 *   - ntt::hybrid::WallEPrime<>               -> void   (em0::345 = Ee'/Ee'')
 *   - ntt::hybrid::WallScratchB<>             -> void   (cur = Bf*, Bf**)
 *   - ntt::hybrid::WallBckpFill<>             -> void   (bckp = Ec'/Bc')
 *   - ntt::hybrid::MomentsWallBC<>            -> void   (aux = V, N)
 * @namespaces:
 *   - ntt::hybrid::
 *
 * PERIODIC boundaries are realized by the metadomain halo exchange
 * (Metadomain::CommunicateFields) and are skipped here.
 *
 * The conductor wall (kernels/hybrid/wall_bcs.hpp) enforces E_tan = 0 on the
 * wall plane; Faraday then freezes the wall-plane B_n automatically, which is
 * the correct condition for an OBLIQUE background field. The wall-plane B_n is
 * deliberately never overwritten (zeroing it is only valid for perpendicular
 * fields and plants a div(B) layer at the wall). B ghosts get an even
 * (zero-gradient) mirror so the Hall term sees no fake wall current.
 *
 * EVERY buffer whose wall ghosts (or wall-plane values) enter a stencil needs
 * explicit treatment -- there is no halo exchange at a physical wall, so
 * untreated ghosts keep their allocation value (zero) forever:
 *   - em  (Ee/Bf)         -> FieldBoundaries (BC::E and/or BC::B)
 *   - em0::345 (Ee'/Ee'') -> WallEPrime, before Faraday #2/#3
 *   - cur (Bf*, Bf**)     -> WallScratchB, before EMF #1/#2 (Hall stencil)
 *   - bckp (Ec'/Bc')      -> WallBckpFill, before the particle gathers
 *   - aux (V/N)           -> MomentsWallBC after each deposit: folds the
 *     ghost deposit tails back as the image-plasma contribution (V_x odd) and
 *     mirror-fills the ghosts for the EMF / filter stencils
 *
 * @note Hybrid is Cartesian Minkowski, so only the conductor wall is
 *       meaningful (no axis / GR-horizon boundaries).
 * @note Only x1 walls are implemented (the hybrid shock geometry); a
 *       CONDUCTOR boundary in x2/x3 raises.
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
#include "kernels/hybrid/wall_bcs.hpp"

#include <Kokkos_Core.hpp>

#include <vector>

namespace ntt {
  namespace hybrid {

    namespace wall {

      /**
       * @brief Range over the wall layer for x1-staggered (edge-E / face-B)
       *        kernels: i1 = 0 (wall plane) .. N_GHOSTS (ghost layers; the
       *        +x1 side has one node less), full transverse extent.
       */
      template <Dimension D>
      auto StaggeredRange(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                          short sign) -> range_t<D> {
        std::vector<ncells_t> xi_min, xi_max;
        xi_min.push_back(0);
        xi_max.push_back((sign < 0) ? (N_GHOSTS + 1) : N_GHOSTS);
        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
        for (auto d { 1u }; d < static_cast<unsigned>(D); ++d) {
          xi_min.push_back(0);
          xi_max.push_back(domain.mesh.n_all(all_dirs[d]));
        }
        if constexpr (D == Dim::_1D) {
          return CreateRangePolicy<D>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (D == Dim::_2D) {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1] },
                                      { xi_max[0], xi_max[1] });
        } else {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1], xi_min[2] },
                                      { xi_max[0], xi_max[1], xi_max[2] });
        }
      }

      /**
       * @brief Range over the wall layer for cell-centered kernels:
       *        i1 = ghost layer 0 .. N_GHOSTS; transverse extent is
       *        active-only when `active_transverse` (the moment fold; corner
       *        tails are already remapped by the transverse sync) and full
       *        otherwise.
       */
      template <Dimension D>
      auto CellRange(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                     bool active_transverse) -> range_t<D> {
        std::vector<ncells_t> xi_min, xi_max;
        xi_min.push_back(0);
        xi_max.push_back(N_GHOSTS);
        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
        for (auto d { 1u }; d < static_cast<unsigned>(D); ++d) {
          const auto dd = all_dirs[d];
          if (active_transverse) {
            xi_min.push_back(domain.mesh.i_min(dd));
            xi_max.push_back(domain.mesh.i_max(dd));
          } else {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd));
          }
        }
        if constexpr (D == Dim::_1D) {
          return CreateRangePolicy<D>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (D == Dim::_2D) {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1] },
                                      { xi_max[0], xi_max[1] });
        } else {
          return CreateRangePolicy<D>({ xi_min[0], xi_min[1], xi_min[2] },
                                      { xi_max[0], xi_max[1], xi_max[2] });
        }
      }

      /**
       * @brief Invoke `f(sign, i_edge)` for every x1 conductor wall this rank
       *        owns. Raises on non-x1 conductor walls (unimplemented).
       */
      template <Dimension D, typename F>
      void ForEachX1Wall(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                         const Grid<D>& global_grid,
                         F              f) {
        for (const auto& direction : dir::Directions<D>::orth) {
          const auto gbc = global_grid.flds_bc_in(direction);
          if (gbc == FldsBC::PERIODIC) {
            continue; // realized by the halo exchange
          } else if (gbc == FldsBC::CONDUCTOR) {
            if (domain.mesh.flds_bc_in(direction) != FldsBC::CONDUCTOR) {
              continue; // not an edge-owning rank
            }
            const auto dim  = direction.get_dim();
            const auto sign = direction.get_sign();
            raise::ErrorIf(dim != in::x1,
                           "hybrid engine: CONDUCTOR walls are only "
                           "implemented in x1",
                           HERE);
            const ncells_t i_edge = (sign > 0) ? domain.mesh.i_max(in::x1)
                                               : domain.mesh.i_min(in::x1);
            f(sign, i_edge);
          } else {
            raise::Error("hybrid engine: only PERIODIC and CONDUCTOR field "
                         "boundaries are implemented",
                         HERE);
          }
        }
      }

    } // namespace wall

    /**
     * @brief Apply the perfect-conductor wall condition on `em` along one
     *        orthogonal direction: BC::E -> Ee (comps 0..2): E_tan = 0 on the
     *        wall plane + mirror ghosts; BC::B -> Bf (comps 3..5): even mirror
     *        ghosts (the wall-plane B_n is left to Faraday, which freezes it
     *        because of E_tan = 0).
     */
    template <Dimension D>
    void PerfectConductorFieldsIn(
      const dir::direction_t<D>&                       direction,
      Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
      BCTags                                           tags) {
      const auto sign = direction.get_sign();
      raise::ErrorIf(direction.get_dim() != in::x1,
                     "hybrid engine: CONDUCTOR walls are only implemented in x1",
                     HERE);
      const ncells_t i_edge = (sign > 0) ? domain.mesh.i_max(in::x1)
                                         : domain.mesh.i_min(in::x1);
      const auto     range  = wall::StaggeredRange<D>(domain, sign);
      if (tags & BC::E) {
        if (sign > 0) {
          Kokkos::parallel_for(
            "WallEdgeE",
            range,
            kernel::hybrid::WallEdgeE_kernel<D, true>(domain.fields.em, i_edge, 0));
        } else {
          Kokkos::parallel_for(
            "WallEdgeE",
            range,
            kernel::hybrid::WallEdgeE_kernel<D, false>(domain.fields.em, i_edge, 0));
        }
      }
      if (tags & BC::B) {
        if (sign > 0) {
          Kokkos::parallel_for(
            "WallFaceB",
            range,
            kernel::hybrid::WallFaceB_kernel<D, true, 6>(domain.fields.em,
                                                         i_edge,
                                                         3));
        } else {
          Kokkos::parallel_for(
            "WallFaceB",
            range,
            kernel::hybrid::WallFaceB_kernel<D, false, 6>(domain.fields.em,
                                                          i_edge,
                                                          3));
        }
      }
    }

    /**
     * @brief Apply non-periodic field boundaries to the evolved `em` fields.
     * @param global_grid the global grid (for the configured boundary type)
     * @param tags        which fields to act on (BC::E, BC::B, or both)
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

    /**
     * @brief Conductor wall condition on the half-step edge-E (em0::345 =
     *        Ee'/Ee''). Call after EMF #1/#2, before the Faraday push that
     *        consumes it.
     */
    template <Dimension D>
    void WallEPrime(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                    const Grid<D>&                                   global_grid) {
      wall::ForEachX1Wall<D>(domain, global_grid, [&](short sign, ncells_t i_edge) {
        const auto range = wall::StaggeredRange<D>(domain, sign);
        if (sign > 0) {
          Kokkos::parallel_for(
            "WallEdgeE",
            range,
            kernel::hybrid::WallEdgeE_kernel<D, true>(domain.fields.em0, i_edge, 3));
        } else {
          Kokkos::parallel_for(
            "WallEdgeE",
            range,
            kernel::hybrid::WallEdgeE_kernel<D, false>(domain.fields.em0, i_edge, 3));
        }
      });
    }

    /**
     * @brief Even-mirror the wall ghosts of the predictor face-B scratch
     *        (cur = Bf*, Bf**). Call after each Faraday #1/#2 (+ halo
     *        exchange), before the EMF that reads it: the Hall stencil reads
     *        the wall ghosts, which no halo exchange fills (left at zero they
     *        fake a wall current sheet J ~ B/dx every substage).
     */
    template <Dimension D>
    void WallScratchB(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                      const Grid<D>& global_grid) {
      wall::ForEachX1Wall<D>(domain, global_grid, [&](short sign, ncells_t i_edge) {
        const auto range = wall::StaggeredRange<D>(domain, sign);
        if (sign > 0) {
          Kokkos::parallel_for(
            "WallFaceB",
            range,
            kernel::hybrid::WallFaceB_kernel<D, true, 3>(domain.fields.cur,
                                                         i_edge,
                                                         0));
        } else {
          Kokkos::parallel_for(
            "WallFaceB",
            range,
            kernel::hybrid::WallFaceB_kernel<D, false, 3>(domain.fields.cur,
                                                          i_edge,
                                                          0));
        }
      });
    }

    /**
     * @brief Conductor-sign mirror fill of the wall ghosts of the
     *        cell-centered gather fields (bckp: Ec' 0..2, Bc' 3..5). Call
     *        after each EMF #1/#2 (+ halo exchange), before the particle push
     *        that gathers from bckp.
     */
    template <Dimension D>
    void WallBckpFill(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                      const Grid<D>& global_grid) {
      wall::ForEachX1Wall<D>(domain, global_grid, [&](short sign, ncells_t i_edge) {
        const auto range = wall::CellRange<D>(domain, false);
        if (sign > 0) {
          Kokkos::parallel_for(
            "WallBckp",
            range,
            kernel::hybrid::WallBckp_kernel<D, true>(domain.fields.bckp, i_edge));
        } else {
          Kokkos::parallel_for(
            "WallBckp",
            range,
            kernel::hybrid::WallBckp_kernel<D, false>(domain.fields.bckp, i_edge));
        }
      });
    }

    /**
     * @brief Image-plasma wall treatment of the deposited moments (aux).
     * @param fold also fold the ghost deposit tails back into the active cells
     *        (V_x sign-flipped) -- exactly once per deposit, after
     *        SynchronizeFields(AUX) + CommunicateFields(AUX). With
     *        fold = false only the (idempotent) ghost mirror fill runs, e.g.
     *        to re-fill after a filter pass.
     */
    template <Dimension D>
    void MomentsWallBC(Domain<SimEngine::HYBRID, metric::Minkowski<D>>& domain,
                       const Grid<D>& global_grid,
                       bool           fold) {
      wall::ForEachX1Wall<D>(domain, global_grid, [&](short sign, ncells_t i_edge) {
        if (fold) {
          const auto fold_range = wall::CellRange<D>(domain, true);
          if (sign > 0) {
            Kokkos::parallel_for("WallMomentsFold",
                                 fold_range,
                                 kernel::hybrid::WallMoments_kernel<D, true, true>(
                                   domain.fields.aux,
                                   i_edge));
          } else {
            Kokkos::parallel_for("WallMomentsFold",
                                 fold_range,
                                 kernel::hybrid::WallMoments_kernel<D, false, true>(
                                   domain.fields.aux,
                                   i_edge));
          }
        }
        const auto fill_range = wall::CellRange<D>(domain, false);
        if (sign > 0) {
          Kokkos::parallel_for("WallMomentsFill",
                               fill_range,
                               kernel::hybrid::WallMoments_kernel<D, true, false>(
                                 domain.fields.aux,
                                 i_edge));
        } else {
          Kokkos::parallel_for("WallMomentsFill",
                               fill_range,
                               kernel::hybrid::WallMoments_kernel<D, false, false>(
                                 domain.fields.aux,
                                 i_edge));
        }
      });
    }

  } // namespace hybrid
} // namespace ntt

#endif // ENGINES_HYBRID_FIELDS_BCS_H

/**
 * @file framework/domain/mesh.h
 * @brief Grid and Mesh classes containing information about the geometry
 * @implements
 *   - ntt::Mesh<> : ntt::Grid<>
 * @namespaces:
 *   - ntt::
 * @note
 * Mesh extends the Grid adding information about the metric,
 * the physical extent, and the boundary conditions
 */

#ifndef FRAMEWORK_DOMAIN_MESH_H
#define FRAMEWORK_DOMAIN_MESH_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/domain/grid.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <MetricClass M>
  struct Mesh : public Grid<M::Dim> {
    static constexpr Dimension D { M::Dim };
    using base_t = Grid<D>;
    using base_t::extent;
    using base_t::m_extent;
    using base_t::m_flds_bc;
    using base_t::m_prtl_bc;

    M metric;

    Mesh(const std::vector<ncells_t>&         res,
         const boundaries_t<real_t>&          ext,
         const std::map<std::string, real_t>& metric_params)
      : Grid<D> { res, ext }
      , metric { res, ext, metric_params }
      , m_metric_params_raw { metric_params } {}

    Mesh(const std::vector<ncells_t>&         res,
         const boundaries_t<real_t>&          ext,
         const std::map<std::string, real_t>& metric_params,
         const boundaries_t<FldsBC>&          flds_bc,
         const boundaries_t<PrtlBC>&          prtl_bc)
      : Grid<D> { res, ext, flds_bc, prtl_bc }
      , metric { res, ext, metric_params }
      , m_metric_params_raw { metric_params } {}

    ~Mesh() = default;

    void set_extent(const boundaries_t<real_t>& new_extent) {
      m_extent = new_extent;
      metric.~M();
      new (&metric) M { this->m_resolution, new_extent, m_metric_params_raw };
    }

    void set_resolution_and_extent(const std::vector<ncells_t>& new_res,
                                   const boundaries_t<real_t>&  new_extent) {
      raise::ErrorIf(new_res.size() != D, "invalid resolution dim", HERE);
      this->m_resolution = new_res;
      m_extent           = new_extent;
      metric.~M();
      new (&metric) M { this->m_resolution, m_extent, m_metric_params_raw };
    }

    /**
     * @brief Get the intersection of the mesh with a box
     * @param box physical extent
     * @return the intersection of the mesh with the box
     * @note pass Range::All to select the entire dimension
     */
    [[nodiscard]]
    auto Intersection(const boundaries_t<real_t>& box) const
      -> boundaries_t<real_t> {
      raise::ErrorIf(box.size() != M::Dim, "Invalid box dimension", HERE);
      boundaries_t<real_t> intersection;
      auto                 d = 0;
      for (const auto& b : box) {
        if (b == Range::All) {
          intersection.push_back({ extent()[d].first, extent()[d].second });
        } else {
          real_t x_min, x_max;
          if (b.first == Range::Min) {
            x_min = extent()[d].first;
          } else {
            x_min = std::min(extent()[d].second,
                             std::max(extent()[d].first, b.first));
          }
          if (b.second == Range::Max) {
            x_max = extent()[d].second;
          } else {
            x_max = std::max(extent()[d].first,
                             std::min(extent()[d].second, b.second));
            intersection.emplace_back(x_min, x_max);
          }
        }
        ++d;
      }
      return intersection;
    }

    /**
     * @brief Check if the mesh intersects with a box
     * @param box physical extent
     * @return true if the mesh intersects with the box
     * @note pass Range::All to select the entire dimension
     */
    [[nodiscard]]
    auto Intersects(const boundaries_t<real_t>& box) const -> bool {
      raise::ErrorIf(box.size() != M::Dim, "Invalid box dimension", HERE);
      const auto intersection = Intersection(box);
      for (const auto& i : intersection) {
        if (i.first > i.second or cmp::AlmostEqual(i.first, i.second)) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Get the range of indices corresponding to a physical extent
     * @param box physical extent
     * @param incl_ghosts include ghost cells in the given direction
     * @return the range of indices corresponding to the physical extent
     * @note pass Range::All to select the entire dimension
     * @note min will be taken with a floor, and max with a ceil
     * @note if the box does not intersect with the mesh, the range will be all {0, 0}
     * @note indices are already shifted by N_GHOSTS (i.e. they start at N_GHOSTS not 0)
     */
    [[nodiscard]]
    auto ExtentToRange(const boundaries_t<real_t>& box,
                       const boundaries_t<bool>&   incl_ghosts) const
      -> boundaries_t<ncells_t> {
      raise::ErrorIf(box.size() != M::Dim, "Invalid box dimension", HERE);
      raise::ErrorIf(incl_ghosts.size() != M::Dim,
                     "Invalid incl_ghosts dimension",
                     HERE);
      boundaries_t<ncells_t> range;
      if (not Intersects(box)) {
        for (auto i { 0u }; i < box.size(); ++i) {
          range.emplace_back(0, 0);
        }
        return range;
      }
      auto d = 0;
      for (const auto& b : box) {
        if (b == Range::All) {
          range.push_back({ incl_ghosts[d].first ? 0 : N_GHOSTS,
                            incl_ghosts[d].second
                              ? this->n_all()[d]
                              : this->n_active()[d] + N_GHOSTS });
        } else {
          const auto xi_min = std::min(std::max(extent()[d].first, b.first),
                                       extent()[d].second);
          const auto xi_max = std::max(std::min(extent()[d].second, b.second),
                                       extent()[d].first);
          real_t     xi_min_Cd { ZERO }, xi_max_Cd { ZERO };
          if (d == 0) {
            xi_min_Cd = math::floor(
              metric.template convert<1, Crd::Ph, Crd::Cd>(xi_min));
            xi_max_Cd = math::ceil(
              metric.template convert<1, Crd::Ph, Crd::Cd>(xi_max));
          } else if (d == 1) {
            if constexpr (D == Dim::_2D or D == Dim::_3D) {
              xi_min_Cd = math::floor(
                metric.template convert<2, Crd::Ph, Crd::Cd>(xi_min));
              xi_max_Cd = math::ceil(
                metric.template convert<2, Crd::Ph, Crd::Cd>(xi_max));
            } else {
              raise::Error("invalid dimension", HERE);
            }
          } else if (d == 2) {
            if constexpr (D == Dim::_3D) {
              xi_min_Cd = math::floor(
                metric.template convert<3, Crd::Ph, Crd::Cd>(xi_min));
              xi_max_Cd = math::ceil(
                metric.template convert<3, Crd::Ph, Crd::Cd>(xi_max));
            } else {
              raise::Error("invalid dimension", HERE);
            }
          } else {
            raise::Error("invalid dimension", HERE);
            throw;
          }
          if (!incl_ghosts[d].first) {
            xi_min_Cd = std::max(xi_min_Cd, static_cast<real_t>(ZERO));
          }
          if (!incl_ghosts[d].second) {
            xi_max_Cd = std::min(xi_max_Cd,
                                 static_cast<real_t>(this->n_active()[d]));
          }
          range.emplace_back(static_cast<ncells_t>(xi_min_Cd) +
                               (incl_ghosts[d].first ? 0 : N_GHOSTS),
                             static_cast<ncells_t>(xi_max_Cd) +
                               (incl_ghosts[d].second ? 2 * N_GHOSTS : N_GHOSTS));
        }
        ++d;
      }

      return range;
    }

  private:
    std::map<std::string, real_t> m_metric_params_raw;
  };
} // namespace ntt

#endif // FRAMEWORK_DOMAIN_MESH_H

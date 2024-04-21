/**
 * @file framework/domain/mesh.h
 * @brief Grid and Mesh classes containing information about the geometry
 * @implements
 *   - ntt::Mesh<> : ntt::Grid<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/directions.h
 *   - utils/error.h
 *   - utils/numeric.h
 *   - framework/domain/grid.h
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

#include "arch/directions.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/domain/grid.h"

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <class M>
  struct Mesh : public Grid<M::Dim> {
    static_assert(M::is_metric, "template arg for Mesh class has to be a metric");
    static constexpr bool      is_mesh { true };
    static constexpr Dimension D { M::Dim };

    M metric;

    Mesh(const std::vector<std::size_t>&      res,
         const boundaries_t<real_t>&          ext,
         const std::map<std::string, real_t>& metric_params) :
      Grid<D> { res },
      metric { res, ext, metric_params },
      m_extent { ext } {}

    Mesh(const std::vector<std::size_t>&      res,
         const boundaries_t<real_t>&          ext,
         const std::map<std::string, real_t>& metric_params,
         const boundaries_t<FldsBC>&          flds_bc,
         const boundaries_t<PrtlBC>&          prtl_bc) :
      Grid<D> { res },
      metric { res, ext, metric_params },
      m_extent { ext } {
      for (auto d { 0 }; d < D; ++d) {
        dir::direction_t<D> dir_plus;
        dir_plus[d] = +1;
        dir::direction_t<D> dir_minus;
        dir_minus[d] = -1;
        set_flds_bc(dir_plus, flds_bc[d].second);
        set_flds_bc(dir_minus, flds_bc[d].first);
        set_prtl_bc(dir_plus, prtl_bc[d].second);
        set_prtl_bc(dir_minus, prtl_bc[d].first);
      }
    }

    ~Mesh() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto extent(in i) const -> std::pair<real_t, real_t> {
      switch (i) {
        case in::x1:
          return (m_extent.size() > 0) ? m_extent[0]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case in::x2:
          return (m_extent.size() > 1) ? m_extent[1]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case in::x3:
          return (m_extent.size() > 2) ? m_extent[2]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        default:
          raise::Error("invalid dimension", HERE);
          throw;
      }
    }

    [[nodiscard]]
    auto extent() const -> boundaries_t<real_t> {
      return m_extent;
    }

    [[nodiscard]]
    auto flds_bc() const -> boundaries_t<FldsBC> {
      if constexpr (D == Dim::_1D) {
        return {
          {flds_bc_in({ -1 }), flds_bc_in({ -1 })}
        };
      } else if constexpr (D == Dim::_2D) {
        return {
          {flds_bc_in({ -1, 0 }), flds_bc_in({ 1, 0 })},
          {flds_bc_in({ 0, -1 }), flds_bc_in({ 0, 1 })}
        };
      } else if constexpr (D == Dim::_3D) {
        return {
          {flds_bc_in({ -1, 0, 0 }), flds_bc_in({ 1, 0, 0 })},
          {flds_bc_in({ 0, -1, 0 }), flds_bc_in({ 0, 1, 0 })},
          {flds_bc_in({ 0, 0, -1 }), flds_bc_in({ 0, 0, 1 })}
        };
      } else {
        raise::Error("invalid dimension", HERE);
        throw;
      }
    }

    [[nodiscard]]
    auto prtl_bc() const -> boundaries_t<PrtlBC> {
      if constexpr (D == Dim::_1D) {
        return {
          {prtl_bc_in({ -1 }), prtl_bc_in({ -1 })}
        };
      } else if constexpr (D == Dim::_2D) {
        return {
          {prtl_bc_in({ -1, 0 }), prtl_bc_in({ 1, 0 })},
          {prtl_bc_in({ 0, -1 }), prtl_bc_in({ 0, 1 })}
        };
      } else if constexpr (D == Dim::_3D) {
        return {
          {prtl_bc_in({ -1, 0, 0 }), prtl_bc_in({ 1, 0, 0 })},
          {prtl_bc_in({ 0, -1, 0 }), prtl_bc_in({ 0, 1, 0 })},
          {prtl_bc_in({ 0, 0, -1 }), prtl_bc_in({ 0, 0, 1 })}
        };
      } else {
        raise::Error("invalid dimension", HERE);
        throw;
      }
    }

    [[nodiscard]]
    auto flds_bc_in(const dir::direction_t<D>& direction) const -> FldsBC {
      raise::ErrorIf(m_flds_bc.find(direction) == m_flds_bc.end(),
                     "direction not found",
                     HERE);
      return m_flds_bc.at(direction);
    }

    [[nodiscard]]
    auto prtl_bc_in(const dir::direction_t<D>& direction) const -> PrtlBC {
      raise::ErrorIf(m_prtl_bc.find(direction) == m_prtl_bc.end(),
                     "direction not found",
                     HERE);
      return m_prtl_bc.at(direction);
    }

    /* setters -------------------------------------------------------------- */
    inline void set_flds_bc(const dir::direction_t<D>& direction, const FldsBC& bc) {
      m_flds_bc.insert_or_assign(direction, bc);
    }

    inline void set_prtl_bc(const dir::direction_t<D>& direction, const PrtlBC& bc) {
      m_prtl_bc.insert_or_assign(direction, bc);
    }

  private:
    boundaries_t<real_t>  m_extent;
    dir::map_t<D, FldsBC> m_flds_bc;
    dir::map_t<D, PrtlBC> m_prtl_bc;
  };
} // namespace ntt

#endif // FRAMEWORK_DOMAIN_MESH_H
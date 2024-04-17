/**
 * @file framework/logistics/mesh.h
 * @brief Grid and Mesh classes containing information about the geometry
 * @implements
 *   - ntt::Mesh<> : ntt::Grid<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/directions.h
 *   - utils/error.h
 *   - utils/numeric.h
 *   - framework/logistics/grid.h
 * @namespaces:
 *   - ntt::
 * @note
 * Mesh extends the Grid adding information about the metric,
 * the physical extent, and the boundary conditions
 */

#ifndef FRAMEWORK_LOGISTICS_MESH_H
#define FRAMEWORK_LOGISTICS_MESH_H

#include "enums.h"
#include "global.h"

#include "arch/directions.h"
#include "utils/error.h"
#include "utils/numeric.h"

#include "framework/logistics/grid.h"

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

    ~Mesh() = default;

    /* getters -------------------------------------------------------------- */
    [[nodiscard]]
    auto extent(unsigned short i) const -> std::pair<real_t, real_t> {
      switch (i) {
        case 0:
          return (m_extent.size() > 0) ? m_extent[0]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case 1:
          return (m_extent.size() > 1) ? m_extent[1]
                                       : std::pair<real_t, real_t> { ZERO, ZERO };
        case 2:
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
    auto flds_bc_in(const dir::direction_t<D>& direction) const -> FldsBC {
      return m_flds_bc.at(direction);
    }

    [[nodiscard]]
    auto prtl_bc_in(const dir::direction_t<D>& direction) const -> PrtlBC {
      return m_prtl_bc.at(direction);
    }

    /* setters -------------------------------------------------------------- */
    inline void setFldsBc(const dir::direction_t<D>& direction, const FldsBC& bc) {
      m_flds_bc.insert_or_assign(direction, bc);
    }

    inline void setPrtlBc(const dir::direction_t<D>& direction, const PrtlBC& bc) {
      m_prtl_bc.insert_or_assign(direction, bc);
    }

  private:
    boundaries_t<real_t>  m_extent;
    dir::map_t<D, FldsBC> m_flds_bc;
    dir::map_t<D, PrtlBC> m_prtl_bc;
  };
} // namespace ntt

#endif // FRAMEWORK_LOGISTICS_MESH_H
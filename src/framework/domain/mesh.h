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
      m_extent { ext } {
      for (unsigned short d = 0; d < D; ++d) {
        m_flds_bc.push_back({ FldsBC::INVALID, FldsBC::INVALID });
        m_prtl_bc.push_back({ PrtlBC::INVALID, PrtlBC::INVALID });
      }
    }

    Mesh(const std::vector<std::size_t>&      res,
         const boundaries_t<real_t>&          ext,
         const std::map<std::string, real_t>& metric_params,
         const boundaries_t<FldsBC>&          flds_bc,
         const boundaries_t<PrtlBC>&          prtl_bc) :
      Grid<D> { res },
      metric { res, ext, metric_params },
      m_extent { ext },
      m_flds_bc { flds_bc },
      m_prtl_bc { prtl_bc } {}

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
      return m_flds_bc;
    }

    [[nodiscard]]
    auto prtl_bc() const -> boundaries_t<PrtlBC> {
      return m_prtl_bc;
    }

    [[nodiscard]]
    auto flds_bc_in(const dir::direction_t<D>& direction) const -> FldsBC {
      unsigned short nonzero = 0;
      FldsBC         bc      = FldsBC::INVALID;
      for (unsigned short d = 0; d < D; ++d) {
        const auto dir = direction[d];
        if (dir == 0) {
          continue;
        } else if (dir == 1) {
          bc = m_flds_bc[d].second;
          ++nonzero;
        } else if (dir == -1) {
          bc = m_flds_bc[d].first;
          ++nonzero;
        }
      }
      raise::ErrorIf(nonzero != 1, "invalid direction", HERE);
      return bc;
    }

    [[nodiscard]]
    auto prtl_bc_in(const dir::direction_t<D>& direction) const -> PrtlBC {
      unsigned short nonzero = 0;
      PrtlBC         bc      = PrtlBC::INVALID;
      for (unsigned short d = 0; d < D; ++d) {
        const auto dir = direction[d];
        if (dir == 0) {
          continue;
        } else if (dir == 1) {
          bc = m_prtl_bc[d].second;
          ++nonzero;
        } else if (dir == -1) {
          bc = m_prtl_bc[d].first;
          ++nonzero;
        }
      }
      raise::ErrorIf(nonzero != 1, "invalid direction", HERE);
      return bc;
    }

    /* setters -------------------------------------------------------------- */
    inline void setFldsBc(const dir::direction_t<D>& direction, const FldsBC& bc) {
      unsigned short nonzero = 0;
      for (unsigned short d = 0; d < D; ++d) {
        const auto dir = direction[d];
        if (dir == 0) {
          continue;
        } else if (dir == 1) {
          m_flds_bc[d].second = bc;
          ++nonzero;
        } else if (dir == -1) {
          m_flds_bc[d].first = bc;
          ++nonzero;
        }
      }
      raise::ErrorIf(nonzero != 1, "invalid direction", HERE);
    }

    inline void setPrtlBc(const dir::direction_t<D>& direction, const PrtlBC& bc) {
      unsigned short nonzero = 0;
      for (unsigned short d = 0; d < D; ++d) {
        const auto dir = direction[d];
        if (dir == 0) {
          continue;
        } else if (dir == 1) {
          m_prtl_bc[d].second = bc;
          ++nonzero;
        } else if (dir == -1) {
          m_prtl_bc[d].first = bc;
          ++nonzero;
        }
      }
      raise::ErrorIf(nonzero != 1, "invalid direction", HERE);
    }

  private:
    boundaries_t<real_t> m_extent;
    boundaries_t<FldsBC> m_flds_bc;
    boundaries_t<PrtlBC> m_prtl_bc;
  };
} // namespace ntt

#endif // FRAMEWORK_DOMAIN_MESH_H
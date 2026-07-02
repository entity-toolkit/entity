/**
 * @file output/render/reduce.hpp
 * @brief Small dimension-generic cell reductions used by the in-situ renderer
 * @implements
 *   - kernel::RenderMagnitude3_kernel<D, N>
 *   - kernel::RenderPickComp_kernel<D, N>
 *   - kernel::RenderDivideComp_kernel<D, N>
 *   - kernel::RenderVmagByRho_kernel<D, N>
 * @namespaces:
 *   - kernel::
 * @macros:
 *   - OUTPUT_ENABLED
 * @note
 * Each functor provides 1D/2D/3D operator() overloads so the same object works
 * with `mesh.rangeActiveCells()` of any dimension (the range policy selects the
 * matching arity). They reduce the prepared (interpolated, synced) `bckp`
 * scratch field down to the single scalar component the ray-march / slice
 * kernel samples, so the field-grammar dispatch stays dimension-agnostic.
 */

#ifndef OUTPUT_RENDER_REDUCE_HPP
#define OUTPUT_RENDER_REDUCE_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"

#include <cstdint>

namespace kernel {
  using namespace ntt;

  /**
   * @brief F(.., co) = sqrt(F(.., c0)^2 + F(.., c1)^2 + F(.., c2)^2)
   */
  template <Dimension D, std::uint8_t N>
  class RenderMagnitude3_kernel {
    ndfield_t<D, N>     F;
    const std::uint8_t  c0, c1, c2, co;

  public:
    RenderMagnitude3_kernel(const ndfield_t<D, N>& f,
                            std::uint8_t           a,
                            std::uint8_t           b,
                            std::uint8_t           c,
                            std::uint8_t           o)
      : F { f }
      , c0 { a }
      , c1 { b }
      , c2 { c }
      , co { o } {}

    Inline void operator()(cellidx_t i1) const {
      const real_t v0 = F(i1, c0), v1 = F(i1, c1), v2 = F(i1, c2);
      F(i1, co) = math::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      const real_t v0 = F(i1, i2, c0), v1 = F(i1, i2, c1), v2 = F(i1, i2, c2);
      F(i1, i2, co) = math::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      const real_t v0 = F(i1, i2, i3, c0), v1 = F(i1, i2, i3, c1),
                   v2 = F(i1, i2, i3, c2);
      F(i1, i2, i3, co) = math::sqrt(v0 * v0 + v1 * v1 + v2 * v2);
    }
  };

  /**
   * @brief F(.., co) = F(.., ci)  (move one component into the render slot)
   */
  template <Dimension D, std::uint8_t N>
  class RenderPickComp_kernel {
    ndfield_t<D, N>    F;
    const std::uint8_t ci, co;

  public:
    RenderPickComp_kernel(const ndfield_t<D, N>& f, std::uint8_t i, std::uint8_t o)
      : F { f }
      , ci { i }
      , co { o } {}

    Inline void operator()(cellidx_t i1) const {
      F(i1, co) = F(i1, ci);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      F(i1, i2, co) = F(i1, i2, ci);
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      F(i1, i2, i3, co) = F(i1, i2, i3, ci);
    }
  };

  /**
   * @brief F(.., cnum) = (F(.., cden) != 0) ? F(.., cnum) / F(.., cden) : 0
   */
  template <Dimension D, std::uint8_t N>
  class RenderDivideComp_kernel {
    ndfield_t<D, N>    F;
    const std::uint8_t cnum, cden;

  public:
    RenderDivideComp_kernel(const ndfield_t<D, N>& f,
                            std::uint8_t           num,
                            std::uint8_t           den)
      : F { f }
      , cnum { num }
      , cden { den } {}

    Inline void operator()(cellidx_t i1) const {
      const real_t d = F(i1, cden);
      F(i1, cnum)    = (d != ZERO) ? (F(i1, cnum) / d) : ZERO;
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      const real_t d  = F(i1, i2, cden);
      F(i1, i2, cnum) = (d != ZERO) ? (F(i1, i2, cnum) / d) : ZERO;
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      const real_t d      = F(i1, i2, i3, cden);
      F(i1, i2, i3, cnum) = (d != ZERO) ? (F(i1, i2, i3, cnum) / d) : ZERO;
    }
  };

  /**
   * @brief F(.., co) = | (F(c0), F(c1), F(c2)) / F(crho) |, else 0.
   * @note SR bulk-speed magnitude: the three mass-weighted flux components are
   * each normalized by Rho before the Euclidean norm, in one pass.
   */
  template <Dimension D, std::uint8_t N>
  class RenderVmagByRho_kernel {
    ndfield_t<D, N>    F;
    const std::uint8_t c0, c1, c2, crho, co;

  public:
    RenderVmagByRho_kernel(const ndfield_t<D, N>& f,
                           std::uint8_t           a,
                           std::uint8_t           b,
                           std::uint8_t           c,
                           std::uint8_t           rho,
                           std::uint8_t           o)
      : F { f }
      , c0 { a }
      , c1 { b }
      , c2 { c }
      , crho { rho }
      , co { o } {}

    Inline auto mag(real_t v0, real_t v1, real_t v2, real_t rho) const -> real_t {
      if (rho == ZERO) {
        return ZERO;
      }
      const real_t a = v0 / rho, b = v1 / rho, c = v2 / rho;
      return math::sqrt(a * a + b * b + c * c);
    }

    Inline void operator()(cellidx_t i1) const {
      F(i1, co) = mag(F(i1, c0), F(i1, c1), F(i1, c2), F(i1, crho));
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      F(i1, i2, co) = mag(F(i1, i2, c0), F(i1, i2, c1), F(i1, i2, c2),
                          F(i1, i2, crho));
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      F(i1, i2, i3, co) = mag(F(i1, i2, i3, c0), F(i1, i2, i3, c1),
                              F(i1, i2, i3, c2), F(i1, i2, i3, crho));
    }
  };

} // namespace kernel

#endif // OUTPUT_RENDER_REDUCE_HPP

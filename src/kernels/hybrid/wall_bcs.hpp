/**
 * @file kernels/hybrid/wall_bcs.hpp
 * @brief Reflecting-wall (perfect-conductor) boundary kernels for the HYBRID
 *        engine, x1 walls only.
 * @implements
 *   - kernel::hybrid::WallEdgeE_kernel<>  -> edge-E: E_tan = 0 on the wall
 *     plane + odd/even mirror into the ghosts
 *   - kernel::hybrid::WallFaceB_kernel<>  -> face-B: even (zero-gradient)
 *     mirror into the ghosts; the wall-plane B_n is NOT touched
 *   - kernel::hybrid::WallMoments_kernel<> -> cell-centered moments (aux):
 *     additive fold of the ghost deposit tails (image plasma) + mirror fill
 *   - kernel::hybrid::WallBckp_kernel<>   -> cell-centered Ec/Bc gather
 *     fields: conductor-sign mirror fill of the ghosts
 * @namespaces:
 *   - kernel::hybrid::
 *
 * Conventions (P = true is the +x1 wall):
 *   - i_edge is the wall-plane NODE index: i_min(x1) for -x1, i_max(x1) for
 *     +x1. Cell i is bounded by nodes i and i+1, so the ghost cells are
 *     i_edge-1-k (minus) / i_edge+k (plus), k = 0..N_GHOSTS-1.
 *   - Edge-E staggering in x1: comp +0 (E_x) is x1-cell-centered, comps +1/+2
 *     (E_y/E_z) sit on x1 nodes -- at i1 == i_edge they lie ON the wall plane.
 *   - Face-B staggering in x1: comp +0 (B_x) sits on x1 nodes, comps +1/+2 are
 *     x1-cell-centered.
 *
 * Physics:
 *   - Perfect conductor: E_tan = 0 on the wall plane. Faraday then freezes the
 *     wall-plane B_n automatically (dB_n/dt = -(curl E)_n involves only
 *     in-plane E_tan derivatives), which is the correct condition for an
 *     OBLIQUE background field (B_n stays at its initial value). The wall-plane
 *     B_n must therefore never be overwritten -- in particular NOT zeroed,
 *     which is only valid when the initial B_n is zero (perpendicular shocks)
 *     and otherwise plants a div(B) monopole layer at the wall.
 *   - B ghosts are filled with an even (zero-gradient) mirror so the Hall term
 *     (curl B) x B sees no fake wall current sheet. Without this, scratch
 *     buffers whose wall ghosts stay zero (e.g. Bf*, Bf** in `cur`) produce a
 *     spurious J ~ B/dx at the wall every substage.
 *   - Moments: a reflecting wall is realized by image plasma (u_x -> -u_x).
 *     The shape-function tails a near-wall particle deposits into the ghost
 *     cells are exactly the image particle's contribution to the active cells:
 *     fold them back with V_x sign-flipped (V_y, V_z, N unchanged), then fill
 *     the ghosts with the same-sign mirror of the active cells for the EMF /
 *     filter stencils. Without the fold the wall-cell density is undercounted
 *     and 1/N in Ohm's law is overestimated.
 *   - bckp (cell-centered Ec/Bc the pusher gathers): conductor-sign mirror,
 *     E_x even, E_y/E_z odd (-> E_tan ~ 0 at the wall), B even. Without it the
 *     gather mixes in zero ghosts and near-wall particles feel ~50% fields.
 */

#ifndef KERNELS_HYBRID_WALL_BCS_HPP
#define KERNELS_HYBRID_WALL_BCS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"

namespace kernel::hybrid {
  using namespace ntt;

  /**
   * @brief E_tan = 0 on the wall plane + conductor mirror into the ghosts.
   * @tparam D dimension
   * @tparam P true for the +x1 wall, false for the -x1 wall
   * Launch range: i1 in [0, N_GHOSTS + 1) for -x1, [0, N_GHOSTS) for +x1
   * (i1 == 0 is the wall plane, i1 >= 1 the ghost layers); other indices span
   * the full transverse extent.
   */
  template <Dimension D, bool P>
  struct WallEdgeE_kernel {
    ndfield_t<D, 6> Fld;
    const ncells_t  i_edge;
    const uint8_t   c0;

    WallEdgeE_kernel(ndfield_t<D, 6>& Fld, ncells_t i_edge, uint8_t c0)
      : Fld { Fld }
      , i_edge { i_edge }
      , c0 { c0 } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (i1 == 0) {
          Fld(i_edge, c0 + 1) = ZERO;
          Fld(i_edge, c0 + 2) = ZERO;
        } else {
          if constexpr (not P) {
            Fld(i_edge - i1, c0 + 0) = Fld(i_edge + i1 - 1, c0 + 0);
            Fld(i_edge - i1, c0 + 1) = -Fld(i_edge + i1, c0 + 1);
            Fld(i_edge - i1, c0 + 2) = -Fld(i_edge + i1, c0 + 2);
          } else {
            Fld(i_edge + i1 - 1, c0 + 0) = Fld(i_edge - i1, c0 + 0);
            Fld(i_edge + i1, c0 + 1)     = -Fld(i_edge - i1, c0 + 1);
            Fld(i_edge + i1, c0 + 2)     = -Fld(i_edge - i1, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallEdgeE_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (i1 == 0) {
          Fld(i_edge, i2, c0 + 1) = ZERO;
          Fld(i_edge, i2, c0 + 2) = ZERO;
        } else {
          if constexpr (not P) {
            Fld(i_edge - i1, i2, c0 + 0) = Fld(i_edge + i1 - 1, i2, c0 + 0);
            Fld(i_edge - i1, i2, c0 + 1) = -Fld(i_edge + i1, i2, c0 + 1);
            Fld(i_edge - i1, i2, c0 + 2) = -Fld(i_edge + i1, i2, c0 + 2);
          } else {
            Fld(i_edge + i1 - 1, i2, c0 + 0) = Fld(i_edge - i1, i2, c0 + 0);
            Fld(i_edge + i1, i2, c0 + 1)     = -Fld(i_edge - i1, i2, c0 + 1);
            Fld(i_edge + i1, i2, c0 + 2)     = -Fld(i_edge - i1, i2, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallEdgeE_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (i1 == 0) {
          Fld(i_edge, i2, i3, c0 + 1) = ZERO;
          Fld(i_edge, i2, i3, c0 + 2) = ZERO;
        } else {
          if constexpr (not P) {
            Fld(i_edge - i1, i2, i3, c0 + 0) = Fld(i_edge + i1 - 1, i2, i3, c0 + 0);
            Fld(i_edge - i1, i2, i3, c0 + 1) = -Fld(i_edge + i1, i2, i3, c0 + 1);
            Fld(i_edge - i1, i2, i3, c0 + 2) = -Fld(i_edge + i1, i2, i3, c0 + 2);
          } else {
            Fld(i_edge + i1 - 1, i2, i3, c0 + 0) = Fld(i_edge - i1, i2, i3, c0 + 0);
            Fld(i_edge + i1, i2, i3, c0 + 1) = -Fld(i_edge - i1, i2, i3, c0 + 1);
            Fld(i_edge + i1, i2, i3, c0 + 2) = -Fld(i_edge - i1, i2, i3, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallEdgeE_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Even (zero-gradient) mirror of face-B into the wall ghosts; the
   *        wall-plane B_n is left untouched (frozen by Faraday + E_tan = 0).
   * @tparam D dimension
   * @tparam P true for the +x1 wall
   * @tparam N number of components of the field array (6 for em, 3 for cur)
   * Launch range: as WallEdgeE_kernel (i1 == 0 is a no-op kept for range
   * symmetry).
   */
  template <Dimension D, bool P, uint8_t N>
  struct WallFaceB_kernel {
    ndfield_t<D, N> Fld;
    const ncells_t  i_edge;
    const uint8_t   c0;

    WallFaceB_kernel(ndfield_t<D, N>& Fld, ncells_t i_edge, uint8_t c0)
      : Fld { Fld }
      , i_edge { i_edge }
      , c0 { c0 } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (i1 != 0) {
          if constexpr (not P) {
            Fld(i_edge - i1, c0 + 0) = Fld(i_edge + i1, c0 + 0);
            Fld(i_edge - i1, c0 + 1) = Fld(i_edge + i1 - 1, c0 + 1);
            Fld(i_edge - i1, c0 + 2) = Fld(i_edge + i1 - 1, c0 + 2);
          } else {
            Fld(i_edge + i1, c0 + 0)     = Fld(i_edge - i1, c0 + 0);
            Fld(i_edge + i1 - 1, c0 + 1) = Fld(i_edge - i1, c0 + 1);
            Fld(i_edge + i1 - 1, c0 + 2) = Fld(i_edge - i1, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallFaceB_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if (i1 != 0) {
          if constexpr (not P) {
            Fld(i_edge - i1, i2, c0 + 0) = Fld(i_edge + i1, i2, c0 + 0);
            Fld(i_edge - i1, i2, c0 + 1) = Fld(i_edge + i1 - 1, i2, c0 + 1);
            Fld(i_edge - i1, i2, c0 + 2) = Fld(i_edge + i1 - 1, i2, c0 + 2);
          } else {
            Fld(i_edge + i1, i2, c0 + 0)     = Fld(i_edge - i1, i2, c0 + 0);
            Fld(i_edge + i1 - 1, i2, c0 + 1) = Fld(i_edge - i1, i2, c0 + 1);
            Fld(i_edge + i1 - 1, i2, c0 + 2) = Fld(i_edge - i1, i2, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallFaceB_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if (i1 != 0) {
          if constexpr (not P) {
            Fld(i_edge - i1, i2, i3, c0 + 0) = Fld(i_edge + i1, i2, i3, c0 + 0);
            Fld(i_edge - i1, i2, i3, c0 + 1) = Fld(i_edge + i1 - 1, i2, i3, c0 + 1);
            Fld(i_edge - i1, i2, i3, c0 + 2) = Fld(i_edge + i1 - 1, i2, i3, c0 + 2);
          } else {
            Fld(i_edge + i1, i2, i3, c0 + 0) = Fld(i_edge - i1, i2, i3, c0 + 0);
            Fld(i_edge + i1 - 1, i2, i3, c0 + 1) = Fld(i_edge - i1, i2, i3, c0 + 1);
            Fld(i_edge + i1 - 1, i2, i3, c0 + 2) = Fld(i_edge - i1, i2, i3, c0 + 2);
          }
        }
      } else {
        raise::KernelError(HERE,
                           "WallFaceB_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Image-plasma treatment of the deposited moments at an x1 wall.
   *        FOLD = true: aux(active) += s_c * aux(ghost) -- folds the ghost
   *        deposit tails back as the image-particle contribution (must run
   *        exactly once per deposit, after SynchronizeFields(AUX)).
   *        FOLD = false: aux(ghost) = s_c * aux(active) -- idempotent mirror
   *        fill for stencils that read the wall ghosts.
   *        Signs: V_x odd (comp 0), V_y/V_z/N even (comps 1..3).
   * @tparam D dimension
   * @tparam P true for the +x1 wall
   * Launch range: i1 in [0, N_GHOSTS) = ghost layer; other indices: active
   * extent for FOLD (corner tails are already remapped by the transverse
   * sync), full extent for fill.
   */
  template <Dimension D, bool P, bool FOLD>
  struct WallMoments_kernel {
    ndfield_t<D, 6> Fld;
    const ncells_t  i_edge;

    WallMoments_kernel(ndfield_t<D, 6>& Fld, ncells_t i_edge)
      : Fld { Fld }
      , i_edge { i_edge } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        if constexpr (FOLD) {
          Fld(ia, 0) += -Fld(ig, 0);
          Fld(ia, 1) += Fld(ig, 1);
          Fld(ia, 2) += Fld(ig, 2);
          Fld(ia, 3) += Fld(ig, 3);
        } else {
          Fld(ig, 0) = -Fld(ia, 0);
          Fld(ig, 1) = Fld(ia, 1);
          Fld(ig, 2) = Fld(ia, 2);
          Fld(ig, 3) = Fld(ia, 3);
        }
      } else {
        raise::KernelError(HERE,
                           "WallMoments_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        if constexpr (FOLD) {
          Fld(ia, i2, 0) += -Fld(ig, i2, 0);
          Fld(ia, i2, 1) += Fld(ig, i2, 1);
          Fld(ia, i2, 2) += Fld(ig, i2, 2);
          Fld(ia, i2, 3) += Fld(ig, i2, 3);
        } else {
          Fld(ig, i2, 0) = -Fld(ia, i2, 0);
          Fld(ig, i2, 1) = Fld(ia, i2, 1);
          Fld(ig, i2, 2) = Fld(ia, i2, 2);
          Fld(ig, i2, 3) = Fld(ia, i2, 3);
        }
      } else {
        raise::KernelError(HERE,
                           "WallMoments_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        if constexpr (FOLD) {
          Fld(ia, i2, i3, 0) += -Fld(ig, i2, i3, 0);
          Fld(ia, i2, i3, 1) += Fld(ig, i2, i3, 1);
          Fld(ia, i2, i3, 2) += Fld(ig, i2, i3, 2);
          Fld(ia, i2, i3, 3) += Fld(ig, i2, i3, 3);
        } else {
          Fld(ig, i2, i3, 0) = -Fld(ia, i2, i3, 0);
          Fld(ig, i2, i3, 1) = Fld(ia, i2, i3, 1);
          Fld(ig, i2, i3, 2) = Fld(ia, i2, i3, 2);
          Fld(ig, i2, i3, 3) = Fld(ia, i2, i3, 3);
        }
      } else {
        raise::KernelError(HERE,
                           "WallMoments_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Conductor-sign mirror fill of the cell-centered gather fields
   *        (bckp: Ec in comps 0..2, Bc in comps 3..5) into the x1-wall ghosts:
   *        E_x even, E_y/E_z odd, B even.
   * @tparam D dimension
   * @tparam P true for the +x1 wall
   * Launch range: i1 in [0, N_GHOSTS); other indices: full extent.
   */
  template <Dimension D, bool P>
  struct WallBckp_kernel {
    ndfield_t<D, 6> Fld;
    const ncells_t  i_edge;

    WallBckp_kernel(ndfield_t<D, 6>& Fld, ncells_t i_edge)
      : Fld { Fld }
      , i_edge { i_edge } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        Fld(ig, 0) = Fld(ia, 0);
        Fld(ig, 1) = -Fld(ia, 1);
        Fld(ig, 2) = -Fld(ia, 2);
        Fld(ig, 3) = Fld(ia, 3);
        Fld(ig, 4) = Fld(ia, 4);
        Fld(ig, 5) = Fld(ia, 5);
      } else {
        raise::KernelError(HERE,
                           "WallBckp_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        Fld(ig, i2, 0) = Fld(ia, i2, 0);
        Fld(ig, i2, 1) = -Fld(ia, i2, 1);
        Fld(ig, i2, 2) = -Fld(ia, i2, 2);
        Fld(ig, i2, 3) = Fld(ia, i2, 3);
        Fld(ig, i2, 4) = Fld(ia, i2, 4);
        Fld(ig, i2, 5) = Fld(ia, i2, 5);
      } else {
        raise::KernelError(HERE,
                           "WallBckp_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        const ncells_t ig = P ? (i_edge + i1) : (i_edge - 1 - i1);
        const ncells_t ia = P ? (i_edge - 1 - i1) : (i_edge + i1);
        Fld(ig, i2, i3, 0) = Fld(ia, i2, i3, 0);
        Fld(ig, i2, i3, 1) = -Fld(ia, i2, i3, 1);
        Fld(ig, i2, i3, 2) = -Fld(ia, i2, i3, 2);
        Fld(ig, i2, i3, 3) = Fld(ia, i2, i3, 3);
        Fld(ig, i2, i3, 4) = Fld(ia, i2, i3, 4);
        Fld(ig, i2, i3, 5) = Fld(ia, i2, i3, 5);
      } else {
        raise::KernelError(HERE,
                           "WallBckp_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_WALL_BCS_HPP

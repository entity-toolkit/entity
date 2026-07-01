/**
 * @file kernels/hybrid/subcycle.hpp
 * @brief Support kernels for the sub-cycled (Pegasus-style) magnetic-field
 *        advance of the HYBRID engine.
 * @implements
 *   - kernel::hybrid::FieldCombine_kernel<> -> dst = a*dst + b*src over a
 *     3-component group (a = 0, b = 1 is a copy). Launched over the FULL
 *     extent so valid ghosts stay valid.
 *   - kernel::hybrid::VwMax_kernel<>        -> parallel_reduce functor for
 *     max over active cells of the local whistler speed at the grid cutoff,
 *     v_w = d0^2 pi |B| / (dx max(N, dens_min)) -- used to pick the number of
 *     field sub-steps each advance needs.
 * @namespaces:
 *   - kernel::hybrid::
 */

#ifndef KERNELS_HYBRID_SUBCYCLE_HPP
#define KERNELS_HYBRID_SUBCYCLE_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::hybrid {
  using namespace ntt;

  template <Dimension D, uint8_t NDST, uint8_t NSRC>
  struct FieldCombine_kernel {
    ndfield_t<D, NDST> Dst;
    ndfield_t<D, NSRC> Src;
    const real_t       a, b;
    const uint8_t      cd, cs;

    FieldCombine_kernel(ndfield_t<D, NDST>&       Dst,
                        const ndfield_t<D, NSRC>& Src,
                        real_t                    a,
                        real_t                    b,
                        uint8_t                   cd,
                        uint8_t                   cs)
      : Dst { Dst }
      , Src { Src }
      , a { a }
      , b { b }
      , cd { cd }
      , cs { cs } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        for (uint8_t c = 0; c < 3; ++c) {
          Dst(i1, cd + c) = a * Dst(i1, cd + c) + b * Src(i1, cs + c);
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldCombine_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        for (uint8_t c = 0; c < 3; ++c) {
          Dst(i1, i2, cd + c) = a * Dst(i1, i2, cd + c) + b * Src(i1, i2, cs + c);
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldCombine_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        for (uint8_t c = 0; c < 3; ++c) {
          Dst(i1, i2, i3, cd + c) = a * Dst(i1, i2, i3, cd + c) +
                                    b * Src(i1, i2, i3, cs + c);
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldCombine_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief max-reduce of v_w(B, N) = d0^2 pi |B|_phys / (dx max(N, dens_min))
   *        over active cells. B is read from a 6-component field (em) at
   *        comp_B (U-basis: in-plane comps are physical/dx); N from aux comp 3.
   *        STENCIL-AWARE like EMF_kernel::hall_limiter: |B| and N are taken
   *        from the worst cells of the +/-1 neighborhood -- at a void /
   *        overshoot interface the large B and the small N sit in DIFFERENT
   *        cells, and a same-cell estimate misses exactly the edge whistler
   *        speed the discrete Hall operator feels.
   */
  template <Dimension D>
  struct VwMax_kernel {
    ndfield_t<D, 6> Bfld;
    ndfield_t<D, 6> NN;
    const uint8_t   comp_B;
    const uint8_t   comp_NN;
    const real_t    d0, dens_min, dx;

    VwMax_kernel(const ndfield_t<D, 6>& Bfld,
                 const ndfield_t<D, 6>& NN,
                 uint8_t                comp_B,
                 uint8_t                comp_NN,
                 real_t                 d0,
                 real_t                 dens_min,
                 real_t                 dx)
      : Bfld { Bfld }
      , NN { NN }
      , comp_B { comp_B }
      , comp_NN { comp_NN }
      , d0 { d0 }
      , dens_min { dens_min }
      , dx { dx } {}

    Inline void operator()(cellidx_t i1, real_t& lmax) const {
      if constexpr (D == Dim::_1D) {
        real_t bsq = ZERO;
        real_t nn  = NN(i1, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          const auto ii = i1 + static_cast<ncells_t>(di);
          // U-basis: b1 = Bx/dx, b2/b3 physical
          const real_t b = SQR(dx * Bfld(ii, comp_B + 0)) +
                           SQR(Bfld(ii, comp_B + 1)) +
                           SQR(Bfld(ii, comp_B + 2));
          bsq = math::max(bsq, b);
          nn  = math::min(nn, NN(ii, comp_NN));
        }
        const real_t vw = SQR(d0) * static_cast<real_t>(constant::PI) *
                          math::sqrt(bsq) / (dx * math::max(nn, dens_min));
        lmax = math::max(lmax, vw);
      } else {
        raise::KernelError(HERE,
                           "VwMax_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, real_t& lmax) const {
      if constexpr (D == Dim::_2D) {
        real_t bsq = ZERO;
        real_t nn  = NN(i1, i2, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            const auto ii = i1 + static_cast<ncells_t>(di);
            const auto jj = i2 + static_cast<ncells_t>(dj);
            // U-basis: b1/b2 = B/dx, b3 physical
            const real_t b = SQR(dx * Bfld(ii, jj, comp_B + 0)) +
                             SQR(dx * Bfld(ii, jj, comp_B + 1)) +
                             SQR(Bfld(ii, jj, comp_B + 2));
            bsq = math::max(bsq, b);
            nn  = math::min(nn, NN(ii, jj, comp_NN));
          }
        }
        const real_t vw = SQR(d0) * static_cast<real_t>(constant::PI) *
                          math::sqrt(bsq) / (dx * math::max(nn, dens_min));
        lmax = math::max(lmax, vw);
      } else {
        raise::KernelError(HERE,
                           "VwMax_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3, real_t& lmax) const {
      if constexpr (D == Dim::_3D) {
        real_t bsq = ZERO;
        real_t nn  = NN(i1, i2, i3, comp_NN);
        for (int di = -1; di <= 1; ++di) {
          for (int dj = -1; dj <= 1; ++dj) {
            for (int dk = -1; dk <= 1; ++dk) {
              const auto ii = i1 + static_cast<ncells_t>(di);
              const auto jj = i2 + static_cast<ncells_t>(dj);
              const auto kk = i3 + static_cast<ncells_t>(dk);
              // U-basis: all comps physical/dx
              const real_t b = SQR(dx * Bfld(ii, jj, kk, comp_B + 0)) +
                               SQR(dx * Bfld(ii, jj, kk, comp_B + 1)) +
                               SQR(dx * Bfld(ii, jj, kk, comp_B + 2));
              bsq = math::max(bsq, b);
              nn  = math::min(nn, NN(ii, jj, kk, comp_NN));
            }
          }
        }
        const real_t vw = SQR(d0) * static_cast<real_t>(constant::PI) *
                          math::sqrt(bsq) / (dx * math::max(nn, dens_min));
        lmax = math::max(lmax, vw);
      } else {
        raise::KernelError(HERE,
                           "VwMax_kernel: 3D implementation called for D != 3");
      }
    }
  };

} // namespace kernel::hybrid

#endif // KERNELS_HYBRID_SUBCYCLE_HPP

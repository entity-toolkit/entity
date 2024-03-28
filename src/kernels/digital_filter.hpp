/**
 * @file digital_filter.hpp
 * @brief Algorithms for covariant digital filtering
 * @implements
 *   - ntt::DigitalFilter_kernel<>
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - utils/error.h
 *   - utils/log.h
 *   - utils/numeric.h
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - BELYAEV_FILTER
 */

#ifndef KERNELS_DIGITAL_FILTER_H
#define KERNELS_DIGITAL_FILTER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/log.h"
#include "utils/numeric.h"

#include <type_traits>

namespace ntt {
  template <Coord::type C>
  struct is_not_cartesian : std::false_type {};

  template <>
  struct is_not_cartesian<Coord::SPH> : std::true_type {};

  template <>
  struct is_not_cartesian<Coord::QSPH> : std::true_type {};

  template <Coord::type C>
  using if_noncart = typename std::enable_if<is_not_cartesian<C>::value>::type;

  template <Dimension D, Coord::type C>
  class DigitalFilterBase {
  protected:
    ndfield_t<D, 3>         array;
    ndfield_t<D, 3>         buffer;
    tuple_t<std::size_t, D> size;

  public:
    DigitalFilterBase(ndfield_t<D, 3>&               array,
                      ndfield_t<D, 3>&               buffer,
                      const tuple_t<std::size_t, D>& size_) :
      array { array },
      buffer { buffer } {
      for (auto i = 0u; i < D; ++i) {
        size[i] = size_[i];
      }
    }

    Inline virtual void operator()(index_t) const {}

    Inline virtual void operator()(index_t, index_t) const {}

    Inline virtual void operator()(index_t, index_t, index_t) const {}
  };

  template <int S, int I, typename = void>
  class DigitalFilter_kernel : public DigitalFilterBase<S, I> {
    using DigitalFilterBase<S, I>::DigitalFilterBase;
    using DigitalFilterBase<S, I>::array;
    using DigitalFilterBase<S, I>::buffer;
    using DigitalFilterBase<S, I>::size;

    Inline void operator()(index_t) const override;
    Inline void operator()(index_t, index_t) const override;
    Inline void operator()(index_t, index_t, index_t) const override;
  };

  /* For flat spacetime ----------------------------------------------------- */
  template <>
  Inline void DigitalFilter_kernel<Dim::_1D, Coord::CART>::operator()(
    index_t i) const {
#pragma unroll
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      array(i, comp) = INV_2 * buffer(i, comp) +
                       INV_4 * (buffer(i - 1, comp) + buffer(i + 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter_kernel<Dim::_2D, Coord::CART>::operator()(
    index_t i,
    index_t j) const {
#pragma unroll
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      array(i,
            j,
            comp) = INV_4 * buffer(i, j, comp) +
                    INV_8 * (buffer(i - 1, j, comp) + buffer(i + 1, j, comp) +
                             buffer(i, j - 1, comp) + buffer(i, j + 1, comp)) +
                    INV_16 *
                      (buffer(i - 1, j - 1, comp) + buffer(i + 1, j + 1, comp) +
                       buffer(i - 1, j + 1, comp) + buffer(i + 1, j - 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter_kernel<Dim::_3D, Coord::CART>::operator()(
    index_t i,
    index_t j,
    index_t k) const {
#pragma unroll
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      array(i, j, k, comp) =
        INV_8 * buffer(i, j, k, comp) +
        INV_16 * (buffer(i - 1, j, k, comp) + buffer(i + 1, j, k, comp) +
                  buffer(i, j - 1, k, comp) + buffer(i, j + 1, k, comp) +
                  buffer(i, j, k - 1, comp) + buffer(i, j, k + 1, comp)) +
        INV_32 * (buffer(i - 1, j - 1, k, comp) + buffer(i + 1, j + 1, k, comp) +
                  buffer(i - 1, j + 1, k, comp) + buffer(i + 1, j - 1, k, comp) +
                  buffer(i, j - 1, k - 1, comp) + buffer(i, j + 1, k + 1, comp) +
                  buffer(i, j, k - 1, comp) + buffer(i, j, k + 1, comp) +
                  buffer(i - 1, j, k - 1, comp) + buffer(i + 1, j, k + 1, comp) +
                  buffer(i - 1, j, k + 1, comp) + buffer(i + 1, j, k - 1, comp)) +
        INV_64 *
          (buffer(i - 1, j - 1, k - 1, comp) + buffer(i + 1, j + 1, k + 1, comp) +
           buffer(i - 1, j + 1, k + 1, comp) + buffer(i + 1, j - 1, k - 1, comp) +
           buffer(i - 1, j - 1, k + 1, comp) + buffer(i + 1, j + 1, k - 1, comp) +
           buffer(i - 1, j + 1, k - 1, comp) + buffer(i + 1, j - 1, k + 1, comp));
    }
  }

  /* For spherical coordinates ---------------------------------------------- */

#define FILTER_IN_I1(ARR, COMP, I, J)                                          \
  INV_2*(ARR)((I), (J), (COMP)) +                                              \
    INV_4*((ARR)((I) - 1, (J), (COMP)) + (ARR)((I) + 1, (J), (COMP)))

  template <Dimension D, Coord::type C>
  class DigitalFilter_kernel<D, C, if_noncart<C>>
    : public DigitalFilterBase<D, C> {
    using DigitalFilterBase<D, C>::DigitalFilterBase;
    using DigitalFilterBase<S, I>::array;
    using DigitalFilterBase<S, I>::buffer;
    using DigitalFilterBase<S, I>::size;

    Inline void operator()(index_t, index_t) const override {
      if constexpr (D == Dim::_2D) {
        const std::size_t j_min = N_GHOSTS, j_min_p1 = j_min + 1;
        const std::size_t j_max = size[1] + N_GHOSTS, j_max_m1 = j_max - 1;
        real_t            cur_ij, cur_ijp1, cur_ijm1;
#if defined(BELYAEV_FILTER) // Belyaev filter
        if (j == j_min) {
          /* --------------------------------- r, phi --------------------------------- */
          for (auto& comp : { cur::jx1, cur::jx3 }) {
            // ... filter in r
            cur_ij            = FILTER_IN_I1(buffer, comp, i, j);
            cur_ijp1          = FILTER_IN_I1(buffer, comp, i, j + 1);
            // ... filter in theta
            array(i, j, comp) = INV_2 * cur_ij + INV_4 * cur_ijp1;
          }

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijp1              = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijp1);
        } else if (j == j_min_p1) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          for (auto& comp : { cur::jx1, cur::jx3 }) {
            // ... filter in r
            cur_ij            = FILTER_IN_I1(buffer, comp, i, j);
            cur_ijp1          = FILTER_IN_I1(buffer, comp, i, j + 1);
            cur_ijm1          = FILTER_IN_I1(buffer, comp, i, j - 1);
            // ... filter in theta
            array(i, j, comp) = INV_2 * (cur_ij + cur_ijm1) + INV_4 * cur_ijp1;
          }

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij   = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijp1 = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
          cur_ijm1 = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_2 * cur_ij + INV_4 * (cur_ijm1 + cur_ijp1);
        } else if (j == j_max_m1) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          for (auto& comp : { cur::jx1, cur::jx3 }) {
            // ... filter in r
            cur_ij            = FILTER_IN_I1(buffer, comp, i, j);
            cur_ijp1          = FILTER_IN_I1(buffer, comp, i, j + 1);
            cur_ijm1          = FILTER_IN_I1(buffer, comp, i, j - 1);
            // ... filter in theta
            array(i, j, comp) = INV_2 * (cur_ij + cur_ijp1) + INV_4 * cur_ijm1;
          }

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijm1              = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijm1);
        } else if (j == j_max) {
          /* --------------------------------- r, phi --------------------------------- */
          for (auto& comp : { cur::jx1, cur::jx3 }) {
            // ... filter in r
            cur_ij            = FILTER_IN_I1(buffer, comp, i, j);
            cur_ijm1          = FILTER_IN_I1(buffer, comp, i, j - 1);
            // ... filter in theta
            array(i, j, comp) = INV_2 * cur_ij + INV_4 * cur_ijm1;
          }
          // no theta component in the last cell
        } else {
#else // more conventional filtering
        if (j == j_min) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx1, i, j);
          cur_ijp1              = FILTER_IN_I1(buffer, cur::jx1, i, j + 1);
          // ... filter in theta
          array(i, j, cur::jx1) = INV_2 * cur_ij + INV_2 * cur_ijp1;

          array(i, j, cur::jx3) = ZERO;

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijp1              = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijp1);
        } else if (j == j_min_p1) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          cur_ij   = FILTER_IN_I1(buffer, cur::jx1, i, j);
          cur_ijp1 = FILTER_IN_I1(buffer, cur::jx1, i, j + 1);
          cur_ijm1 = FILTER_IN_I1(buffer, cur::jx1, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * (cur_ijp1 + cur_ijm1);

          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx3, i, j);
          cur_ijp1              = FILTER_IN_I1(buffer, cur::jx3, i, j + 1);
          // ... filter in theta
          array(i, j, cur::jx3) = INV_2 * cur_ij + INV_4 * cur_ijp1;

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij   = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijp1 = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
          cur_ijm1 = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_2 * cur_ij + INV_4 * (cur_ijm1 + cur_ijp1);
        } else if (j == j_max_m1) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          cur_ij   = FILTER_IN_I1(buffer, cur::jx1, i, j);
          cur_ijp1 = FILTER_IN_I1(buffer, cur::jx1, i, j + 1);
          cur_ijm1 = FILTER_IN_I1(buffer, cur::jx1, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * (cur_ijm1 + cur_ijp1);

          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx3, i, j);
          cur_ijm1              = FILTER_IN_I1(buffer, cur::jx3, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx3) = INV_2 * cur_ij + INV_4 * cur_ijm1;

          /* ---------------------------------- theta --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
          cur_ijm1              = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx2) = INV_4 * (cur_ij + cur_ijm1);
        } else if (j == j_max) {
          /* --------------------------------- r, phi --------------------------------- */
          // ... filter in r
          cur_ij                = FILTER_IN_I1(buffer, cur::jx1, i, j);
          cur_ijm1              = FILTER_IN_I1(buffer, cur::jx1, i, j - 1);
          // ... filter in theta
          array(i, j, cur::jx1) = INV_2 * cur_ij + INV_2 * cur_ijm1;

          array(i, j, cur::jx3) = ZERO;
        } else {
#endif
#pragma unroll
          for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
            array(i, j, comp) =
              INV_4 * buffer(i, j, comp) +
              INV_8 * (buffer(i - 1, j, comp) + buffer(i + 1, j, comp) +
                       buffer(i, j - 1, comp) + buffer(i, j + 1, comp)) +
              INV_16 * (buffer(i - 1, j - 1, comp) + buffer(i + 1, j + 1, comp) +
                        buffer(i - 1, j + 1, comp) + buffer(i + 1, j - 1, comp));
          }
        }
      } else { // D != Dim::_2D
        raise::KernelError(
          HERE,
          "DigitalFilter_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t, index_t, index_t) const override {
      if constexpr (D == Dim::_3D) {
        raise::KernelNotImplementedError(HERE);
      } else {
        raise::KernelError(
          HERE,
          "DigitalFilter_kernel: 3D implementation called for D != 3");
      }
    }
  };

#undef FILTER_IN_I1

} // namespace ntt

#endif // DIGITAL_FILTER_H

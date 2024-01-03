#ifndef FRAMEWORK_DIGITAL_FILTER_H
#define FRAMEWORK_DIGITAL_FILTER_H

#include "wrapper.h"

#include "meshblock/fields.h"

namespace ntt {
  /**
   * @brief Spatial filtering of all the deposited currents.
   * @tparam D Dimension.
   */
  template <Dimension D>
  class DigitalFilter_kernel {
    ndfield_t<D, 3>         array;
    ndfield_t<D, 3>         buffer;
    tuple_t<std::size_t, D> size;

  public:
    /**
     * @brief Constructor.
     * @param ndfield_t array to be filtered.
     * @param ndfield_t buffer to store the intermediate filtered array.
     */
    DigitalFilter_kernel(ndfield_t<D, 3>&               array,
                         ndfield_t<D, 3>&               buffer,
                         const tuple_t<std::size_t, D>& size_) :
      array { array },
      buffer { buffer } {
      for (auto i = 0; i < (short)D; ++i) {
        size[i] = size_[i];
      }
    }

    /**
     * @brief 1D implementation of the algorithm.
     * @param i1 index.
     */
    Inline void operator()(index_t) const;
    /**
     * @brief 2D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     */
    Inline void operator()(index_t, index_t) const;
    /**
     * @brief 3D implementation of the algorithm.
     * @param i1 index.
     * @param i2 index.
     * @param i3 index.
     */
    Inline void operator()(index_t, index_t, index_t) const;
  };

#ifdef MINKOWSKI_METRIC
  template <>
  Inline void DigitalFilter_kernel<Dim1>::operator()(index_t i) const {
  #pragma unroll
    for (auto& comp : { cur::jx1, cur::jx2, cur::jx3 }) {
      array(i, comp) = INV_2 * buffer(i, comp) +
                       INV_4 * (buffer(i - 1, comp) + buffer(i + 1, comp));
    }
  }

  template <>
  Inline void DigitalFilter_kernel<Dim2>::operator()(index_t i, index_t j) const {
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
  Inline void DigitalFilter_kernel<Dim3>::operator()(index_t i,
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
#else
  template <>
  Inline void DigitalFilter_kernel<Dim1>::operator()(index_t) const {}

  #define FILTER_IN_I1(ARR, COMP, I, J)                                        \
    INV_2*(ARR)((I), (J), (COMP)) +                                            \
      INV_4*((ARR)((I)-1, (J), (COMP)) + (ARR)((I) + 1, (J), (COMP)))

    // #define FILTER_IN_I1(ARR, COMP, I, J) (ARR)((I), (J), (COMP))

  #define BELYAEV_FILTER

  template <>
  Inline void DigitalFilter_kernel<Dim2>::operator()(index_t i, index_t j) const {
    const std::size_t j_min = N_GHOSTS, j_min_p1 = j_min + 1;
    const std::size_t j_max = size[1] + N_GHOSTS, j_max_m1 = j_max - 1;
    real_t            cur_ij, cur_ijp1, cur_ijm1;
  #ifdef BELYAEV_FILTER // Belyaev filter
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
      cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
      cur_ijp1              = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
      cur_ijm1              = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
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
  #else                 // more conventional filtering
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
      cur_ij                = FILTER_IN_I1(buffer, cur::jx1, i, j);
      cur_ijp1              = FILTER_IN_I1(buffer, cur::jx1, i, j + 1);
      cur_ijm1              = FILTER_IN_I1(buffer, cur::jx1, i, j - 1);
      // ... filter in theta
      array(i, j, cur::jx1) = INV_2 * cur_ij + INV_4 * (cur_ijp1 + cur_ijm1);

      // ... filter in r
      cur_ij                = FILTER_IN_I1(buffer, cur::jx3, i, j);
      cur_ijp1              = FILTER_IN_I1(buffer, cur::jx3, i, j + 1);
      // ... filter in theta
      array(i, j, cur::jx3) = INV_2 * cur_ij + INV_4 * cur_ijp1;

      /* ---------------------------------- theta --------------------------------- */
      // ... filter in r
      cur_ij                = FILTER_IN_I1(buffer, cur::jx2, i, j);
      cur_ijp1              = FILTER_IN_I1(buffer, cur::jx2, i, j + 1);
      cur_ijm1              = FILTER_IN_I1(buffer, cur::jx2, i, j - 1);
      // ... filter in theta
      array(i, j, cur::jx2) = INV_2 * cur_ij + INV_4 * (cur_ijm1 + cur_ijp1);
    } else if (j == j_max_m1) {
      /* --------------------------------- r, phi --------------------------------- */
      // ... filter in r
      cur_ij                = FILTER_IN_I1(buffer, cur::jx1, i, j);
      cur_ijp1              = FILTER_IN_I1(buffer, cur::jx1, i, j + 1);
      cur_ijm1              = FILTER_IN_I1(buffer, cur::jx1, i, j - 1);
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
        array(i, j, comp) = INV_4 * buffer(i, j, comp) +
                            INV_8 *
                              (buffer(i - 1, j, comp) + buffer(i + 1, j, comp) +
                               buffer(i, j - 1, comp) + buffer(i, j + 1, comp)) +
                            INV_16 * (buffer(i - 1, j - 1, comp) +
                                      buffer(i + 1, j + 1, comp) +
                                      buffer(i - 1, j + 1, comp) +
                                      buffer(i + 1, j - 1, comp));
      }
    }
  }

  #undef FILTER_IN_I1

  template <>
  Inline void DigitalFilter_kernel<Dim3>::operator()(index_t, index_t, index_t) const {
  }

#endif

} // namespace ntt

#endif // FRAMEWORK_DIGITAL_FILTER_H

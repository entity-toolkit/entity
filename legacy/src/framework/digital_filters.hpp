
template <Dimension D, Coord::type C>
class DigitalFilter_kernel<D, C, if_noncart<C>> : public DigitalFilterBase<D, C> {

  using DigitalFilterBase<D, C>::DigitalFilterBase;
  using DigitalFilterBase<D, C>::array;
  using DigitalFilterBase<D, C>::buffer;
  using DigitalFilterBase<D, C>::size;

public:
  Inline void operator()(index_t i, index_t j) const override {
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
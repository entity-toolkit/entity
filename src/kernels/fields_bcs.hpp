/**
 * @brief: kernels/fields_bcs.hpp
 */

#ifndef KERNELS_FIELDS_BCS_HPP
#define KERNELS_FIELDS_BCS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <class M, idx_t i>
  struct AbsorbFields_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(i <= static_cast<unsigned short>(M::Dim),
                  "Invalid component index");

    ndfield_t<M::Dim, 6> Fld;
    const M              metric;
    const real_t         xg_edge;
    const real_t         dx_abs;
    const BCTags         tags;

    AbsorbFields_kernel(ndfield_t<M::Dim, 6> Fld,
                        const M&             metric,
                        real_t               xg_edge,
                        real_t               dx_abs,
                        BCTags               tags)
      : Fld { Fld }
      , metric { metric }
      , xg_edge { xg_edge }
      , dx_abs { dx_abs }
      , tags { tags } {}

    Inline void operator()(index_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        const auto i1_ = COORD(i1);
        for (const auto comp :
             { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
          if ((comp == em::ex1) and not(tags & BC::Ex1)) {
            continue;
          } else if ((comp == em::ex2) and not(tags & BC::Ex2)) {
            continue;
          } else if ((comp == em::ex3) and not(tags & BC::Ex3)) {
            continue;
          } else if ((comp == em::bx1) and not(tags & BC::Bx1)) {
            continue;
          } else if ((comp == em::bx2) and not(tags & BC::Bx2)) {
            continue;
          } else if ((comp == em::bx3) and not(tags & BC::Bx3)) {
            continue;
          }
          coord_t<M::Dim> x_Cd { ZERO };
          if (comp == em::ex1 or comp == em::bx2 or comp == em::bx3) {
            x_Cd[0] = i1_ + HALF;
          } else if (comp == em::ex2 or comp == em::bx1 or comp == em::ex3) {
            x_Cd[0] = i1_;
          }
          const auto dx = math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(x_Cd[i - 1]) - xg_edge);
          Fld(i1, comp) *= math::tanh(dx / (INV_4 * dx_abs));
        }
      } else {
        raise::KernelError(
          HERE,
          "AbsorbFields_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        for (const auto comp :
             { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
          if ((comp == em::ex1) and not(tags & BC::Ex1)) {
            continue;
          } else if ((comp == em::ex2) and not(tags & BC::Ex2)) {
            continue;
          } else if ((comp == em::ex3) and not(tags & BC::Ex3)) {
            continue;
          } else if ((comp == em::bx1) and not(tags & BC::Bx1)) {
            continue;
          } else if ((comp == em::bx2) and not(tags & BC::Bx2)) {
            continue;
          } else if ((comp == em::bx3) and not(tags & BC::Bx3)) {
            continue;
          }
          coord_t<M::Dim> x_Cd { ZERO };
          if (comp == em::ex1 or comp == em::bx2) {
            x_Cd[0] = i1_ + HALF;
            x_Cd[1] = i2_;
          } else if (comp == em::ex2 or comp == em::bx1) {
            x_Cd[0] = i1_;
            x_Cd[1] = i2_ + HALF;
          } else if (comp == em::ex3) {
            x_Cd[0] = i1_;
            x_Cd[1] = i2_;
          } else if (comp == em::bx3) {
            x_Cd[0] = i1_ + HALF;
            x_Cd[1] = i2_ + HALF;
          }
          const auto dx = math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(x_Cd[i - 1]) - xg_edge);
          Fld(i1, i2, comp) *= math::tanh(dx / (INV_4 * dx_abs));
        }
      } else {
        raise::KernelError(
          HERE,
          "AbsorbFields_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        const auto i3_ = COORD(i3);
        for (const auto comp :
             { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 }) {
          if ((comp == em::ex1) and not(tags & BC::Ex1)) {
            continue;
          } else if ((comp == em::ex2) and not(tags & BC::Ex2)) {
            continue;
          } else if ((comp == em::ex3) and not(tags & BC::Ex3)) {
            continue;
          } else if ((comp == em::bx1) and not(tags & BC::Bx1)) {
            continue;
          } else if ((comp == em::bx2) and not(tags & BC::Bx2)) {
            continue;
          } else if ((comp == em::bx3) and not(tags & BC::Bx3)) {
            continue;
          }
          coord_t<M::Dim> x_Cd { ZERO };
          if (comp == em::ex1) {
            x_Cd[0] = i1_ + HALF;
            x_Cd[1] = i2_;
            x_Cd[2] = i3_;
          } else if (comp == em::ex2) {
            x_Cd[0] = i1_;
            x_Cd[1] = i2_ + HALF;
            x_Cd[2] = i3_;
          } else if (comp == em::ex3) {
            x_Cd[0] = i1_;
            x_Cd[1] = i2_;
            x_Cd[2] = i3_ + HALF;
          } else if (comp == em::bx1) {
            x_Cd[0] = i1_;
            x_Cd[1] = i2_ + HALF;
            x_Cd[2] = i3_ + HALF;
          } else if (comp == em::bx2) {
            x_Cd[0] = i1_ + HALF;
            x_Cd[1] = i2_;
            x_Cd[2] = i3_ + HALF;
          } else if (comp == em::bx3) {
            x_Cd[0] = i1_ + HALF;
            x_Cd[1] = i2_ + HALF;
            x_Cd[2] = i3_;
          }
          const auto dx = math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(x_Cd[i - 1]) - xg_edge);
          Fld(i1, i2, i3, comp) *= math::tanh(dx / (INV_4 * dx_abs));
        }
      } else {
        raise::KernelError(
          HERE,
          "AbsorbFields_kernel: 3D implementation called for D != 3");
      }
    }
  };

  template <Dimension D, bool UPPER>
  struct AxisBoundaries_kernel {
    ndfield_t<D, 6>   Fld;
    const std::size_t i_edge;
    const bool        setE, setB;

    AxisBoundaries_kernel(ndfield_t<D, 6> Fld, std::size_t i_edge, BCTags tags)
      : Fld { Fld }
      , i_edge { i_edge }
      , setE { tags & BC::Ex1 or tags & BC::Ex2 or tags & BC::Ex3 }
      , setB { tags & BC::Bx1 or tags & BC::Bx2 or tags & BC::Bx3 } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (not UPPER) {
          if (setE) {
            Fld(i1, i_edge - 1, em::ex2) = -Fld(i1, i_edge, em::ex2);
            Fld(i1, i_edge, em::ex3)     = ZERO;
          }
          if (setB) {
            Fld(i1, i_edge - 1, em::bx1) = Fld(i1, i_edge, em::bx1);
            Fld(i1, i_edge, em::bx2)     = ZERO;
            Fld(i1, i_edge - 1, em::bx3) = Fld(i1, i_edge, em::bx3);
          }
        } else {
          if (setE) {
            Fld(i1, i_edge, em::ex2) = -Fld(i1, i_edge - 1, em::ex2);
            Fld(i1, i_edge, em::ex3) = ZERO;
          }
          if (setB) {
            Fld(i1, i_edge, em::bx1) = Fld(i1, i_edge - 1, em::bx1);
            Fld(i1, i_edge, em::bx2) = ZERO;
            Fld(i1, i_edge, em::bx3) = Fld(i1, i_edge - 1, em::bx3);
          }
        }
      } else {
        raise::KernelError(HERE, "AxisBoundaries_kernel: D != 2");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_FIELDS_BCS_HPP

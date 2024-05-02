/**
 * @brief: kernels/fields_bcs.hpp
 */

#ifndef KERNELS_FIELDS_BCS_HPP
#define KERNELS_FIELDS_BCS_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel {
  using namespace ntt;

  template <class M, idx_t i>
  struct AbsorbBoundaries_kernel {
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(i <= static_cast<unsigned short>(M::Dim),
                  "Invalid component index");

    ndfield_t<M::Dim, 6> Fld;
    const M              metric;
    const real_t         xg_edge;
    const real_t         dx_abs;
    const BCTags         tags;

    AbsorbBoundaries_kernel(ndfield_t<M::Dim, 6> Fld,
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

  template <Dimension D, bool P>
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
        if constexpr (not P) {
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

  template <class I, class M, bool P, in O>
  struct AtmosphereBoundaries_kernel {
    static constexpr Dimension D = M::Dim;
    static constexpr bool defines_ex1 = traits::has_method<traits::ex1_t, I>::value;
    static constexpr bool defines_ex2 = traits::has_method<traits::ex2_t, I>::value;
    static constexpr bool defines_ex3 = traits::has_method<traits::ex3_t, I>::value;
    static constexpr bool defines_bx1 = traits::has_method<traits::bx1_t, I>::value;
    static constexpr bool defines_bx2 = traits::has_method<traits::bx2_t, I>::value;
    static constexpr bool defines_bx3 = traits::has_method<traits::bx3_t, I>::value;

    static_assert(defines_ex1 and defines_ex2 and defines_ex3 and
                    defines_bx1 and defines_bx2 and defines_bx3,
                  "not all components of E or B are specified in PGEN");
    static_assert(M::is_metric, "M must be a metric class");
    static_assert(static_cast<unsigned short>(O) <
                    static_cast<unsigned short>(M::Dim),
                  "Invalid Orientation");

    ndfield_t<D, 6>   Fld;
    const I           finit;
    const M           metric;
    const std::size_t i_edge;
    const bool        setE, setB;

    AtmosphereBoundaries_kernel(ndfield_t<M::Dim, 6>& Fld,
                                const I&              finit,
                                const M&              metric,
                                std::size_t           i_edge,
                                BCTags                tags)
      : Fld { Fld }
      , finit { finit }
      , metric { metric }
      , i_edge { i_edge + N_GHOSTS }
      , setE { tags & BC::Ex1 or tags & BC::Ex2 or tags & BC::Ex3 }
      , setB { tags & BC::Bx1 or tags & BC::Bx2 or tags & BC::Bx3 } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Ph_0 { ZERO };
        coord_t<Dim::_1D> x_Ph_H { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph_0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph_H);
        bool setEx1 = setE, setEx2 = setE, setEx3 = setE, setBx1 = setB,
             setBx2 = setB, setBx3 = setB;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if (setEx1) {
          Fld(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.ex1(x_Ph_H));
        }
        if (setEx2) {
          Fld(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ },
            finit.ex2(x_Ph_0));
        }
        if (setEx3) {
          Fld(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ },
            finit.ex3(x_Ph_0));
        }
        if (setBx1) {
          Fld(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ },
            finit.bx1(x_Ph_0));
        }
        if (setBx2) {
          Fld(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.bx2(x_Ph_H));
        }
        if (setBx3) {
          Fld(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF },
            finit.bx3(x_Ph_H));
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        coord_t<Dim::_2D> x_Ph_00 { ZERO };
        coord_t<Dim::_2D> x_Ph_0H { ZERO };
        coord_t<Dim::_2D> x_Ph_H0 { ZERO };
        coord_t<Dim::_2D> x_Ph_HH { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Ph_00);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Ph_0H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Ph_H0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                  x_Ph_HH);
        bool setEx1 = setE, setEx2 = setE, setEx3 = setE, setBx1 = setB,
             setBx2 = setB, setBx3 = setB;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else if (O == in::x2) {
          // x2 -- normal
          // x1,x3 -- tangential
          if constexpr (P) {
            setEx2 &= (i2 >= i_edge);
            setBx1 &= (i2 >= i_edge);
            setBx3 &= (i2 >= i_edge);
          } else {
            setEx2 &= (i2 < i_edge);
            setBx1 &= (i2 < i_edge);
            setBx3 &= (i2 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if (setEx1) {
          Fld(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ },
            finit.ex1(x_Ph_H0));
        }
        if (setEx2) {
          Fld(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_, i2_ + HALF },
            finit.ex2(x_Ph_0H));
        }
        if (setEx3) {
          Fld(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_, i2_ },
            finit.ex3(x_Ph_00));
        }
        if (setBx1) {
          Fld(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_, i2_ + HALF },
            finit.bx1(x_Ph_0H));
        }
        if (setBx2) {
          Fld(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ },
            finit.bx2(x_Ph_H0));
        }
        if (setBx3) {
          Fld(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ + HALF },
            finit.bx3(x_Ph_HH));
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        const auto        i1_ = COORD(i1);
        const auto        i2_ = COORD(i2);
        const auto        i3_ = COORD(i3);
        coord_t<Dim::_3D> x_Ph_00H { ZERO };
        coord_t<Dim::_3D> x_Ph_0H0 { ZERO };
        coord_t<Dim::_3D> x_Ph_H00 { ZERO };
        coord_t<Dim::_3D> x_Ph_HH0 { ZERO };
        coord_t<Dim::_3D> x_Ph_H0H { ZERO };
        coord_t<Dim::_3D> x_Ph_0HH { ZERO };

        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                  x_Ph_00H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                  x_Ph_0H0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                  x_Ph_H00);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF, i3_ },
                                                  x_Ph_HH0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ + HALF },
                                                  x_Ph_H0H);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ + HALF },
                                                  x_Ph_0HH);
        bool setEx1 = setE, setEx2 = setE, setEx3 = setE, setBx1 = setB,
             setBx2 = setB, setBx3 = setB;
        if constexpr (O == in::x1) {
          // x1 -- normal
          // x2,x3 -- tangential
          if constexpr (P) {
            setEx1 &= (i1 >= i_edge);
            setBx2 &= (i1 >= i_edge);
            setBx3 &= (i1 >= i_edge);
          } else {
            setEx1 &= (i1 < i_edge);
            setBx2 &= (i1 < i_edge);
            setBx3 &= (i1 < i_edge);
          }
        } else if (O == in::x2) {
          // x2 -- normal
          // x1,x3 -- tangential
          if constexpr (P) {
            setEx2 &= (i2 >= i_edge);
            setBx1 &= (i2 >= i_edge);
            setBx3 &= (i2 >= i_edge);
          } else {
            setEx2 &= (i2 < i_edge);
            setBx1 &= (i2 < i_edge);
            setBx3 &= (i2 < i_edge);
          }
        } else if (O == in::x3) {
          // x3 -- normal
          // x1,x2 -- tangential
          if constexpr (P) {
            setEx3 &= (i3 >= i_edge);
            setBx1 &= (i3 >= i_edge);
            setBx2 &= (i3 >= i_edge);
          } else {
            setEx3 &= (i3 < i_edge);
            setBx1 &= (i3 < i_edge);
            setBx2 &= (i3 < i_edge);
          }
        } else {
          raise::KernelError(HERE, "Invalid Orientation");
        }
        if (setEx1) {
          Fld(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_ + HALF, i2_, i3_ },
            finit.ex1(x_Ph_H00));
        }
        if (setEx2) {
          Fld(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_, i2_ + HALF, i3_ },
            finit.ex2(x_Ph_0H0));
        }
        if (setEx3) {
          Fld(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_, i2_, i3_ + HALF },
            finit.ex3(x_Ph_00H));
        }
        if (setBx1) {
          Fld(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
            { i1_, i2_ + HALF, i3_ + HALF },
            finit.bx1(x_Ph_0HH));
        }
        if (setBx2) {
          Fld(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
            { i1_ + HALF, i2_, i3_ + HALF },
            finit.bx2(x_Ph_H0H));
        }
        if (setBx3) {
          Fld(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
            { i1_ + HALF, i2_ + HALF, i3_ },
            finit.bx3(x_Ph_HH0));
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

} // namespace kernel

#endif // KERNELS_FIELDS_BCS_HPP

/**
 * @file kernels/fields_bcs.hpp
 * @brief Kernels used for field boundary conditions
 * @implements
 *   - kernel::bc::MatchBoundaries_kernel<>
 *   - kernel::bc::AxisBoundaries_kernel<>
 *   - kernel::bc::AxisBoundariesGR_kernel<>
 *   - kernel::bc::AbsorbCurrentsGR_kernel<>
 *   - kernel::bc::EnforcedBoundaries_kernel<>
 *   - kernel::bc::HorizonBoundaries_kernel<>
 *   - kernel::bc::ConductorBoundaries_kernel<>
 * @namespaces:
 *   - kernel::bc::
 */

#ifndef KERNELS_FIELDS_BCS_HPP
#define KERNELS_FIELDS_BCS_HPP

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "traits/archetypes.h"
#include "traits/engine.h"
#include "traits/metric.h"
#include "utils/error.h"
#include "utils/numeric.h"

namespace kernel::bc {
  using namespace ntt;

  /*
   * @tparam S: Simulation Engine
   * @tparam I: Field Setter class
   * @tparam M: Metric
   * @tparam o: Orientation
   *
   * @brief Applies matching boundary conditions (with a smooth profile) in a specific direction.
   * @note If a component is not specified in the field setter, it is ignored.
   * @note It is supposed to only be called on the active side of the absorbing edge (so sign is not needed).
   */
  template <SimEngine::type S, MetricClass M, FieldSetterClass<S, M::Dim> FS, in o>
  struct MatchBoundaries_kernel {
    static_assert(static_cast<dim_t>(o) < static_cast<dim_t>(M::Dim),
                  "Invalid component index");
    static constexpr auto  D      = M::Dim;
    static constexpr auto  HasEx1 = ::traits::fieldsetter::HasEx1<FS, D>;
    static constexpr auto  HasEx2 = ::traits::fieldsetter::HasEx2<FS, D>;
    static constexpr auto  HasEx3 = ::traits::fieldsetter::HasEx3<FS, D>;
    static constexpr auto  HasBx1 = ::traits::fieldsetter::HasBx1<FS, D>;
    static constexpr auto  HasBx2 = ::traits::fieldsetter::HasBx2<FS, D>;
    static constexpr auto  HasBx3 = ::traits::fieldsetter::HasBx3<FS, D>;
    static constexpr auto  HasDx1 = ::traits::fieldsetter::HasDx1<FS, D>;
    static constexpr auto  HasDx2 = ::traits::fieldsetter::HasDx2<FS, D>;
    static constexpr auto  HasDx3 = ::traits::fieldsetter::HasDx3<FS, D>;
    static constexpr idx_t i      = static_cast<idx_t>(o) + 1u;

    ndfield_t<M::Dim, 6> Fld;
    const FS             fset;
    const M              metric;
    const real_t         xg_edge;
    const real_t         dx_abs;
    const BCTags         tags;

    ncells_t extent_2 { 0u };
    bool     is_axis_i2min { false }, is_axis_i2max { false };

    MatchBoundaries_kernel(ndfield_t<M::Dim, 6>&       Fld,
                           const FS&                   fset,
                           const M&                    metric,
                           real_t                      xg_edge,
                           real_t                      dx_abs,
                           BCTags                      tags,
                           const boundaries_t<FldsBC>& boundaries)
      : Fld { Fld }
      , fset { fset }
      , metric { metric }
      , xg_edge { xg_edge }
      , dx_abs { dx_abs }
      , tags { tags } {
      if constexpr ((M::CoordType != Coord::Cartesian) &&
                    ((D == Dim::_2D) || (D == Dim::_3D))) {
        raise::ErrorIf(boundaries.size() < 2, "boundaries defined incorrectly", HERE);
        is_axis_i2min = (boundaries[1].first == FldsBC::AXIS);
        is_axis_i2max = (boundaries[1].second == FldsBC::AXIS);
        extent_2      = static_cast<ncells_t>(Fld.extent(1));
      }
    }

    Inline auto shape(real_t dx) const -> real_t {
      return math::tanh(dx * FOUR / dx_abs);
    }

    Inline void operator()(cellidx_t i1) const {
      if constexpr (M::Dim == Dim::_1D) {
        const auto i1_ = COORD(i1);

        if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
          coord_t<Dim::_1D> x_Ph_0 { ZERO };
          coord_t<Dim::_1D> x_Ph_H { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph_0);
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph_H);

          if constexpr (HasEx1 or HasBx2 or HasBx3) {
            const auto s = shape(math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(i1_ + HALF) - xg_edge));
            if constexpr (HasEx1) {
              if (tags & BC::E) {
                Fld(i1, em::ex1) = s * Fld(i1, em::ex1) +
                                   (ONE - s) *
                                     metric.template transform<1, Idx::T, Idx::U>(
                                       { i1_ + HALF },
                                       fset.ex1(x_Ph_H));
              }
            }
            if constexpr (HasBx2 or HasBx3) {
              if (tags & BC::B) {
                if constexpr (HasBx2) {
                  Fld(i1, em::bx2) = s * Fld(i1, em::bx2) +
                                     (ONE - s) *
                                       metric.template transform<2, Idx::T, Idx::U>(
                                         { i1_ + HALF },
                                         fset.bx2(x_Ph_H));
                }
                if constexpr (HasBx3) {
                  Fld(i1, em::bx3) = s * Fld(i1, em::bx3) +
                                     (ONE - s) *
                                       metric.template transform<3, Idx::T, Idx::U>(
                                         { i1_ + HALF },
                                         fset.bx3(x_Ph_H));
                }
              }
            }
          }
          if constexpr (HasBx1 or HasEx2 or HasEx3) {
            const auto s = shape(math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(i1_) - xg_edge));
            if constexpr (HasBx1) {
              if (tags & BC::B) {
                Fld(i1, em::bx1) = s * Fld(i1, em::bx1) +
                                   (ONE - s) *
                                     metric.template transform<1, Idx::T, Idx::U>(
                                       { i1_ },
                                       fset.bx1(x_Ph_0));
              }
            }
            if constexpr (HasEx2 or HasEx3) {
              if (tags & BC::E) {
                if constexpr (HasEx2) {
                  Fld(i1, em::ex2) = s * Fld(i1, em::ex2) +
                                     (ONE - s) *
                                       metric.template transform<2, Idx::T, Idx::U>(
                                         { i1_ },
                                         fset.ex2(x_Ph_0));
                }
                if constexpr (HasEx3) {
                  Fld(i1, em::ex3) = s * Fld(i1, em::ex3) +
                                     (ONE - s) *
                                       metric.template transform<3, Idx::T, Idx::U>(
                                         { i1_ },
                                         fset.ex3(x_Ph_0));
                }
              }
            }
          }
        } else {
          raise::KernelError(HERE, "not implemented");
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (M::Dim == Dim::_2D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);

        // SRPIC
        if constexpr (HasEx1 or HasDx1 or HasBx2) {
          // i1 + 1/2, i2
          real_t xi_Cd;
          if constexpr (o == in::x1) {
            xi_Cd = i1_ + HALF;
          } else {
            xi_Cd = i2_;
          }

          const auto s = shape(math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

          coord_t<Dim::_2D> x_Ph_H0 { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ }, x_Ph_H0);

          if constexpr (HasEx1 or HasDx1) {
            if ((tags & BC::E) or (tags & BC::D)) {
              if constexpr (HasEx1 and
                            ::traits::engine::UserFieldsInTetradBasis<S>) {
                Fld(i1, i2, em::ex1) = s * Fld(i1, i2, em::ex1) +
                                       (ONE - s) *
                                         metric.template transform<1, Idx::T, Idx::U>(
                                           { i1_ + HALF, i2_ },
                                           fset.ex1(x_Ph_H0));
              } else if constexpr (
                HasDx1 and (not ::traits::engine::UserFieldsInTetradBasis<S>)) {
                Fld(i1, i2, em::dx1) = s * Fld(i1, i2, em::dx1) +
                                       (ONE - s) * fset.dx1(x_Ph_H0);
              }
            }
          }

          if constexpr (HasBx2) {
            if (tags & BC::B) {
              if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
                Fld(i1, i2, em::bx2) = s * Fld(i1, i2, em::bx2) +
                                       (ONE - s) *
                                         metric.template transform<2, Idx::T, Idx::U>(
                                           { i1_ + HALF, i2_ },
                                           fset.bx2(x_Ph_H0));
              } else {
                Fld(i1, i2, em::bx2) = s * Fld(i1, i2, em::bx2) +
                                       (ONE - s) * fset.bx2(x_Ph_H0);
              }
            }
          }
        }

        if constexpr (HasEx2 or HasDx2 or HasBx1) {
          // i1, i2 + 1/2
          real_t xi_Cd;
          if constexpr (o == in::x1) {
            xi_Cd = i1_;
          } else {
            xi_Cd = i2_ + HALF;
          }

          const auto s = shape(math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

          coord_t<Dim::_2D> x_Ph_0H { ZERO };
          metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF }, x_Ph_0H);

          if constexpr (HasEx2 or HasDx2) {
            if ((tags & BC::E) or (tags & BC::D)) {
              if constexpr (HasEx2 and
                            ::traits::engine::UserFieldsInTetradBasis<S>) {
                Fld(i1, i2, em::ex2) = s * Fld(i1, i2, em::ex2) +
                                       (ONE - s) *
                                         metric.template transform<2, Idx::T, Idx::U>(
                                           { i1_, i2_ + HALF },
                                           fset.ex2(x_Ph_0H));
              } else if constexpr (
                HasDx2 and (not ::traits::engine::UserFieldsInTetradBasis<S>)) {
                Fld(i1, i2, em::dx2) = s * Fld(i1, i2, em::dx2) +
                                       (ONE - s) * fset.dx2(x_Ph_0H);
              }
            }
          }

          if constexpr (HasBx1) {
            if (tags & BC::B) {
              if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
                Fld(i1, i2, em::bx1) = s * Fld(i1, i2, em::bx1) +
                                       (ONE - s) *
                                         metric.template transform<1, Idx::T, Idx::U>(
                                           { i1_, i2_ + HALF },
                                           fset.bx1(x_Ph_0H));
              } else {
                Fld(i1, i2, em::bx1) = s * Fld(i1, i2, em::bx1) +
                                       (ONE - s) * fset.bx1(x_Ph_0H);
              }
            }
          }
        }

        // if constexpr (defines_ex3 or defines_dx3) {
        if constexpr (HasEx3 or HasDx3) {
          if (tags & BC::E) {
            // i1, i2
            real_t xi_Cd;
            if constexpr (o == in::x1) {
              xi_Cd = i1_;
            } else {
              xi_Cd = i2_;
            }

            const auto s = shape(math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

            coord_t<Dim::_2D> x_Ph_00 { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ }, x_Ph_00);

            if constexpr (HasEx3 and ::traits::engine::UserFieldsInTetradBasis<S>) {
              Fld(i1, i2, em::ex3) = s * Fld(i1, i2, em::ex3);
              if ((!is_axis_i2min or (i2 > N_GHOSTS)) and
                  (!is_axis_i2max or (i2 < extent_2 - N_GHOSTS))) {
                Fld(i1, i2, em::ex3) += (ONE - s) *
                                        metric.template transform<3, Idx::T, Idx::U>(
                                          { i1_, i2_ },
                                          fset.ex3(x_Ph_00));
              }
            } else if constexpr (
              HasDx3 and (not ::traits::engine::UserFieldsInTetradBasis<S>)) {
              Fld(i1, i2, em::dx3) = s * Fld(i1, i2, em::dx3);
              if ((!is_axis_i2min or (i2 > N_GHOSTS)) and
                  (!is_axis_i2max or (i2 < extent_2 - N_GHOSTS))) {
                Fld(i1, i2, em::dx3) += (ONE - s) * fset.dx3(x_Ph_00);
              }
            }
          }
        }

        if constexpr (HasBx3) {
          if (tags & BC::B) {
            // i1 + 1/2, i2 + 1/2
            real_t xi_Cd;
            if constexpr (o == in::x1) {
              xi_Cd = i1_ + HALF;
            } else {
              xi_Cd = i2_ + HALF;
            }

            const auto s = shape(math::abs(
              metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

            coord_t<Dim::_2D> x_Ph_HH { ZERO };
            metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_ + HALF },
                                                      x_Ph_HH);

            if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
              Fld(i1, i2, em::bx3) = s * Fld(i1, i2, em::bx3) +
                                     (ONE - s) *
                                       metric.template transform<3, Idx::T, Idx::U>(
                                         { i1_ + HALF, i2_ + HALF },
                                         fset.bx3(x_Ph_HH));
            } else {
              Fld(i1, i2, em::bx3) = s * Fld(i1, i2, em::bx3) +
                                     (ONE - s) * fset.bx3(x_Ph_HH);
            }
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (M::Dim == Dim::_3D) {
        const auto i1_ = COORD(i1);
        const auto i2_ = COORD(i2);
        const auto i3_ = COORD(i3);

        if constexpr (::traits::engine::UserFieldsInTetradBasis<S>) {
          // SRPIC
          if constexpr (HasEx1 or HasEx2 or HasEx3) {
            if (tags & BC::E) {
              if constexpr (HasEx1) {
                // i1 + 1/2, i2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_;
                }
                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_H00 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF, i2_, i3_ },
                                                          x_Ph_H00);

                Fld(i1, i2, i3, em::ex1) =
                  s * Fld(i1, i2, i3, em::ex1) +
                  (ONE - s) * metric.template transform<1, Idx::T, Idx::U>(
                                { i1_ + HALF, i2_, i3_ },
                                fset.ex1(x_Ph_H00));
              }

              if constexpr (HasEx2) {
                // i1, i2 + 1/2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_;
                }
                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_0H0 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_ + HALF, i3_ },
                                                          x_Ph_0H0);

                Fld(i1, i2, i3, em::ex2) =
                  s * Fld(i1, i2, i3, em::ex2) +
                  (ONE - s) * metric.template transform<2, Idx::T, Idx::U>(
                                { i1_, i2_ + HALF, i3_ },
                                fset.ex2(x_Ph_0H0));
              }

              if constexpr (HasEx3) {
                // i1, i2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_00H { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>({ i1_, i2_, i3_ + HALF },
                                                          x_Ph_00H);
                Fld(i1, i2, i3, em::ex3) = s * Fld(i1, i2, i3, em::ex3);
                if ((!is_axis_i2min or (i2 > N_GHOSTS)) and
                    (!is_axis_i2max or (i2 < extent_2 - N_GHOSTS))) {
                  Fld(i1, i2, i3, em::ex3) +=
                    (ONE - s) * metric.template transform<3, Idx::T, Idx::U>(
                                  { i1_, i2_, i3_ + HALF },
                                  fset.ex3(x_Ph_00H));
                }
              }
            }
          }

          if constexpr (HasBx1 or HasBx2 or HasBx3) {
            if (tags & BC::B) {
              if constexpr (HasBx1) {
                // i1, i2 + 1/2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_0HH { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>(
                  { i1_, i2_ + HALF, i3_ + HALF },
                  x_Ph_0HH);

                Fld(i1, i2, i3, em::bx1) =
                  s * Fld(i1, i2, i3, em::bx1) +
                  (ONE - s) * metric.template transform<1, Idx::T, Idx::U>(
                                { i1_, i2_ + HALF, i3_ + HALF },
                                fset.bx1(x_Ph_0HH));
              }

              if constexpr (HasBx2) {
                // i1 + 1/2, i2, i3 + 1/2
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_;
                } else {
                  xi_Cd = i3_ + HALF;
                }
                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_H0H { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>(
                  { i1_ + HALF, i2_, i3_ + HALF },
                  x_Ph_H0H);

                Fld(i1, i2, i3, em::bx2) =
                  s * Fld(i1, i2, i3, em::bx2) +
                  (ONE - s) * metric.template transform<2, Idx::T, Idx::U>(
                                { i1_ + HALF, i2_, i3_ + HALF },
                                fset.bx2(x_Ph_H0H));
              }

              if constexpr (HasBx3) {
                // i1 + 1/2, i2 + 1/2, i3
                real_t xi_Cd;
                if constexpr (o == in::x1) {
                  xi_Cd = i1_ + HALF;
                } else if constexpr (o == in::x2) {
                  xi_Cd = i2_ + HALF;
                } else {
                  xi_Cd = i3_;
                }

                const auto s = shape(math::abs(
                  metric.template convert<i, Crd::Cd, Crd::Ph>(xi_Cd) - xg_edge));

                coord_t<Dim::_3D> x_Ph_HH0 { ZERO };
                metric.template convert<Crd::Cd, Crd::Ph>(
                  { i1_ + HALF, i2_ + HALF, i3_ },
                  x_Ph_HH0);

                Fld(i1, i2, i3, em::bx3) =
                  s * Fld(i1, i2, i3, em::bx3) +
                  (ONE - s) * metric.template transform<3, Idx::T, Idx::U>(
                                { i1_ + HALF, i2_ + HALF, i3_ },
                                fset.bx3(x_Ph_HH0));
              }
            }
          }
        } else {
          raise::KernelError(HERE, "not implemented");
        }
      } else {
        raise::KernelError(
          HERE,
          "MatchBoundaries_kernel: 3D implementation called for D != 3");
      }
    }
  };

  template <Dimension D, in o, bool P>
  struct ConductorBoundaries_kernel {
    static_assert(static_cast<dim_t>(o) < static_cast<dim_t>(D),
                  "Invalid component index");

    ndfield_t<D, 6> Fld;
    const ncells_t  i_edge;
    const BCTags    tags;

    ConductorBoundaries_kernel(ndfield_t<D, 6>& Fld, ncells_t i_edge, BCTags tags)
      : Fld { Fld }
      , i_edge { i_edge }
      , tags { tags } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (tags & BC::E) {
          if (i1 == 0) {
            Fld(i_edge, em::ex2) = ZERO;
            Fld(i_edge, em::ex3) = ZERO;
          } else {
            if constexpr (not P) {
              Fld(i_edge - i1, em::ex1) = Fld(i_edge + i1 - 1, em::ex1);
              Fld(i_edge - i1, em::ex2) = -Fld(i_edge + i1, em::ex2);
              Fld(i_edge - i1, em::ex3) = -Fld(i_edge + i1, em::ex3);
            } else {
              Fld(i_edge + i1 - 1, em::ex1) = Fld(i_edge - i1, em::ex1);
              Fld(i_edge + i1, em::ex2)     = -Fld(i_edge - i1, em::ex2);
              Fld(i_edge + i1, em::ex3)     = -Fld(i_edge - i1, em::ex3);
            }
          }
        }

        if (tags & BC::B) {
          if (i1 == 0) {
            Fld(i_edge, em::bx1) = ZERO;
          } else {
            if constexpr (not P) {
              Fld(i_edge - i1, em::bx1) = -Fld(i_edge + i1, em::bx1);
              Fld(i_edge - i1, em::bx2) = Fld(i_edge + i1 - 1, em::bx2);
              Fld(i_edge - i1, em::bx3) = Fld(i_edge + i1 - 1, em::bx3);
            } else {
              Fld(i_edge + i1, em::bx1)     = -Fld(i_edge - i1, em::bx1);
              Fld(i_edge + i1 - 1, em::bx2) = Fld(i_edge - i1, em::bx2);
              Fld(i_edge + i1 - 1, em::bx3) = Fld(i_edge - i1, em::bx3);
            }
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "ConductorBoundaries_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            if (i1 == 0) {
              Fld(i_edge, i2, em::ex2) = ZERO;
              Fld(i_edge, i2, em::ex3) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i_edge - i1, i2, em::ex1) = Fld(i_edge + i1 - 1, i2, em::ex1);
                Fld(i_edge - i1, i2, em::ex2) = -Fld(i_edge + i1, i2, em::ex2);
                Fld(i_edge - i1, i2, em::ex3) = -Fld(i_edge + i1, i2, em::ex3);
              } else {
                Fld(i_edge + i1 - 1, i2, em::ex1) = Fld(i_edge - i1, i2, em::ex1);
                Fld(i_edge + i1, i2, em::ex2) = -Fld(i_edge - i1, i2, em::ex2);
                Fld(i_edge + i1, i2, em::ex3) = -Fld(i_edge - i1, i2, em::ex3);
              }
            }
          }

          if (tags & BC::B) {
            if (i1 == 0) {
              Fld(i_edge, i2, em::bx1) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i_edge - i1, i2, em::bx1) = -Fld(i_edge + i1, i2, em::bx1);
                Fld(i_edge - i1, i2, em::bx2) = Fld(i_edge + i1 - 1, i2, em::bx2);
                Fld(i_edge - i1, i2, em::bx3) = Fld(i_edge + i1 - 1, i2, em::bx3);
              } else {
                Fld(i_edge + i1, i2, em::bx1) = -Fld(i_edge - i1, i2, em::bx1);
                Fld(i_edge + i1 - 1, i2, em::bx2) = Fld(i_edge - i1, i2, em::bx2);
                Fld(i_edge + i1 - 1, i2, em::bx3) = Fld(i_edge - i1, i2, em::bx3);
              }
            }
          }
        } else {
          if (tags & BC::E) {
            if (i2 == 0) {
              Fld(i1, i_edge, em::ex1) = ZERO;
              Fld(i1, i_edge, em::ex3) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i_edge - i2, em::ex1) = -Fld(i1, i_edge + i2, em::ex1);
                Fld(i1, i_edge - i2, em::ex2) = Fld(i1, i_edge + i2 - 1, em::ex2);
                Fld(i1, i_edge - i2, em::ex3) = -Fld(i1, i_edge + i2, em::ex3);
              } else {
                Fld(i1, i_edge + i2, em::ex1) = -Fld(i1, i_edge - i2, em::ex1);
                Fld(i1, i_edge + i2 - 1, em::ex2) = Fld(i1, i_edge - i2, em::ex2);
                Fld(i1, i_edge + i2, em::ex3) = -Fld(i1, i_edge - i2, em::ex3);
              }
            }
          }

          if (tags & BC::B) {
            if (i2 == 0) {
              Fld(i1, i_edge, em::bx2) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i_edge - i2, em::bx1) = Fld(i1, i_edge + i2 - 1, em::bx1);
                Fld(i1, i_edge - i2, em::bx2) = -Fld(i1, i_edge + i2, em::bx2);
                Fld(i1, i_edge - i2, em::bx3) = Fld(i1, i_edge + i2 - 1, em::bx3);
              } else {
                Fld(i1, i_edge + i2 - 1, em::bx1) = Fld(i1, i_edge - i2, em::bx1);
                Fld(i1, i_edge + i2, em::bx2) = -Fld(i1, i_edge - i2, em::bx2);
                Fld(i1, i_edge + i2 - 1, em::bx3) = Fld(i1, i_edge - i2, em::bx3);
              }
            }
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "ConductorBoundaries_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            if (i1 == 0) {
              Fld(i_edge, i2, i3, em::ex2) = ZERO;
              Fld(i_edge, i2, i3, em::ex3) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i_edge - i1, i2, i3, em::ex1) = Fld(i_edge + i1 - 1,
                                                        i2,
                                                        i3,
                                                        em::ex1);
                Fld(i_edge - i1, i2, i3, em::ex2) = -Fld(i_edge + i1, i2, i3, em::ex2);
                Fld(i_edge - i1, i2, i3, em::ex3) = -Fld(i_edge + i1, i2, i3, em::ex3);
              } else {
                Fld(i_edge + i1 - 1, i2, i3, em::ex1) = Fld(i_edge - i1,
                                                            i2,
                                                            i3,
                                                            em::ex1);
                Fld(i_edge + i1, i2, i3, em::ex2) = -Fld(i_edge - i1, i2, i3, em::ex2);
                Fld(i_edge + i1, i2, i3, em::ex3) = -Fld(i_edge - i1, i2, i3, em::ex3);
              }
            }
          }

          if (tags & BC::B) {
            if (i1 == 0) {
              Fld(i_edge, i2, i3, em::bx1) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i_edge - i1, i2, i3, em::bx1) = -Fld(i_edge + i1, i2, i3, em::bx1);
                Fld(i_edge - i1, i2, i3, em::bx2) = Fld(i_edge + i1 - 1,
                                                        i2,
                                                        i3,
                                                        em::bx2);
                Fld(i_edge - i1, i2, i3, em::bx3) = Fld(i_edge + i1 - 1,
                                                        i2,
                                                        i3,
                                                        em::bx3);
              } else {
                Fld(i_edge + i1, i2, i3, em::bx1) = -Fld(i_edge - i1, i2, i3, em::bx1);
                Fld(i_edge + i1 - 1, i2, i3, em::bx2) = Fld(i_edge - i1,
                                                            i2,
                                                            i3,
                                                            em::bx2);
                Fld(i_edge + i1 - 1, i2, i3, em::bx3) = Fld(i_edge - i1,
                                                            i2,
                                                            i3,
                                                            em::bx3);
              }
            }
          }
        } else if (o == in::x2) {
          if (tags & BC::E) {
            if (i2 == 0) {
              Fld(i1, i_edge, i3, em::ex1) = ZERO;
              Fld(i1, i_edge, i3, em::ex3) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i_edge - i2, i3, em::ex1) = -Fld(i1, i_edge + i2, i3, em::ex1);
                Fld(i1, i_edge - i2, i3, em::ex2) = Fld(i1,
                                                        i_edge + i2 - 1,
                                                        i3,
                                                        em::ex2);
                Fld(i1, i_edge - i2, i3, em::ex3) = -Fld(i1, i_edge + i2, i3, em::ex3);
              } else {
                Fld(i1, i_edge + i2, i3, em::ex1) = -Fld(i1, i_edge - i2, i3, em::ex1);
                Fld(i1, i_edge + i2 - 1, i3, em::ex2) = Fld(i1,
                                                            i_edge - i2,
                                                            i3,
                                                            em::ex2);
                Fld(i1, i_edge + i2, i3, em::ex3) = -Fld(i1, i_edge - i2, i3, em::ex3);
              }
            }
          }

          if (tags & BC::B) {
            if (i2 == 0) {
              Fld(i1, i_edge, i3, em::bx2) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i_edge - i2, i3, em::bx1) = Fld(i1,
                                                        i_edge + i2 - 1,
                                                        i3,
                                                        em::bx1);
                Fld(i1, i_edge - i2, i3, em::bx2) = -Fld(i1, i_edge + i2, i3, em::bx2);
                Fld(i1, i_edge - i2, i3, em::bx3) = Fld(i1,
                                                        i_edge + i2 - 1,
                                                        i3,
                                                        em::bx3);
              } else {
                Fld(i1, i_edge + i2 - 1, i3, em::bx1) = Fld(i1,
                                                            i_edge - i2,
                                                            i3,
                                                            em::bx1);
                Fld(i1, i_edge + i2, i3, em::bx2) = -Fld(i1, i_edge - i2, i3, em::bx2);
                Fld(i1, i_edge + i2 - 1, i3, em::bx3) = Fld(i1,
                                                            i_edge - i2,
                                                            i3,
                                                            em::bx3);
              }
            }
          }
        } else {
          if (tags & BC::E) {
            if (i3 == 0) {
              Fld(i1, i2, i_edge, em::ex1) = ZERO;
              Fld(i1, i2, i_edge, em::ex2) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i2, i_edge - i3, em::ex1) = -Fld(i1, i2, i_edge + i3, em::ex1);
                Fld(i1, i2, i_edge - i3, em::ex2) = -Fld(i1, i2, i_edge + i3, em::ex2);
                Fld(i1, i2, i_edge - i3, em::ex3) = Fld(i1,
                                                        i2,
                                                        i_edge + i3 - 1,
                                                        em::ex3);
              } else {
                Fld(i1, i2, i_edge + i3, em::ex1) = -Fld(i1, i2, i_edge - i3, em::ex1);
                Fld(i1, i2, i_edge + i3, em::ex2) = -Fld(i1, i2, i_edge - i3, em::ex2);
                Fld(i1, i2, i_edge + i3 - 1, em::ex3) = Fld(i1,
                                                            i2,
                                                            i_edge - i3,
                                                            em::ex3);
              }
            }
          }

          if (tags & BC::B) {
            if (i3 == 0) {
              Fld(i1, i2, i_edge, em::bx3) = ZERO;
            } else {
              if constexpr (not P) {
                Fld(i1, i2, i_edge - i3, em::bx1) = Fld(i1,
                                                        i2,
                                                        i_edge + i3 - 1,
                                                        em::bx1);
                Fld(i1, i2, i_edge - i3, em::bx2) = Fld(i1,
                                                        i2,
                                                        i_edge + i3 - 1,
                                                        em::bx2);
                Fld(i1, i2, i_edge - i3, em::bx3) = -Fld(i1, i2, i_edge + i3, em::bx3);
              } else {
                Fld(i1, i2, i_edge + i3 - 1, em::bx1) = Fld(i1,
                                                            i2,
                                                            i_edge - i3,
                                                            em::bx1);
                Fld(i1, i2, i_edge + i3 - 1, em::bx2) = Fld(i1,
                                                            i2,
                                                            i_edge - i3,
                                                            em::bx2);
                Fld(i1, i2, i_edge + i3, em::bx3) = -Fld(i1, i2, i_edge - i3, em::bx3);
              }
            }
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "ConductorBoundaries_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /*
   * @tparam D: Dimension
   * @tparam P: Positive/Negative direction
   *
   * @brief Applies boundary conditions near the polar axis
   */
  template <Dimension D, bool P>
  struct AxisBoundaries_kernel {
    ndfield_t<D, 6> Fld;
    const ncells_t  i_edge;
    const bool      setE { false }, setB { false };

    AxisBoundaries_kernel(ndfield_t<D, 6>& Fld, ncells_t i_edge, BCTags tags)
      : Fld { Fld }
      , i_edge { i_edge }
      , setE { tags & BC::Ex1 or tags & BC::Ex2 or tags & BC::Ex3 }
      , setB { tags & BC::Bx1 or tags & BC::Bx2 or tags & BC::Bx3 } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_2D) {
        // ! TODO: not all components are necessary
        if constexpr (not P) {
          if (setE) {
            Fld(i1, i_edge - 1, em::ex2) = -Fld(i1, i_edge, em::ex2);
            Fld(i1, i_edge, em::ex3)     = ZERO;
            Fld(i1, i_edge - 1, em::ex3) = Fld(i1, i_edge + 1, em::ex3);
          }
          if (setB) {
            Fld(i1, i_edge - 1, em::bx1) = Fld(i1, i_edge, em::bx1);
            Fld(i1, i_edge, em::bx2)     = ZERO;
            Fld(i1, i_edge - 1, em::bx2) = -Fld(i1, i_edge + 1, em::bx2);
            Fld(i1, i_edge - 1, em::bx3) = Fld(i1, i_edge, em::bx3);
          }
        } else {
          if (setE) {
            Fld(i1, i_edge, em::ex2)     = -Fld(i1, i_edge - 1, em::ex2);
            Fld(i1, i_edge, em::ex3)     = ZERO;
            Fld(i1, i_edge + 1, em::ex3) = Fld(i1, i_edge - 1, em::ex3);
          }
          if (setB) {
            Fld(i1, i_edge, em::bx1)     = Fld(i1, i_edge - 1, em::bx1);
            Fld(i1, i_edge, em::bx2)     = ZERO;
            Fld(i1, i_edge + 1, em::bx2) = -Fld(i1, i_edge - 1, em::bx2);
            Fld(i1, i_edge, em::bx3)     = Fld(i1, i_edge - 1, em::bx3);
          }
        }
      } else {
        raise::KernelError(HERE, "AxisBoundaries_kernel: D != 2");
      }
    }
  };

  template <SRMetricClass M, SRFieldSetterClass<M::Dim> FS, bool P, in O>
  struct EnforcedBoundaries_kernel {
    static_assert(static_cast<dim_t>(O) < static_cast<dim_t>(M::Dim),
                  "Invalid component index");
    static constexpr auto D      = M::Dim;
    static constexpr auto HasEx1 = ::traits::fieldsetter::HasEx1<FS, D>;
    static constexpr auto HasEx2 = ::traits::fieldsetter::HasEx2<FS, D>;
    static constexpr auto HasEx3 = ::traits::fieldsetter::HasEx3<FS, D>;
    static constexpr auto HasBx1 = ::traits::fieldsetter::HasBx1<FS, D>;
    static constexpr auto HasBx2 = ::traits::fieldsetter::HasBx2<FS, D>;
    static constexpr auto HasBx3 = ::traits::fieldsetter::HasBx3<FS, D>;

    ndfield_t<D, 6> Fld;
    const FS        fset;
    const M         metric;
    const ncells_t  i_edge;
    const BCTags    tags;

    EnforcedBoundaries_kernel(ndfield_t<M::Dim, 6>& Fld,
                              const FS&             fset,
                              const M&              metric,
                              ncells_t              i_edge,
                              BCTags                tags)
      : Fld { Fld }
      , fset { fset }
      , metric { metric }
      , i_edge { i_edge + N_GHOSTS }
      , tags { tags } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        const auto        i1_ = COORD(i1);
        coord_t<Dim::_1D> x_Ph_0 { ZERO };
        coord_t<Dim::_1D> x_Ph_H { ZERO };
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ }, x_Ph_0);
        metric.template convert<Crd::Cd, Crd::Ph>({ i1_ + HALF }, x_Ph_H);
        bool setEx1 = HasEx1 and (tags & BC::E),
             setEx2 = HasEx2 and (tags & BC::E),
             setEx3 = HasEx3 and (tags & BC::E),
             setBx1 = HasBx1 and (tags & BC::B),
             setBx2 = HasBx2 and (tags & BC::B),
             setBx3 = HasBx3 and (tags & BC::B);
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
        if constexpr (HasEx1) {
          if (setEx1) {
            Fld(i1, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.ex1(x_Ph_H));
          }
        }
        if constexpr (HasEx2) {
          if (setEx2) {
            Fld(i1, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ },
              fset.ex2(x_Ph_0));
          }
        }
        if constexpr (HasEx3) {
          if (setEx3) {
            Fld(i1, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ },
              fset.ex3(x_Ph_0));
          }
        }
        if constexpr (HasBx1) {
          if (setBx1) {
            Fld(i1, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ },
              fset.bx1(x_Ph_0));
          }
        }
        if constexpr (HasBx2) {
          if (setBx2) {
            Fld(i1, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.bx2(x_Ph_H));
          }
        }
        if constexpr (HasBx3) {
          if (setBx3) {
            Fld(i1, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF },
              fset.bx3(x_Ph_H));
          }
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
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
        bool setEx1 = HasEx1 and (tags & BC::E),
             setEx2 = HasEx2 and (tags & BC::E),
             setEx3 = HasEx3 and (tags & BC::E),
             setBx1 = HasBx1 and (tags & BC::B),
             setBx2 = HasBx2 and (tags & BC::B),
             setBx3 = HasBx3 and (tags & BC::B);

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
        if constexpr (HasEx1) {
          if (setEx1) {
            Fld(i1, i2, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              fset.ex1(x_Ph_H0));
          }
        }
        if constexpr (HasEx2) {
          if (setEx2) {
            Fld(i1, i2, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              fset.ex2(x_Ph_0H));
          }
        }
        if constexpr (HasEx3) {
          if (setEx3) {
            Fld(i1, i2, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_ },
              fset.ex3(x_Ph_00));
          }
        }
        if constexpr (HasBx1) {
          if (setBx1) {
            Fld(i1, i2, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF },
              fset.bx1(x_Ph_0H));
          }
        }
        if constexpr (HasBx2) {
          if (setBx2) {
            Fld(i1, i2, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ },
              fset.bx2(x_Ph_H0));
          }
        }
        if constexpr (HasBx3) {
          if (setBx3) {
            Fld(i1, i2, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF },
              fset.bx3(x_Ph_HH));
          }
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
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
        bool setEx1 = HasEx1 and (tags & BC::E),
             setEx2 = HasEx2 and (tags & BC::E),
             setEx3 = HasEx3 and (tags & BC::E),
             setBx1 = HasBx1 and (tags & BC::B),
             setBx2 = HasBx2 and (tags & BC::B),
             setBx3 = HasBx3 and (tags & BC::B);
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
        if constexpr (HasEx1) {
          if (setEx1) {
            Fld(i1, i2, i3, em::ex1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ },
              fset.ex1(x_Ph_H00));
          }
        }
        if constexpr (HasEx2) {
          if (setEx2) {
            Fld(i1, i2, i3, em::ex2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ },
              fset.ex2(x_Ph_0H0));
          }
        }
        if constexpr (HasEx3) {
          if (setEx3) {
            Fld(i1, i2, i3, em::ex3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_, i2_, i3_ + HALF },
              fset.ex3(x_Ph_00H));
          }
        }
        if constexpr (HasBx1) {
          if (setBx1) {
            Fld(i1, i2, i3, em::bx1) = metric.template transform<1, Idx::T, Idx::U>(
              { i1_, i2_ + HALF, i3_ + HALF },
              fset.bx1(x_Ph_0HH));
          }
        }
        if constexpr (HasBx2) {
          if (setBx2) {
            Fld(i1, i2, i3, em::bx2) = metric.template transform<2, Idx::T, Idx::U>(
              { i1_ + HALF, i2_, i3_ + HALF },
              fset.bx2(x_Ph_H0H));
          }
        }
        if constexpr (HasBx3) {
          if (setBx3) {
            Fld(i1, i2, i3, em::bx3) = metric.template transform<3, Idx::T, Idx::U>(
              { i1_ + HALF, i2_ + HALF, i3_ },
              fset.bx3(x_Ph_HH0));
          }
        }
      } else {
        raise::KernelError(HERE, "Invalid Dimension");
      }
    }
  };

  namespace gr {

    template <Dimension D>
    struct HorizonBoundaries_kernel {
      ndfield_t<D, 6> Fld;
      const ncells_t  i1_min;
      const bool      setE { false }, setB { false };
      const ncells_t  nfilter;

      HorizonBoundaries_kernel(ndfield_t<D, 6>& Fld,
                               ncells_t         i1_min,
                               BCTags           tags,
                               ncells_t         nfilter)
        : Fld { Fld }
        , i1_min { i1_min }
        , setE { (tags & BC::Ex1 or tags & BC::Ex2 or tags & BC::Ex3) or
                 (tags & BC::Dx1 or tags & BC::Dx2 or tags & BC::Dx3) }
        , setB { (tags & BC::Bx1 or tags & BC::Bx2 or tags & BC::Bx3) or
                 (tags & BC::Hx1 or tags & BC::Hx2 or tags & BC::Hx3) }
        , nfilter { nfilter } {}

      Inline void operator()(cellidx_t i2) const {
        if constexpr (D == Dim::_2D) {
          if (setE) {
            for (unsigned short i = 0; i <= 2 + nfilter; ++i) {
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::dx1) = Fld(i1_min + 1 + nfilter, i2, em::dx1);
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::dx2) = Fld(i1_min + 1 + nfilter, i2, em::dx2);
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::dx3) = Fld(i1_min + 1 + nfilter, i2, em::dx3);
            }
          }
          if (setB) {
            for (unsigned short i = 0; i <= 2 + nfilter; ++i) {
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::bx1) = Fld(i1_min + 1 + nfilter, i2, em::bx1);
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::bx2) = Fld(i1_min + 1 + nfilter, i2, em::bx2);
              Fld(i1_min - N_GHOSTS + i,
                  i2,
                  em::bx3) = Fld(i1_min + 1 + nfilter, i2, em::bx3);
            }
          }
        } else {
          raise::KernelError(
            HERE,
            "HorizonBoundaries_kernel: 2D implementation called for D != 2");
        }
      }
    };

    template <MetricClass M, idx_t i>
    struct AbsorbCurrents_kernel {
      static_assert(i <= static_cast<unsigned short>(M::Dim),
                    "Invalid component index");

      ndfield_t<M::Dim, 3> J;
      const M              metric;
      const real_t         xg_edge;
      const real_t         dx_abs;

      AbsorbCurrents_kernel(ndfield_t<M::Dim, 3>& J,
                            const M&              metric,
                            real_t                xg_edge,
                            real_t                dx_abs)
        : J { J }
        , metric { metric }
        , xg_edge { xg_edge }
        , dx_abs { dx_abs } {}

      Inline void operator()(cellidx_t i1, cellidx_t i2) const {
        if constexpr (M::Dim == Dim::_2D) {
          const auto      i1_ = COORD(i1);
          const auto      i2_ = COORD(i2);
          coord_t<M::Dim> x_Cd { ZERO };
          x_Cd[0]       = i1_;
          x_Cd[1]       = i2_;
          const auto dx = math::abs(
            metric.template convert<i, Crd::Cd, Crd::Ph>(x_Cd[i - 1]) - xg_edge);
          J(i1, i2, 0) *= math::tanh(dx / (INV_4 * dx_abs));
          J(i1, i2, 1) *= math::tanh(dx / (INV_4 * dx_abs));
          J(i1, i2, 2) *= math::tanh(dx / (INV_4 * dx_abs));

        } else {
          raise::KernelError(
            HERE,
            "gr::AbsorbCurrents_kernel: 2D implementation called for D != 2");
        }
      }
    };
  } // namespace gr

} // namespace kernel::bc

#endif // KERNELS_FIELDS_BCS_HPP

/**
 * @file archetypes/piston.h
 * @brief Piston functions for implementing the piston in the CustomPrtlUpdate in pgran
 * @implements
 *   - arch::PistonUpdate<> -> void
 *   - arch::CrossesPiston<> -> Bool
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_WINDOW_H
#define ARCHETYPES_WINDOW_H

#include "enums.h"
#include "global.h"

#include "archetypes/energy_dist.h"
#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "framework/domain/metadomain.h"

#include <utility>


namespace arch {

  /**
   * @brief Updates particle position and fields in the moving window.

   */
  template <SimEngine::type S, class M>
  Inline void MoveWindow(dir::direction_t<M::Dim> direction,
                         Domain<S, M>& domain, 
                         const int window_shift) {
    
    if constexpr (M::CoordType != Coord::Cart) {
        (void)direction;
        (void)domain;
        (void)tags;
        raise::Error(
          "Moving window only applicable to cartesian coordinates",
          HERE);
      } else {

        const auto sign = direction.get_sign();
        const auto dim  = direction.get_dim();

        /*
          move particles in the window back by the window size
        */
        const auto& mesh = domain.mesh;

        // loop over particle species
        for (auto s { 0u }; s < 2; ++s) {
          // get particle properties
          auto& species = domain.species[s];
          auto  i1      = species.i1;

          Kokkos::parallel_for(
            "MoveParticles",
            species.rangeActiveParticles(),
            Lambda(index_t p) {
              // shift particle position back by window update frequency
              i1(p) -= window_shift;
            });
        }

        // shift fields in the window back by the window size
        std::vector<std::size_t> xi_min, xi_max;
        const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
        for (auto d { 0u }; d < M::Dim; ++d) {
          const auto dd = all_dirs[d];
          if (dim == dd) {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd) - window_shift);
          } else {
            xi_min.push_back(0);
            xi_max.push_back(domain.mesh.n_all(dd));
          }
        }
        raise::ErrorIf(xi_min.size() != xi_max.size() or
                         xi_min.size() != static_cast<std::size_t>(M::Dim),
                       "Invalid range size",
                       HERE);

        // loop range for shifting fields
        range_t<M::Dim> range;
        if constexpr (M::Dim == Dim::_1D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0] }, { xi_max[0] });
        } else if constexpr (M::Dim == Dim::_2D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1] },
                                            { xi_max[0], xi_max[1] });
        } else if constexpr (M::Dim == Dim::_3D) {
          range = CreateRangePolicy<M::Dim>({ xi_min[0], xi_min[1], xi_min[2] },
                                            { xi_max[0], xi_max[1], xi_max[2] });
        } else {
          raise::Error("Invalid dimension", HERE);
        }

        if (dir == in::x1) {
          Kokkos::parallel_for(
              "ShiftFields",
              range,
              FieldShift_kernel<M::Dim, in::x1>(
                domain.fields.em,
                window_shift,
                tags));
        } else if (dir == in::x2) {
          if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
            Kokkos::parallel_for(
              "ShiftFields",
              range,
              FieldShift_kernel<M::Dim, in::x2>(
                domain.fields.em,
                window_shift,
                tags));
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        } else {
          if constexpr (M::Dim == Dim::_3D) {
            Kokkos::parallel_for(
              "ShiftFields",
              range,
              FieldShift_kernel<M::Dim, in::x3>(
                domain.fields.em,
                window_shift,
                tags));
          } else {
            raise::Error("Invalid dimension", HERE);
          }
        }

      }
  }

  template <Dimension D, in o>
  struct FieldShift_kernel {
    static_assert(static_cast<dim_t>(o) < static_cast<dim_t>(D),
                  "Invalid component index");

    ndfield_t<D, 6>   Fld;
    const BCTags      tags;
    const int         window_shift;

    FieldShift_kernel(ndfield_t<D, 6> Fld, const int window_shift, BCTags tags)
      : Fld { Fld }
      , window_shift { window_shift }
      , tags { tags } {}

    Inline void operator()(index_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (tags & BC::E) {
          Fld(i1, em::ex1) = Fld(i1 + window_shift, em::ex1);
          Fld(i1, em::ex2) = Fld(i1 + window_shift, em::ex2);
          Fld(i1, em::ex3) = Fld(i1 + window_shift, em::ex3);
        }
        if (tags & BC::B) {
          Fld(i1, em::bx1) = Fld(i1 + window_shift, em::bx1);
          Fld(i1, em::bx2) = Fld(i1 + window_shift, em::bx2);
          Fld(i1, em::bx3) = Fld(i1 + window_shift, em::bx3);
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(index_t i1, index_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            Fld(i1, i2, em::ex1) = Fld(i1 + window_shift, i2, em::ex1);
            Fld(i1, i2, em::ex2) = Fld(i1 + window_shift, i2, em::ex2);
            Fld(i1, i2, em::ex3) = Fld(i1 + window_shift, i2, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, em::bx1) = Fld(i1 + window_shift, i2, em::bx1);
            Fld(i1, i2, em::bx2) = Fld(i1 + window_shift, i2, em::bx2);
            Fld(i1, i2, em::bx3) = Fld(i1 + window_shift, i2, em::bx3);
          }
        } else if constexpr (o == in::x2) {
          if (tags & BC::E) {
            Fld(i1, i2, em::ex1) = Fld(i1, i2 + window_shift, em::ex1);
            Fld(i1, i2, em::ex2) = Fld(i1, i2 + window_shift, em::ex2);
            Fld(i1, i2, em::ex3) = Fld(i1, i2 + window_shift, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, em::bx1) = Fld(i1, i2 + window_shift, em::bx1);
            Fld(i1, i2, em::bx2) = Fld(i1, i2 + window_shift, em::bx2);
            Fld(i1, i2, em::bx3) = Fld(i1, i2 + window_shift, em::bx3);
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(index_t i1, index_t i2, index_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = Fld(i1 + window_shift, i2, i3, em::ex1);
            Fld(i1, i2, i3, em::ex2) = Fld(i1 + window_shift, i2, i3, em::ex2);
            Fld(i1, i2, i3, em::ex3) = Fld(i1 + window_shift, i2, i3, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = Fld(i1 + window_shift, i2, i3, em::bx1);
            Fld(i1, i2, i3, em::bx2) = Fld(i1 + window_shift, i2, i3, em::bx2);
            Fld(i1, i2, i3, em::bx3) = Fld(i1 + window_shift, i2, i3, em::bx3);
          }
        } else if constexpr (o == in::x2) {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = Fld(i1, i2 + window_shift, i3, em::ex1);
            Fld(i1, i2, i3, em::ex2) = Fld(i1, i2 + window_shift, i3, em::ex2);
            Fld(i1, i2, i3, em::ex3) = Fld(i1, i2 + window_shift, i3, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = Fld(i1, i2 + window_shift, i3, em::bx1);
            Fld(i1, i2, i3, em::bx2) = Fld(i1, i2 + window_shift, i3, em::bx2);
            Fld(i1, i2, i3, em::bx3) = Fld(i1, i2 + window_shift, i3, em::bx3);
          }
        } else {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = Fld(i1, i2, i3 + window_shift, em::ex1);
            Fld(i1, i2, i3, em::ex2) = Fld(i1, i2, i3 + window_shift, em::ex2);
            Fld(i1, i2, i3, em::ex3) = Fld(i1, i2, i3 + window_shift, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = Fld(i1, i2, i3 + window_shift, em::bx1);
            Fld(i1, i2, i3, em::bx2) = Fld(i1, i2, i3 + window_shift, em::bx2);
            Fld(i1, i2, i3, em::bx3) = Fld(i1, i2, i3 + window_shift, em::bx3);
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 3D implementation called for D != 3");
      }
    }
  };
} // namespace arch

#endif // ARCHETYPES_UTILS_H

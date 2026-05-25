/**
 * @file archetypes/moving_window.h
 * @brief Moving window functions for implementing the moving window in the CustomPostStep
 * @implements
 *   - arch::MoveWindow<> -> void
 * @namespaces:
 *   - arch::
 */

#ifndef ARCHETYPES_WINDOW_H
#define ARCHETYPES_WINDOW_H

#include "enums.h"
#include "global.h"

#include "framework/domain/domain.h"
#include "framework/domain/metadomain.h"

namespace arch {

  template <Dimension D, in o>
  struct FieldShift_kernel {
    static_assert(static_cast<dim_t>(o) < static_cast<dim_t>(D),
                  "Invalid component index");

    ndfield_t<D, 6> Fld;
    ndfield_t<D, 6> backup_Fld;
    const BCTags    tags;
    const int       window_shift;

    FieldShift_kernel(ndfield_t<D, 6>&       Fld,
                      const ndfield_t<D, 6>& backup_Fld,
                      int                    window_shift,
                      BCTags                 tags)
      : Fld { Fld }
      , backup_Fld { backup_Fld }
      , window_shift { window_shift }
      , tags { tags } {}

    Inline void operator()(cellidx_t i1) const {
      if constexpr (D == Dim::_1D) {
        if (tags & BC::E) {
          Fld(i1, em::ex1) = backup_Fld(i1 + window_shift, em::ex1);
          Fld(i1, em::ex2) = backup_Fld(i1 + window_shift, em::ex2);
          Fld(i1, em::ex3) = backup_Fld(i1 + window_shift, em::ex3);
        }
        if (tags & BC::B) {
          Fld(i1, em::bx1) = backup_Fld(i1 + window_shift, em::bx1);
          Fld(i1, em::bx2) = backup_Fld(i1 + window_shift, em::bx2);
          Fld(i1, em::bx3) = backup_Fld(i1 + window_shift, em::bx3);
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 1D implementation called for D != 1");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2) const {
      if constexpr (D == Dim::_2D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            Fld(i1, i2, em::ex1) = backup_Fld(i1 + window_shift, i2, em::ex1);
            Fld(i1, i2, em::ex2) = backup_Fld(i1 + window_shift, i2, em::ex2);
            Fld(i1, i2, em::ex3) = backup_Fld(i1 + window_shift, i2, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, em::bx1) = backup_Fld(i1 + window_shift, i2, em::bx1);
            Fld(i1, i2, em::bx2) = backup_Fld(i1 + window_shift, i2, em::bx2);
            Fld(i1, i2, em::bx3) = backup_Fld(i1 + window_shift, i2, em::bx3);
          }
        } else if constexpr (o == in::x2) {
          if (tags & BC::E) {
            Fld(i1, i2, em::ex1) = backup_Fld(i1, i2 + window_shift, em::ex1);
            Fld(i1, i2, em::ex2) = backup_Fld(i1, i2 + window_shift, em::ex2);
            Fld(i1, i2, em::ex3) = backup_Fld(i1, i2 + window_shift, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, em::bx1) = backup_Fld(i1, i2 + window_shift, em::bx1);
            Fld(i1, i2, em::bx2) = backup_Fld(i1, i2 + window_shift, em::bx2);
            Fld(i1, i2, em::bx3) = backup_Fld(i1, i2 + window_shift, em::bx3);
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 2D implementation called for D != 2");
      }
    }

    Inline void operator()(cellidx_t i1, cellidx_t i2, cellidx_t i3) const {
      if constexpr (D == Dim::_3D) {
        if constexpr (o == in::x1) {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = backup_Fld(i1 + window_shift, i2, i3, em::ex1);
            Fld(i1, i2, i3, em::ex2) = backup_Fld(i1 + window_shift, i2, i3, em::ex2);
            Fld(i1, i2, i3, em::ex3) = backup_Fld(i1 + window_shift, i2, i3, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = backup_Fld(i1 + window_shift, i2, i3, em::bx1);
            Fld(i1, i2, i3, em::bx2) = backup_Fld(i1 + window_shift, i2, i3, em::bx2);
            Fld(i1, i2, i3, em::bx3) = backup_Fld(i1 + window_shift, i2, i3, em::bx3);
          }
        } else if constexpr (o == in::x2) {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = backup_Fld(i1, i2 + window_shift, i3, em::ex1);
            Fld(i1, i2, i3, em::ex2) = backup_Fld(i1, i2 + window_shift, i3, em::ex2);
            Fld(i1, i2, i3, em::ex3) = backup_Fld(i1, i2 + window_shift, i3, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = backup_Fld(i1, i2 + window_shift, i3, em::bx1);
            Fld(i1, i2, i3, em::bx2) = backup_Fld(i1, i2 + window_shift, i3, em::bx2);
            Fld(i1, i2, i3, em::bx3) = backup_Fld(i1, i2 + window_shift, i3, em::bx3);
          }
        } else {
          if (tags & BC::E) {
            Fld(i1, i2, i3, em::ex1) = backup_Fld(i1, i2, i3 + window_shift, em::ex1);
            Fld(i1, i2, i3, em::ex2) = backup_Fld(i1, i2, i3 + window_shift, em::ex2);
            Fld(i1, i2, i3, em::ex3) = backup_Fld(i1, i2, i3 + window_shift, em::ex3);
          }
          if (tags & BC::B) {
            Fld(i1, i2, i3, em::bx1) = backup_Fld(i1, i2, i3 + window_shift, em::bx1);
            Fld(i1, i2, i3, em::bx2) = backup_Fld(i1, i2, i3 + window_shift, em::bx2);
            Fld(i1, i2, i3, em::bx3) = backup_Fld(i1, i2, i3 + window_shift, em::bx3);
          }
        }
      } else {
        raise::KernelError(
          HERE,
          "FieldShift_kernel: 3D implementation called for D != 3");
      }
    }
  };

  /**
   * @brief Updates particle position and fields in the moving window.

   */
  template <CartesianMetricClass M, in o>
  inline void MoveWindow(Domain<SimEngine::SRPIC, M>&     domain,
                         Metadomain<SimEngine::SRPIC, M>& metadomain,
                         int                              window_shift) {

    /*
      move particles in the window back by the window size
    */
    const auto  nspec = domain.species.size();
    const auto& mesh  = domain.mesh;
    if (o == in::x1) {
      // loop over particle species
      for (auto s { 0u }; s < nspec; ++s) {
        // get particle properties
        auto& species = domain.species[s];
        auto  i1      = species.i1;
        Kokkos::parallel_for(
          "MoveParticles",
          species.rangeActiveParticles(),
          Lambda(prtlidx_t p) {
            // shift particle position back by window update frequency
            i1(p) -= window_shift;
          });
      }
    } else if (o == in::x2) {
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        // loop over particle species
        for (auto s { 0u }; s < nspec; ++s) {
          // get particle properties
          auto& species = domain.species[s];
          auto  i2      = species.i2;
          Kokkos::parallel_for(
            "MoveParticles",
            species.rangeActiveParticles(),
            Lambda(prtlidx_t p) {
              // shift particle position back by window update frequency
              i2(p) -= window_shift;
            });
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else if (o == in::x3) {
      if constexpr (M::Dim == Dim::_3D) {
        // loop over particle species
        for (auto s { 0u }; s < nspec; ++s) {
          // get particle properties
          auto& species = domain.species[s];
          auto  i3      = species.i3;
          Kokkos::parallel_for(
            "MoveParticles",
            species.rangeActiveParticles(),
            Lambda(prtlidx_t p) {
              // shift particle position back by window update frequency
              i3(p) -= window_shift;
            });
        }
      } else {
        raise::Error("Invalid direction", HERE);
      }
    } else {
      raise::Error("Invalid direction", HERE);
    }

    // shift fields in the window back by the window size
    std::vector<ncells_t> xi_min, xi_max;
    const std::vector<in> all_dirs { in::x1, in::x2, in::x3 };
    for (auto d { 0u }; d < M::Dim; ++d) {
      const auto dd = all_dirs[d];
      if (o == dd) {
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

    // copy fields to backup before shifting
    Kokkos::deep_copy(domain.fields.bckp, domain.fields.em);

    if (o == in::x1) {
      Kokkos::parallel_for("ShiftFields",
                           range,
                           FieldShift_kernel<M::Dim, in::x1>(domain.fields.em,
                                                             domain.fields.bckp,
                                                             window_shift,
                                                             BC::B | BC::E));
    } else if (o == in::x2) {
      if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
        Kokkos::parallel_for("ShiftFields",
                             range,
                             FieldShift_kernel<M::Dim, in::x2>(domain.fields.em,
                                                               domain.fields.bckp,
                                                               window_shift,
                                                               BC::B | BC::E));
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    } else {
      if constexpr (M::Dim == Dim::_3D) {
        Kokkos::parallel_for("ShiftFields",
                             range,
                             FieldShift_kernel<M::Dim, in::x3>(domain.fields.em,
                                                               domain.fields.bckp,
                                                               window_shift,
                                                               BC::B | BC::E));
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    }

    // synch ghost zones after moving the window
    metadomain.CommunicateFields(domain, Comm::EM);
    // communicate particles after moving
    metadomain.CommunicateParticles(domain);

    metadomain.ShiftByCells(window_shift, o);
  }
} // namespace arch

#endif // ARCHETYPES_UTILS_H

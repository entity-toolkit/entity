/**
 * @file engines/grpic/fields_bcs.h
 * @brief Field boundary condition routines for the GRPIC engine
 * @implements
 *   - enum ntt::grpic::gr_bc
 *   - ntt::grpic::MatchFieldsIn<> -> void
 *   - ntt::grpic::HorizonFieldsIn<> -> void
 *   - ntt::grpic::AxisFieldsIn<> -> void
 *   - ntt::grpic::CustomFieldsIn<> -> void
 *   - ntt::grpic::FieldBoundaries<> -> void
 * @namespaces:
 *   - ntt::grpic::
 */

#ifndef ENGINES_GRPIC_FIELDS_BCS_H
#define ENGINES_GRPIC_FIELDS_BCS_H

#include "enums.h"
#include "global.h"

#include "traits/metric.h"
#include "traits/pgen.h"

#include "framework/domain/domain.h"
#include "framework/parameters/parameters.h"
#include "kernels/fields_bcs.hpp"

#include "engines/engine.hpp"

#include <cstdint>

namespace ntt {
  namespace grpic {

    enum class gr_bc : uint8_t {
      main,
      aux,
      curr
    };

    template <GRMetricClass M, PGenClass<SimEngine::GRPIC, M> PG>
    void MatchFieldsIn(dir::direction_t<M::Dim>     direction,
                       Domain<SimEngine::GRPIC, M>& domain,
                       const Grid<M::Dim>&          global_grid,
                       const PG&                    pgen,
                       const SimulationParams&      params,
                       BCTags                       tags,
                       const gr_bc&                 g) {
      /**
       * match boundaries
       */
      const auto ds_array = params.template get<boundaries_t<real_t>>(
        "grid.boundaries.match.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      auto       sign = direction.get_sign();

      raise::ErrorIf(((dim != in::x1) or (sign < 0)) and (g == gr_bc::curr),
                     "Absorption of currents only possible in +x1 (+r)",
                     HERE);

      real_t ds;
      if (sign > 0) { // + direction
        ds      = ds_array[(short)dim].second;
        xg_max  = global_grid.extent(dim).second;
        xg_min  = xg_max - ds;
        xg_edge = xg_max;
      } else { // - direction
        ds      = ds_array[(short)dim].first;
        xg_min  = global_grid.extent(dim).first;
        xg_max  = xg_min + ds;
        xg_edge = xg_min;
      }
      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<unsigned short>(dim)) {
          box.emplace_back(xg_min, xg_max);
          incl_ghosts.emplace_back(false, true);
        } else {
          box.push_back(Range::All);
          incl_ghosts.emplace_back(true, true);
        }
      }
      if (not domain.mesh.Intersects(box)) {
        return;
      }
      const auto intersect_range = domain.mesh.ExtentToRange(box, incl_ghosts);
      tuple_t<std::size_t, M::Dim> range_min { 0 };
      tuple_t<std::size_t, M::Dim> range_max { 0 };

      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        range_min[d] = intersect_range[d].first;
        range_max[d] = intersect_range[d].second;
      }
      if (dim == in::x1) {
        if (g != gr_bc::curr) {
          if constexpr (::traits::pgen::HasInitFlds<PG>) {
            Kokkos::parallel_for(
              "MatchBoundaries",
              CreateRangePolicy<M::Dim>(range_min, range_max),
              kernel::bc::MatchBoundaries_kernel<SimEngine::GRPIC,
                                                 M,
                                                 decltype(pgen.init_flds),
                                                 in::x1>(domain.fields.em,
                                                         pgen.init_flds,
                                                         domain.mesh.metric,
                                                         xg_edge,
                                                         ds,
                                                         tags,
                                                         domain.mesh.flds_bc()));
            Kokkos::parallel_for(
              "MatchBoundaries",
              CreateRangePolicy<M::Dim>(range_min, range_max),
              kernel::bc::MatchBoundaries_kernel<SimEngine::GRPIC,
                                                 M,
                                                 decltype(pgen.init_flds),
                                                 in::x1>(domain.fields.em0,
                                                         pgen.init_flds,
                                                         domain.mesh.metric,
                                                         xg_edge,
                                                         ds,
                                                         tags,
                                                         domain.mesh.flds_bc()));
          }
        } else {
          Kokkos::parallel_for(
            "AbsorbCurrents",
            CreateRangePolicy<M::Dim>(range_min, range_max),
            kernel::bc::gr::AbsorbCurrents_kernel<M, 1>(domain.fields.cur0,
                                                        domain.mesh.metric,
                                                        xg_edge,
                                                        ds));
        }
      } else {
        raise::Error("Invalid dimension", HERE);
      }
    }

    template <GRMetricClass M>
    void HorizonFieldsIn(dir::direction_t<M::Dim>     direction,
                         Domain<SimEngine::GRPIC, M>& domain,
                         const SimulationParams&      params,
                         BCTags                       tags,
                         const gr_bc&                 g) {
      /**
       * open boundaries
       */
      raise::ErrorIf(M::CoordType == Coord::Cartesian,
                     "Invalid coordinate type for horizon BCs",
                     HERE);
      raise::ErrorIf(direction.get_dim() != in::x1,
                     "Invalid horizon direction, should be x1",
                     HERE);
      const auto i1_min = domain.mesh.i_min(in::x1);
      auto range = CreateRangePolicy<Dim::_1D>({ domain.mesh.i_min(in::x2) },
                                               { domain.mesh.i_max(in::x2) + 1 });
      const auto nfilter = params.template get<unsigned short>(
        "algorithms.current_filters");
      if (g == gr_bc::main) {
        Kokkos::parallel_for(
          "OpenBCFields",
          range,
          kernel::bc::gr::HorizonBoundaries_kernel<M::Dim>(domain.fields.em,
                                                           i1_min,
                                                           tags,
                                                           nfilter));
        Kokkos::parallel_for(
          "OpenBCFields",
          range,
          kernel::bc::gr::HorizonBoundaries_kernel<M::Dim>(domain.fields.em0,
                                                           i1_min,
                                                           tags,
                                                           nfilter));
      }
    }

    template <GRMetricClass M>
    void AxisFieldsIn(dir::direction_t<M::Dim>     direction,
                      Domain<SimEngine::GRPIC, M>& domain,
                      BCTags                       tags) {
      /**
       * axis boundaries
       */
      raise::ErrorIf(M::CoordType == Coord::Cartesian,
                     "Invalid coordinate type for axis BCs",
                     HERE);
      raise::ErrorIf(direction.get_dim() != in::x2,
                     "Invalid axis direction, should be x2",
                     HERE);
      const auto i2_min = domain.mesh.i_min(in::x2);
      const auto i2_max = domain.mesh.i_max(in::x2);
      if (direction.get_sign() < 0) {
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::bc::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em,
                                                           i2_min,
                                                           tags));
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::bc::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em0,
                                                           i2_min,
                                                           tags));
      } else {
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::bc::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em,
                                                          i2_max,
                                                          tags));
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::bc::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em0,
                                                          i2_max,
                                                          tags));
      }
    }

    template <GRMetricClass M>
    void CustomFieldsIn(dir::direction_t<M::Dim>     direction,
                        Domain<SimEngine::GRPIC, M>& domain,
                        BCTags                       tags,
                        const gr_bc&                 g) {
      (void)direction;
      (void)domain;
      (void)tags;
      (void)g;
      raise::Error("Custom boundaries not implemented", HERE);
    }

    template <GRMetricClass M, PGenClass<SimEngine::GRPIC, M> PG>
    void FieldBoundaries(Domain<SimEngine::GRPIC, M>& domain,
                         const Grid<M::Dim>&          global_grid,
                         const PG&                    pgen,
                         const SimulationParams&      params,
                         BCTags                       tags,
                         const gr_bc&                 g) {
      if (g == gr_bc::main) {
        for (auto& direction : dir::Directions<M::Dim>::orth) {
          if (global_grid.flds_bc_in(direction) == FldsBC::MATCH) {
            MatchFieldsIn<M, PG>(direction, domain, global_grid, pgen, params, tags, g);
          } else if (domain.mesh.flds_bc_in(direction) == FldsBC::AXIS) {
            AxisFieldsIn<M>(direction, domain, tags);
          } else if (global_grid.flds_bc_in(direction) == FldsBC::CUSTOM) {
            CustomFieldsIn<M>(direction, domain, tags, g);
          } else if (domain.mesh.flds_bc_in(direction) == FldsBC::HORIZON) {
            HorizonFieldsIn<M>(direction, domain, params, tags, g);
          }
        } // loop over directions
      } else if (g == gr_bc::aux) {
        for (auto& direction : dir::Directions<M::Dim>::orth) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::HORIZON) {
            HorizonFieldsIn<M>(direction, domain, params, tags, g);
          }
        }
      } else if (g == gr_bc::curr) {
        for (auto& direction : dir::Directions<M::Dim>::orth) {
          if (domain.mesh.prtl_bc_in(direction) == PrtlBC::ABSORB) {
            MatchFieldsIn<M, PG>(direction, domain, global_grid, pgen, params, tags, g);
          }
        }
      }
    }

  } // namespace grpic
} // namespace ntt

#endif // ENGINES_GRPIC_FIELDS_BCS_H
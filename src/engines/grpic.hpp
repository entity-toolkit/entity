/**
 * @file engines/grpic.hpp
 * @brief Simulation engien class which specialized on GRPIC
 * @implements
 *   - ntt::GRPICEngine<> : ntt::Engine<>
 * @cpp:
 *   - grpic.cpp
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_GRPIC_GRPIC_H
#define ENGINES_GRPIC_GRPIC_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/traits.h"
#include "utils/log.h"
#include "utils/numeric.h"
#include "utils/timer.h"

#include "archetypes/particle_injector.h"
#include "framework/domain/domain.h"
#include "framework/parameters.h"

#include "engines/engine.hpp"

#include "kernels/ampere_gr.hpp"
#include "kernels/aux_fields_gr.hpp"
#include "kernels/currents_deposit.hpp"
#include "kernels/digital_filter.hpp"
#include "kernels/faraday_gr.hpp"
#include "kernels/fields_bcs.hpp"
#include "kernels/particle_moments.hpp"
#include "kernels/particle_pusher_gr.hpp"

#include "pgen.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>
#include <utility>

namespace ntt {

  template <class M>
  class GRPICEngine : public Engine<SimEngine::GRPIC, M> {
    using base_t   = Engine<SimEngine::GRPIC, M>;
    using pgen_t   = user::PGen<SimEngine::GRPIC, M>;
    using domain_t = Domain<SimEngine::GRPIC, M>;
    // constexprs
    using base_t::pgen_is_ok;
    // contents
    using base_t::m_metadomain;
    using base_t::m_params;
    using base_t::m_pgen;
    // methods
    using base_t::init;
    // variables
    using base_t::dt;
    using base_t::max_steps;
    using base_t::runtime;
    using base_t::step;
    using base_t::time;

  public:
    static constexpr auto S { SimEngine::GRPIC };

    GRPICEngine(SimulationParams& params) : base_t { params } {}

    ~GRPICEngine() = default;

    void step_forward(timer::Timers& timers, domain_t& dom) override {
      const auto fieldsolver_enabled = m_params.template get<bool>(
        "algorithms.toggles.fieldsolver");
      const auto deposit_enabled = m_params.template get<bool>(
        "algorithms.toggles.deposit");
      const auto sort_interval = m_params.template get<std::size_t>(
        "particles.sort_interval");

      if (step == 0) {
        // communicate fields and apply BCs on the first timestep
        /**
        * Initially: em0::B   --
        *            em0::D   --
        *            em::B    at -1/2
        *            em::D    at -1/2
        *
        *            cur0::J  --
        *            cur::J   --
        *
        *            aux::E   --
        *            aux::H   --
        *
        *            x_prtl   at -1/2
        *            u_prtl   at -1/2
        */

        /**
        * em0::D, em::D, em0::B, em::B <- boundary conditions
        */
        m_metadomain.CommunicateFields(dom, Comm::B | Comm::B0 | Comm::D | Comm::D0);
        FieldBoundaries(dom, BC::B | BC::D);

      }
    }

    void FieldBoundaries(domain_t& domain, BCTags tags) {
      for (auto& direction : dir::Directions<M::Dim>::orth) {
        if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::ABSORB) {
          AbsorbFieldsIn(direction, domain, tags);
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::AXIS) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::AXIS) {
            AxisFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::CUSTOM) {
          if (domain.mesh.flds_bc_in(direction) == FldsBC::CUSTOM) {
            CustomFieldsIn(direction, domain, tags);
          }
        } else if (m_metadomain.mesh().flds_bc_in(direction) == FldsBC::HORIZON) {
          raise::Error("HORIZON BCs only applicable for GR", HERE);
        }
      } // loop over directions
    }

    void AbsorbFieldsIn(dir::direction_t<M::Dim> direction,
                        domain_t&                domain,
                        BCTags                   tags) {
      /**
       * absorbing boundaries
       */
      const auto ds = m_params.template get<real_t>(
        "grid.boundaries.absorb.ds");
      const auto dim = direction.get_dim();
      real_t     xg_min, xg_max, xg_edge;
      auto       sign = direction.get_sign();
      if (sign > 0) { // + direction
        xg_max  = m_metadomain.mesh().extent(dim).second;
        xg_min  = xg_max - ds;
        xg_edge = xg_max;
      } else { // - direction
        xg_min  = m_metadomain.mesh().extent(dim).first;
        xg_max  = xg_min + ds;
        xg_edge = xg_min;
      }
      boundaries_t<real_t> box;
      boundaries_t<bool>   incl_ghosts;
      for (unsigned short d { 0 }; d < M::Dim; ++d) {
        if (d == static_cast<unsigned short>(dim)) {
          box.push_back({ xg_min, xg_max });
          if (sign > 0) {
            incl_ghosts.push_back({ false, true });
          } else {
            incl_ghosts.push_back({ true, false });
          }
        } else {
          box.push_back(Range::All);
          incl_ghosts.push_back({ true, true });
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
        Kokkos::parallel_for(
          "AbsorbFields",
          CreateRangePolicy<M::Dim>(range_min, range_max),
          kernel::AbsorbBoundaries_kernel<M, 1>(domain.fields.em,
                                                domain.mesh.metric,
                                                xg_edge,
                                                ds,
                                                tags));
      } else if (dim == in::x2) {
        if constexpr (M::Dim == Dim::_2D or M::Dim == Dim::_3D) {
          Kokkos::parallel_for(
            "AbsorbFields",
            CreateRangePolicy<M::Dim>(range_min, range_max),
            kernel::AbsorbBoundaries_kernel<M, 2>(domain.fields.em,
                                                  domain.mesh.metric,
                                                  xg_edge,
                                                  ds,
                                                  tags));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      } else if (dim == in::x3) {
        if constexpr (M::Dim == Dim::_3D) {
          Kokkos::parallel_for(
            "AbsorbFields",
            CreateRangePolicy<M::Dim>(range_min, range_max),
            kernel::AbsorbBoundaries_kernel<M, 3>(domain.fields.em,
                                                  domain.mesh.metric,
                                                  xg_edge,
                                                  ds,
                                                  tags));
        } else {
          raise::Error("Invalid dimension", HERE);
        }
      }
    }

    void AxisFieldsIn(dir::direction_t<M::Dim> direction,
                      domain_t&                domain,
                      BCTags                   tags) {
      /**
       * axis boundaries
       */
      raise::ErrorIf(M::CoordType == Coord::Cart,
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
          kernel::AxisBoundaries_kernel<M::Dim, false>(domain.fields.em, i2_min, tags));
      } else {
        Kokkos::parallel_for(
          "AxisBCFields",
          domain.mesh.n_all(in::x1),
          kernel::AxisBoundaries_kernel<M::Dim, true>(domain.fields.em, i2_max, tags));
      }
    }

    void CustomFieldsIn(dir::direction_t<M::Dim> direction,
                        domain_t&                domain,
                        BCTags                   tags) {
      (void)direction;
      (void)domain;
      (void)tags;
      raise::Error("Custom boundaries not implemented", HERE);
      // if constexpr (
      //   traits::has_member<traits::pgen::custom_fields_t, pgen_t>::value) {
      //   const auto [box, custom_fields] = m_pgen.CustomFields(time);
      //   if (domain.mesh.Intersects(box)) {
      //   }
      //
      // } else {
      //   raise::Error("Custom boundaries not implemented", HERE);
      // }
    }

  };
} // namespace ntt

#endif // ENGINES_GRPIC_GRPIC_H

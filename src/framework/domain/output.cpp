#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/log.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "framework/containers/particles.h"
#include "framework/domain/metadomain.h"
#include "framework/parameters.h"

#include "kernels/fields_to_phys.hpp"
#include "kernels/particle_moments.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <vector>

namespace ntt {

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::InitWriter(const SimulationParams& params) {
    raise::ErrorIf(
      local_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(local_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);

    const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");

    auto glob_shape_with_ghosts = mesh().n_active();
    auto off_ncells_with_ghosts = local_domain->offset_ncells();
    auto off_ndomains           = local_domain->offset_ndomains();
    auto loc_shape_with_ghosts  = local_domain->mesh.n_active();
    if (incl_ghosts) {
      for (auto d { 0 }; d <= M::Dim; ++d) {
        glob_shape_with_ghosts[d] += 2 * N_GHOSTS * ndomains_per_dim()[d];
        off_ncells_with_ghosts[d] += 2 * N_GHOSTS * off_ndomains[d];
        loc_shape_with_ghosts[d]  += 2 * N_GHOSTS;
      }
    }

    g_writer.defineMeshLayout(glob_shape_with_ghosts,
                              off_ncells_with_ghosts,
                              loc_shape_with_ghosts,
                              incl_ghosts,
                              M::CoordType);
    const auto fields_to_write = params.template get<std::vector<std::string>>(
      "output.fields");
    g_writer.defineFieldOutputs(S, fields_to_write);
  }

  template <SimEngine::type S, class M, FldsID::type F>
  void ComputeMoments(const SimulationParams& params,
                      const Mesh<M>&          mesh,
                      const std::vector<Particles<M::Dim, M::CoordType>>& prtl_species,
                      const std::vector<unsigned short>& species,
                      const std::vector<unsigned short>& components,
                      ndfield_t<M::Dim, 6>&              buffer,
                      unsigned short                     buff_idx) {
    std::vector<unsigned short> specs = species;
    if (specs.size() == 0) {
      // if no species specific, take all massive species
      for (auto& sp : prtl_species) {
        if (sp.mass() > 0 && sp.charge() > 0) {
          specs.push_back(sp.index());
        }
      }
    }
    // replace species indexes with positions in prtl_species
    for (auto& sp : specs) {
      unsigned short idx = 0;
      for (auto& prtl_sp : prtl_species) {
        if (prtl_sp.index() == sp) {
          sp = idx;
          break;
        }
        ++idx;
      }
    }
    auto scatter_buff = Kokkos::Experimental::create_scatter_view(buffer);

    // some parameters
    const auto use_weights = params.template get<bool>("particles.use_weights");
    const auto ni2         = mesh.n_active(in::x2);
    const auto inv_n0      = ONE / params.template get<real_t>("scales.n0");
    const auto window      = params.template get<unsigned short>(
      "output.mom_smooth");

    for (const auto& sp : specs) {
      auto& prtl_spec = prtl_species[sp];
      kernel::ParticleMoments_kernel<S, M, F, 6>(components,
                                                 scatter_buff,
                                                 buff_idx,
                                                 prtl_spec.i1,
                                                 prtl_spec.i2,
                                                 prtl_spec.i3,
                                                 prtl_spec.dx1,
                                                 prtl_spec.dx2,
                                                 prtl_spec.dx3,
                                                 prtl_spec.ux1,
                                                 prtl_spec.ux2,
                                                 prtl_spec.ux3,
                                                 prtl_spec.phi,
                                                 prtl_spec.weight,
                                                 prtl_spec.tag,
                                                 prtl_spec.mass(),
                                                 prtl_spec.charge(),
                                                 use_weights,
                                                 mesh.metric,
                                                 mesh.flds_bc(),
                                                 ni2,
                                                 inv_n0,
                                                 window);
    }
    Kokkos::Experimental::contribute(buffer, scatter_buff);
  }

  template <Dimension D, int N, int M>
  void DeepCopyFields(ndfield_t<D, N>&     fld_from,
                      ndfield_t<D, M>&     fld_to,
                      const range_tuple_t& from,
                      const range_tuple_t& to) {
    for (unsigned short d = 0; d < D; ++d) {
      raise::ErrorIf(fld_from.extent(d) != fld_to.extent(d),
                     "Fields have different sizes " +
                       std::to_string(fld_from.extent(d)) +
                       " != " + std::to_string(fld_to.extent(d)),
                     HERE);
    }
    if constexpr (D == Dim::_1D) {
      Kokkos::deep_copy(Kokkos::subview(fld_to, Kokkos::ALL, to),
                        Kokkos::subview(fld_from, Kokkos::ALL, from));
    } else if constexpr (D == Dim::_2D) {
      Kokkos::deep_copy(Kokkos::subview(fld_to, Kokkos::ALL, Kokkos::ALL, to),
                        Kokkos::subview(fld_from, Kokkos::ALL, Kokkos::ALL, from));
    } else if constexpr (D == Dim::_3D) {
      Kokkos::deep_copy(
        Kokkos::subview(fld_to, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, to),
        Kokkos::subview(fld_from, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, from));
    }
  }

  template <SimEngine::type S, class M>
  void Metadomain<S, M>::Write(const SimulationParams& params,
                               const std::string&      fname,
                               std::size_t             step,
                               long double             time) {
    raise::ErrorIf(
      local_subdomain_indices().size() != 1,
      "Output for now is only supported for one subdomain per rank",
      HERE);
    auto local_domain = subdomain_ptr(local_subdomain_indices()[0]);
    raise::ErrorIf(local_domain->is_placeholder(),
                   "local_domain is a placeholder",
                   HERE);
    logger::Checkpoint("Writing output", HERE);
    g_writer.beginWriting(fname, step, time);

    const auto incl_ghosts = params.template get<bool>("output.debug.ghosts");

    for (unsigned short dim = 0; dim < M::Dim; ++dim) {
      const auto is_last = local_domain->offset_ncells()[dim] +
                             local_domain->mesh.n_active()[dim] ==
                           mesh().n_active()[dim];
      array_t<real_t*> xc { "Xc",
                            local_domain->mesh.n_active()[dim] +
                              (incl_ghosts ? 2 * N_GHOSTS : 0) };
      array_t<real_t*> xe { "Xe",
                            local_domain->mesh.n_active()[dim] +
                              (incl_ghosts ? 2 * N_GHOSTS : 0) +
                              (is_last ? 1 : 0) };
      const auto       offset = (incl_ghosts ? N_GHOSTS : 0);
      const auto       ncells = local_domain->mesh.n_active()[dim];
      const auto&      metric = g_mesh.metric;
      Kokkos::parallel_for(
        "GenerateMesh",
        ncells,
        Lambda(index_t i) {
          const auto      i_ = static_cast<real_t>(i);
          coord_t<M::Dim> x_Cd { ZERO }, x_Ph { ZERO };
          x_Cd[dim] = i_ + HALF;
          metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
          xc(offset + i) = x_Ph[dim];
          x_Cd[dim]      = i_;
          metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
          xe(offset + i) = x_Ph[dim];
          if (is_last && i == ncells - 1) {
            x_Cd[dim] = i_ + ONE;
            metric.template convert<Crd::Cd, Crd::Ph>(x_Cd, x_Ph);
            xe(offset + i + 1) = x_Ph[dim];
          }
        });
      g_writer.writeMesh(dim, xc, xe);
    }

    const auto output_asis = params.template get<bool>("output.debug.as_is");
    // !TODO: this can probably be optimized to dump things at once
    for (auto& fld : g_writer.fieldWriters()) {
      std::vector<std::string> names;
      std::vector<std::size_t> addresses;
      if (fld.comp.size() == 0 || fld.comp.size() == 1) { // scalar
        names.push_back(fld.name());
        addresses.push_back(0);
        if (fld.is_moment()) {
          // output a particle distribution moment (single component)
          // this includes T, Rho, Charge, N, Nppc
          const auto c = static_cast<unsigned short>(addresses.back());
          if (fld.id() == FldsID::T) {
            raise::ErrorIf(fld.comp.size() != 1,
                           "Wrong # of components requested for T output",
                           HERE);
            ComputeMoments<S, M, FldsID::T>(params,
                                            local_domain->mesh,
                                            local_domain->species,
                                            fld.species,
                                            fld.comp[0],
                                            local_domain->fields.bckp,
                                            c);
          } else if (fld.id() == FldsID::Rho) {
            ComputeMoments<S, M, FldsID::Rho>(params,
                                              local_domain->mesh,
                                              local_domain->species,
                                              fld.species,
                                              {},
                                              local_domain->fields.bckp,
                                              c);
          } else if (fld.id() == FldsID::Charge) {
            ComputeMoments<S, M, FldsID::Charge>(params,
                                                 local_domain->mesh,
                                                 local_domain->species,
                                                 fld.species,
                                                 {},
                                                 local_domain->fields.bckp,
                                                 c);
          } else if (fld.id() == FldsID::N) {
            ComputeMoments<S, M, FldsID::N>(params,
                                            local_domain->mesh,
                                            local_domain->species,
                                            fld.species,
                                            {},
                                            local_domain->fields.bckp,
                                            c);
          } else if (fld.id() == FldsID::Nppc) {
            ComputeMoments<S, M, FldsID::Nppc>(params,
                                               local_domain->mesh,
                                               local_domain->species,
                                               fld.species,
                                               {},
                                               local_domain->fields.bckp,
                                               c);
          } else {
            raise::Error("Wrong moment requested for output", HERE);
          }
        } else {
          raise::Error("Wrong # of components requested for non-moment output",
                       HERE);
        }
      } else if (fld.comp.size() == 3) { // vector
        for (auto i = 0; i < 3; ++i) {
          names.push_back(fld.name(i));
          addresses.push_back(i + 3);
        }
        if (fld.is_moment()) {
          for (auto i = 0; i < 3; ++i) {
            const auto c = static_cast<unsigned short>(addresses[i]);
            raise::ErrorIf(fld.comp[i].size() != 2,
                           "Wrong # of components requested for moment",
                           HERE);
            ComputeMoments<S, M, FldsID::T>(params,
                                            local_domain->mesh,
                                            local_domain->species,
                                            fld.species,
                                            fld.comp[i],
                                            local_domain->fields.bckp,
                                            c);
          }
        } else {
          // copy fields to bckp (:, 0, 1, 2)
          // if as-is specified ==> copy directly to 3, 4, 5
          range_tuple_t copy_to = { 0, 3 };
          if (output_asis) {
            copy_to = { 3, 6 };
          }
          if (fld.is_current()) {
            DeepCopyFields<M::Dim, 3, 6>(local_domain->fields.cur,
                                         local_domain->fields.bckp,
                                         { cur::jx1, cur::jx3 + 1 },
                                         copy_to);
          } else if (fld.is_field()) {
            if (S == SimEngine::GRPIC && fld.is_gr_aux_field()) {
              if (fld.is_efield()) {
                // GR: E
                DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.aux,
                                             local_domain->fields.bckp,
                                             { em::ex1, em::ex3 + 1 },
                                             copy_to);
              } else {
                // GR: H
                DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.aux,
                                             local_domain->fields.bckp,
                                             { em::hx1, em::hx3 + 1 },
                                             copy_to);
              }
            } else {
              if (fld.is_efield()) {
                // GR/SR: D/E
                DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.em,
                                             local_domain->fields.bckp,
                                             { em::ex1, em::ex3 + 1 },
                                             copy_to);
              } else {
                // GR/SR: B
                DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.em,
                                             local_domain->fields.bckp,
                                             { em::bx1, em::bx3 + 1 },
                                             copy_to);
              }
            }
          } else {
            raise::Error("Wrong field requested for output", HERE);
          }
          if (not output_asis) {
            // copy fields from bckp(:, 0, 1, 2) -> bckp(:, 3, 4, 5)
            // converting to proper basis and properly interpolating
            list_t<unsigned short, 3> comp_from = { 0, 1, 2 };
            list_t<unsigned short, 3> comp_to   = { 3, 4, 5 };
            DeepCopyFields<M::Dim, 6, 6>(local_domain->fields.bckp,
                                         local_domain->fields.bckp,
                                         { 0, 3 },
                                         { 3, 6 });
            Kokkos::parallel_for("FieldsToPhys",
                                 local_domain->mesh.rangeActiveCells(),
                                 kernel::FieldsToPhys_kernel<M, 6, 6>(
                                   local_domain->fields.bckp,
                                   local_domain->fields.bckp,
                                   comp_from,
                                   comp_to,
                                   fld.interp_flag | fld.prepare_flag,
                                   local_domain->mesh.metric));
          }
        }
      } else if (fld.comp.size() == 6) { // tensor
        raise::ErrorIf(not fld.is_moment() or fld.id() != FldsID::T,
                       "Only T tensor has 6 components",
                       HERE);
        for (auto i = 0; i < 6; ++i) {
          names.push_back(fld.name(i));
          addresses.push_back(i);
          const auto c = static_cast<unsigned short>(addresses.back());
          raise::ErrorIf(fld.comp[i].size() != 2,
                         "Wrong # of components requested for moment",
                         HERE);
          ComputeMoments<S, M, FldsID::T>(params,
                                          local_domain->mesh,
                                          local_domain->species,
                                          fld.species,
                                          fld.comp[i],
                                          local_domain->fields.bckp,
                                          c);
        }
      } else {
        raise::Error("Wrong # of components requested for output", HERE);
      }
      g_writer.writeField<M::Dim, 6>(names, local_domain->fields.bckp, addresses);
    }

    g_writer.endWriting();
  }

  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
  template struct Metadomain<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;

} // namespace ntt
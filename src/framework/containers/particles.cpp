#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/sorting.h"

#include "framework/containers/species.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>

#include <string>
#include <vector>

namespace ntt {
  template <Dimension D, Coord::type C>
  Particles<D, C>::Particles(unsigned short     index,
                             const std::string& label,
                             float              m,
                             float              ch,
                             std::size_t        maxnpart,
                             const PrtlPusher&  pusher,
                             bool               use_gca,
                             const Cooling&     cooling,
                             unsigned short     npld)
    : ParticleSpecies(index, label, m, ch, maxnpart, pusher, use_gca, cooling, npld) {
    i1    = array_t<int*> { label + "_i1", maxnpart };
    i1_h  = Kokkos::create_mirror_view(i1);
    dx1   = array_t<prtldx_t*> { label + "_dx1", maxnpart };
    dx1_h = Kokkos::create_mirror_view(dx1);

    i1_prev  = array_t<int*> { label + "_i1_prev", maxnpart };
    dx1_prev = array_t<prtldx_t*> { label + "_dx1_prev", maxnpart };

    ux1   = array_t<real_t*> { label + "_ux1", maxnpart };
    ux1_h = Kokkos::create_mirror_view(ux1);
    ux2   = array_t<real_t*> { label + "_ux2", maxnpart };
    ux2_h = Kokkos::create_mirror_view(ux2);
    ux3   = array_t<real_t*> { label + "_ux3", maxnpart };
    ux3_h = Kokkos::create_mirror_view(ux3);

    weight   = array_t<real_t*> { label + "_w", maxnpart };
    weight_h = Kokkos::create_mirror_view(weight);

    tag   = array_t<short*> { label + "_tag", maxnpart };
    tag_h = Kokkos::create_mirror_view(tag);

    for (unsigned short n { 0 }; n < npld; ++n) {
      pld.push_back(array_t<real_t*>("pld", maxnpart));
      pld_h.push_back(Kokkos::create_mirror_view(pld[n]));
    }

    if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
      i2    = array_t<int*> { label + "_i2", maxnpart };
      i2_h  = Kokkos::create_mirror_view(i2);
      dx2   = array_t<prtldx_t*> { label + "_dx2", maxnpart };
      dx2_h = Kokkos::create_mirror_view(dx2);

      i2_prev  = array_t<int*> { label + "_i2_prev", maxnpart };
      dx2_prev = array_t<prtldx_t*> { label + "_dx2_prev", maxnpart };
    }
    if ((D == Dim::_2D) && (C != Coord::Cart)) {
      phi   = array_t<real_t*> { label + "_phi", maxnpart };
      phi_h = Kokkos::create_mirror_view(phi);
    }

    if constexpr (D == Dim::_3D) {
      i3    = array_t<int*> { label + "_i3", maxnpart };
      i3_h  = Kokkos::create_mirror_view(i3);
      dx3   = array_t<prtldx_t*> { label + "_dx3", maxnpart };
      dx3_h = Kokkos::create_mirror_view(dx3);

      i3_prev  = array_t<int*> { label + "_i3_prev", maxnpart };
      dx3_prev = array_t<prtldx_t*> { label + "_dx3_prev", maxnpart };
    }
  }

  template <Dimension D, Coord::type C>
  auto Particles<D, C>::npart_per_tag() const -> std::vector<std::size_t> {
    auto                  this_tag = tag;
    array_t<std::size_t*> npart_tag("npart_tags", ntags());

    auto npart_tag_scatter = Kokkos::Experimental::create_scatter_view(npart_tag);
    Kokkos::parallel_for(
      "NpartPerTag",
      npart(),
      Lambda(index_t p) {
        auto npart_tag_scatter_access = npart_tag_scatter.access();
        npart_tag_scatter_access((int)(this_tag(p))) += 1;
      });
    Kokkos::Experimental::contribute(npart_tag, npart_tag_scatter);

    auto npart_tag_host = Kokkos::create_mirror_view(npart_tag);
    Kokkos::deep_copy(npart_tag_host, npart_tag);

    std::vector<std::size_t> npart_tag_vec;
    for (std::size_t t { 0 }; t < ntags(); ++t) {
      npart_tag_vec.push_back(npart_tag_host(t));
    }
    return npart_tag_vec;
  }

  template <Dimension D, Coord::type C>
  auto Particles<D, C>::SortByTags() -> std::vector<std::size_t> {
    if (npart() == 0 || is_sorted()) {
      return npart_per_tag();
    }
    using KeyType = array_t<short*>;
    using BinOp   = sort::BinTag<KeyType>;
    BinOp bin_op(ntags());
    auto  slice = range_tuple_t(0, npart());
    Kokkos::BinSort<KeyType, BinOp> Sorter(Kokkos::subview(tag, slice), bin_op, false);
    Sorter.create_permute_vector();

    Sorter.sort(Kokkos::subview(i1, slice));
    Sorter.sort(Kokkos::subview(dx1, slice));
    Sorter.sort(Kokkos::subview(i1_prev, slice));
    Sorter.sort(Kokkos::subview(dx1_prev, slice));
    Sorter.sort(Kokkos::subview(ux1, slice));
    Sorter.sort(Kokkos::subview(ux2, slice));
    Sorter.sort(Kokkos::subview(ux3, slice));

    Sorter.sort(Kokkos::subview(tag, slice));
    Sorter.sort(Kokkos::subview(weight, slice));

    for (unsigned short n { 0 }; n < npld(); ++n) {
      Sorter.sort(Kokkos::subview(pld[n], slice));
    }

    if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
      Sorter.sort(Kokkos::subview(i2, slice));
      Sorter.sort(Kokkos::subview(dx2, slice));

      Sorter.sort(Kokkos::subview(i2_prev, slice));
      Sorter.sort(Kokkos::subview(dx2_prev, slice));
    }
    if constexpr (D == Dim::_3D) {
      Sorter.sort(Kokkos::subview(i3, slice));
      Sorter.sort(Kokkos::subview(dx3, slice));

      Sorter.sort(Kokkos::subview(i3_prev, slice));
      Sorter.sort(Kokkos::subview(dx3_prev, slice));
    }

    if ((D == Dim::_2D) && (C != Coord::Cart)) {
      Sorter.sort(Kokkos::subview(phi, slice));
    }

    const auto np_per_tag = npart_per_tag();
    set_npart(np_per_tag[(short)(ParticleTag::alive)]);

    m_is_sorted = true;
    return np_per_tag;
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::SyncHostDevice() {
    Kokkos::deep_copy(i1_h, i1);
    Kokkos::deep_copy(dx1_h, dx1);
    Kokkos::deep_copy(ux1_h, ux1);
    Kokkos::deep_copy(ux2_h, ux2);
    Kokkos::deep_copy(ux3_h, ux3);

    Kokkos::deep_copy(tag_h, tag);
    Kokkos::deep_copy(weight_h, weight);

    for (auto n { 0 }; n < npld(); ++n) {
      Kokkos::deep_copy(pld_h[n], pld[n]);
    }

    if constexpr ((D == Dim::_2D) || (D == Dim::_3D)) {
      Kokkos::deep_copy(i2_h, i2);
      Kokkos::deep_copy(dx2_h, dx2);
    }
    if constexpr (D == Dim::_3D) {
      Kokkos::deep_copy(i3_h, i3);
      Kokkos::deep_copy(dx3_h, dx3);
    }

    if ((D == Dim::_2D) && (C != Coord::Cart)) {
      Kokkos::deep_copy(phi_h, phi);
    }
  }

  template struct Particles<Dim::_1D, Coord::Cart>;
  template struct Particles<Dim::_2D, Coord::Cart>;
  template struct Particles<Dim::_3D, Coord::Cart>;
  template struct Particles<Dim::_2D, Coord::Sph>;
  template struct Particles<Dim::_3D, Coord::Sph>;
  template struct Particles<Dim::_2D, Coord::Qsph>;
  template struct Particles<Dim::_3D, Coord::Qsph>;

} // namespace ntt

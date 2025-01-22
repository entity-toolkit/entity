#include "framework/containers/particles.h"

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/sorting.h"

#include "framework/containers/species.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <iterator>
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
  auto Particles<D, C>::NpartsPerTagAndOffsets() const
    -> std::pair<std::vector<std::size_t>, array_t<std::size_t*>> {
    auto                  this_tag = tag;
    const auto            num_tags = ntags();
    array_t<std::size_t*> npptag("nparts_per_tag", ntags());

    // count # of particles per each tag
    auto npptag_scat = Kokkos::Experimental::create_scatter_view(npptag);
    Kokkos::parallel_for(
      "NpartPerTag",
      rangeActiveParticles(),
      Lambda(index_t p) {
        auto npptag_acc = npptag_scat.access();
        if (this_tag(p) < 0 || this_tag(p) >= num_tags) {
          raise::KernelError(HERE, "Invalid tag value");
        }
        npptag_acc(this_tag(p)) += 1;
      });
    Kokkos::Experimental::contribute(npptag, npptag_scat);

    // copy the count to a vector on the host
    auto npptag_h = Kokkos::create_mirror_view(npptag);
    Kokkos::deep_copy(npptag_h, npptag);
    std::vector<std::size_t> npptag_vec(num_tags);
    for (auto t { 0u }; t < num_tags; ++t) {
      npptag_vec[t] = npptag_h(t);
    }

    // count the offsets on the host and copy to device
    array_t<std::size_t*> tag_offset("tag_offset", num_tags - 3);
    auto                  tag_offset_h = Kokkos::create_mirror_view(tag_offset);

    for (auto t { 0u }; t < num_tags - 3; ++t) {
      tag_offset_h(t) = npptag_vec[t + 2] + (t > 0u ? tag_offset_h(t - 1) : 0);
    }
    Kokkos::deep_copy(tag_offset, tag_offset_h);

    return { npptag_vec, tag_offset };
  }

  template <typename T>
  void RemoveDeadInArray(array_t<T*>&                 arr,
                         const array_t<std::size_t*>& indices_alive) {
    auto n_alive = indices_alive.extent(0);
    auto buffer  = Kokkos::View<T*>("buffer", n_alive);
    Kokkos::parallel_for(
      "PopulateBufferAlive",
      n_alive,
      Lambda(index_t p) { buffer(p) = arr(indices_alive(p)); });

    Kokkos::deep_copy(
      Kokkos::subview(arr, std::make_pair(static_cast<std::size_t>(0), n_alive)),
      buffer);
  }

  template <Dimension D, Coord::type C>
  void Particles<D, C>::RemoveDead() {
    const auto  n_part  = npart();
    std::size_t n_alive = 0, n_dead = 0;
    auto&       this_tag = tag;

    Kokkos::parallel_reduce(
      "CountDeadAlive",
      rangeActiveParticles(),
      Lambda(index_t p, std::size_t & nalive, std::size_t & ndead) {
        nalive += (this_tag(p) == ParticleTag::alive);
        ndead  += (this_tag(p) == ParticleTag::dead);
        if (this_tag(p) != ParticleTag::alive and this_tag(p) != ParticleTag::dead) {
          raise::KernelError(HERE, "wrong particle tag");
        }
      },
      n_alive,
      n_dead);

    array_t<std::size_t*> indices_alive { "indices_alive", n_alive };
    array_t<std::size_t*> alive_counter { "counter_alive", 1 };

    Kokkos::parallel_for(
      "AliveIndices",
      rangeActiveParticles(),
      Lambda(index_t p) {
        if (this_tag(p) == ParticleTag::alive) {
          const auto idx     = Kokkos::atomic_fetch_add(&alive_counter(0), 1);
          indices_alive(idx) = p;
        }
      });

    {
      auto alive_counter_h = Kokkos::create_mirror_view(alive_counter);
      Kokkos::deep_copy(alive_counter_h, alive_counter);
      raise::ErrorIf(alive_counter_h(0) != n_alive,
                     "error in finding alive particle indices",
                     HERE);
    }

    if constexpr (D == Dim::_1D or D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i1, indices_alive);
      RemoveDeadInArray(i1_prev, indices_alive);
      RemoveDeadInArray(dx1, indices_alive);
      RemoveDeadInArray(dx1_prev, indices_alive);
    }

    if constexpr (D == Dim::_2D or D == Dim::_3D) {
      RemoveDeadInArray(i2, indices_alive);
      RemoveDeadInArray(i2_prev, indices_alive);
      RemoveDeadInArray(dx2, indices_alive);
      RemoveDeadInArray(dx2_prev, indices_alive);
    }

    if constexpr (D == Dim::_3D) {
      RemoveDeadInArray(i3, indices_alive);
      RemoveDeadInArray(i3_prev, indices_alive);
      RemoveDeadInArray(dx3, indices_alive);
      RemoveDeadInArray(dx3_prev, indices_alive);
    }

    RemoveDeadInArray(ux1, indices_alive);
    RemoveDeadInArray(ux2, indices_alive);
    RemoveDeadInArray(ux3, indices_alive);
    RemoveDeadInArray(weight, indices_alive);

    if constexpr (D == Dim::_2D && C != Coord::Cart) {
      RemoveDeadInArray(phi, indices_alive);
    }

    for (auto& payload : pld) {
      RemoveDeadInArray(payload, indices_alive);
    }

    Kokkos::Experimental::fill(
      "TagAliveParticles",
      AccelExeSpace(),
      Kokkos::subview(this_tag,
                      std::make_pair(static_cast<std::size_t>(0), n_alive)),
      ParticleTag::alive);

    Kokkos::Experimental::fill(
      "TagDeadParticles",
      AccelExeSpace(),
      Kokkos::subview(this_tag, std::make_pair(n_alive, n_alive + n_dead)),
      ParticleTag::dead);

    set_npart(n_alive);
    m_is_sorted = true;
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

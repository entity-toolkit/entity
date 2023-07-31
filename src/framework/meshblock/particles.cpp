#include "meshblock/particles.h"

#include "wrapper.h"

#include "species.h"

#include "utils/utils.h"

#include <cstddef>
#include <string>

namespace ntt {
  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Particles<Dim1, PICEngine>::Particles(const int&            index_,
                                        const std::string&    label_,
                                        const float&          m_,
                                        const float&          ch_,
                                        const std::size_t&    maxnpart_,
                                        const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }

#ifdef MINKOWSKI_METRIC
  template <>
  Particles<Dim2, PICEngine>::Particles(const int&            index_,
                                        const std::string&    label_,
                                        const float&          m_,
                                        const float&          ch_,
                                        const std::size_t&    maxnpart_,
                                        const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      i2_h { Kokkos::create_mirror_view(i2) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      dx2_h { Kokkos::create_mirror_view(dx2) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }
#else    // axisymmetry
  template <>
  Particles<Dim2, PICEngine>::Particles(const int&            index_,
                                        const std::string&    label_,
                                        const float&          m_,
                                        const float&          ch_,
                                        const std::size_t&    maxnpart_,
                                        const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      phi { label_ + "_phi", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      i2_h { Kokkos::create_mirror_view(i2) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      dx2_h { Kokkos::create_mirror_view(dx2) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      phi_h { Kokkos::create_mirror_view(phi) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }
#endif
  template <>
  Particles<Dim3, PICEngine>::Particles(const int&            index_,
                                        const std::string&    label_,
                                        const float&          m_,
                                        const float&          ch_,
                                        const std::size_t&    maxnpart_,
                                        const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      i3 { label_ + "_i3", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      dx3 { label_ + "_dx3", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      i2_h { Kokkos::create_mirror_view(i2) },
      i3_h { Kokkos::create_mirror_view(i3) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      dx2_h { Kokkos::create_mirror_view(dx2) },
      dx3_h { Kokkos::create_mirror_view(dx3) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }

  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific (not Cartesian)
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Particles<Dim2, GRPICEngine>::Particles(const int&            index_,
                                          const std::string&    label_,
                                          const float&          m_,
                                          const float&          ch_,
                                          const std::size_t&    maxnpart_,
                                          const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      i1_prev { label_ + "_i1_prev", maxnpart_ },
      i2_prev { label_ + "_i2_prev", maxnpart_ },
      dx1_prev { label_ + "_dx1_prev", maxnpart_ },
      dx2_prev { label_ + "_dx2_prev", maxnpart_ },
      phi { label_ + "_phi", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      i2_h { Kokkos::create_mirror_view(i2) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      dx2_h { Kokkos::create_mirror_view(dx2) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      phi_h { Kokkos::create_mirror_view(phi) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }

  template <>
  Particles<Dim3, GRPICEngine>::Particles(const int&            index_,
                                          const std::string&    label_,
                                          const float&          m_,
                                          const float&          ch_,
                                          const std::size_t&    maxnpart_,
                                          const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      i3 { label_ + "_i3", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      dx3 { label_ + "_dx3", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      i1_prev { label_ + "_i1_prev", maxnpart_ },
      i2_prev { label_ + "_i2_prev", maxnpart_ },
      i3_prev { label_ + "_i3_prev", maxnpart_ },
      dx1_prev { label_ + "_dx1_prev", maxnpart_ },
      dx2_prev { label_ + "_dx2_prev", maxnpart_ },
      dx3_prev { label_ + "_dx3_prev", maxnpart_ },
      tag { label_ + "_tag", maxnpart_ },
      i1_h { Kokkos::create_mirror_view(i1) },
      i2_h { Kokkos::create_mirror_view(i2) },
      i3_h { Kokkos::create_mirror_view(i3) },
      dx1_h { Kokkos::create_mirror_view(dx1) },
      dx2_h { Kokkos::create_mirror_view(dx2) },
      dx3_h { Kokkos::create_mirror_view(dx3) },
      ux1_h { Kokkos::create_mirror_view(ux1) },
      ux2_h { Kokkos::create_mirror_view(ux2) },
      ux3_h { Kokkos::create_mirror_view(ux3) },
      weight_h { Kokkos::create_mirror_view(weight) },
      tag_h { Kokkos::create_mirror_view(tag) } {
    NTTLog();
  }

  template <>
  Particles<Dim1, SANDBOXEngine>::Particles(const int&            index_,
                                            const std::string&    label_,
                                            const float&          m_,
                                            const float&          ch_,
                                            const std::size_t&    maxnpart_,
                                            const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ } {
    NTTLog();
  }

  template <>
  Particles<Dim2, SANDBOXEngine>::Particles(const int&            index_,
                                            const std::string&    label_,
                                            const float&          m_,
                                            const float&          ch_,
                                            const std::size_t&    maxnpart_,
                                            const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ } {
    NTTLog();
  }

  template <>
  Particles<Dim3, SANDBOXEngine>::Particles(const int&            index_,
                                            const std::string&    label_,
                                            const float&          m_,
                                            const float&          ch_,
                                            const std::size_t&    maxnpart_,
                                            const ParticlePusher& pusher_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_, pusher_ } {
    NTTLog();
  }

  template <Dimension D, SimulationEngine S>
  Particles<D, S>::Particles(const ParticleSpecies& spec)
    : Particles(
      spec.index(), spec.label(), spec.mass(), spec.charge(), spec.maxnpart(), spec.pusher()) {
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::rangeActiveParticles() -> range_t<Dim1> {
    return CreateRangePolicy<Dim1>({ 0 }, { npart() });
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::rangeAllParticles() -> range_t<Dim1> {
    return CreateRangePolicy<Dim1>({ 0 }, { maxnpart() });
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::NpartPerTag() const -> std::vector<std::size_t> {
    auto                      this_tag = this->tag;
    array_t<std::size_t[100]> npart_tag("npart_tags");
    auto npart_tag_scatter { Kokkos::Experimental::create_scatter_view(npart_tag) };
    Kokkos::parallel_for(
      "NpartPerTag", npart(), Lambda(index_t p) {
        auto npart_tag_scatter_access = npart_tag_scatter.access();
        npart_tag_scatter_access((int)(this_tag(p))) += 1;
      });
    Kokkos::Experimental::contribute(npart_tag, npart_tag_scatter);
    auto npart_tag_host = Kokkos::create_mirror_view(npart_tag);
    Kokkos::deep_copy(npart_tag_host, npart_tag);
    std::vector<std::size_t> npart_tag_vec;
    for (auto i { 0 }; i < 100; ++i) {
      npart_tag_vec.push_back(npart_tag_host(i));
    }
    return npart_tag_vec;
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::ReshuffleByTags(bool remove_dead) -> std::vector<std::size_t> {
    using KeyType = array_t<short*>;
    using BinOp   = BinTag<KeyType>;
#ifndef MPI_ENABLED
    const auto ntags = 2;
#else    // MPI_ENABLED
    const auto ntags = 2 + math::pow(3, (int)D) - 1;
#endif
    BinOp                           bin_op(ntags);
    auto                            slice = range_tuple_t(0, npart());
    Kokkos::BinSort<KeyType, BinOp> Sorter(Kokkos::subview(tag, slice), bin_op, false);
    Sorter.create_permute_vector();

    Sorter.sort(Kokkos::subview(tag, slice));
    Sorter.sort(Kokkos::subview(i1, slice));
    Sorter.sort(Kokkos::subview(dx1, slice));
    if constexpr (D == Dim2 || D == Dim3) {
      Sorter.sort(Kokkos::subview(i2, slice));
      Sorter.sort(Kokkos::subview(dx2, slice));
    }
    if constexpr (D == Dim3) {
      Sorter.sort(Kokkos::subview(i3, slice));
      Sorter.sort(Kokkos::subview(dx3, slice));
    }
    Sorter.sort(Kokkos::subview(ux1, slice));
    Sorter.sort(Kokkos::subview(ux2, slice));
    Sorter.sort(Kokkos::subview(ux3, slice));
#ifndef MINKOWSKI_METRIC
    if constexpr (D == Dim2) {
      Sorter.sort(Kokkos::subview(phi, slice));
    }
#endif
    if constexpr (S == GRPICEngine) {
      Sorter.sort(Kokkos::subview(i1_prev, slice));
      Sorter.sort(Kokkos::subview(dx1_prev, slice));
      if constexpr (D == Dim2 || D == Dim3) {
        Sorter.sort(Kokkos::subview(i2_prev, slice));
        Sorter.sort(Kokkos::subview(dx2_prev, slice));
      }
      if constexpr (D == Dim3) {
        Sorter.sort(Kokkos::subview(i3_prev, slice));
        Sorter.sort(Kokkos::subview(dx3_prev, slice));
      }
    }
    Sorter.sort(Kokkos::subview(weight, slice));

    const auto npart_per_tag = NpartPerTag();
    if (remove_dead) {
      setNpart(npart_per_tag[(short)(ParticleTag::alive)]);
    }
    return npart_per_tag;
  }

  template <Dimension D, SimulationEngine S>
  void Particles<D, S>::SyncHostDeviceImpl(DimensionTag<Dim1>) {
    Kokkos::deep_copy(i1_h, i1);
    Kokkos::deep_copy(dx1_h, dx1);
    Kokkos::deep_copy(ux1_h, ux1);
    Kokkos::deep_copy(ux2_h, ux2);
    Kokkos::deep_copy(ux3_h, ux3);
    Kokkos::deep_copy(weight_h, weight);
    Kokkos::deep_copy(tag_h, tag);
  }

  template <Dimension D, SimulationEngine S>
  void Particles<D, S>::SyncHostDeviceImpl(DimensionTag<Dim2>) {
    Kokkos::deep_copy(i1_h, i1);
    Kokkos::deep_copy(i2_h, i2);
    Kokkos::deep_copy(dx1_h, dx1);
    Kokkos::deep_copy(dx2_h, dx2);
    Kokkos::deep_copy(ux1_h, ux1);
    Kokkos::deep_copy(ux2_h, ux2);
    Kokkos::deep_copy(ux3_h, ux3);
    Kokkos::deep_copy(weight_h, weight);
#ifndef MINKOWSKI_METRIC
    Kokkos::deep_copy(phi_h, phi);
#endif
    Kokkos::deep_copy(tag_h, tag);
  }

  template <Dimension D, SimulationEngine S>
  void Particles<D, S>::SyncHostDeviceImpl(DimensionTag<Dim3>) {
    Kokkos::deep_copy(i1_h, i1);
    Kokkos::deep_copy(i2_h, i2);
    Kokkos::deep_copy(i3_h, i3);
    Kokkos::deep_copy(dx1_h, dx1);
    Kokkos::deep_copy(dx2_h, dx2);
    Kokkos::deep_copy(dx3_h, dx3);
    Kokkos::deep_copy(ux1_h, ux1);
    Kokkos::deep_copy(ux2_h, ux2);
    Kokkos::deep_copy(ux3_h, ux3);
    Kokkos::deep_copy(weight_h, weight);
    Kokkos::deep_copy(tag_h, tag);
  }

}    // namespace ntt
#include "particles.h"

#include "wrapper.h"

#include "utils.h"

#include <cstddef>
#include <string>

namespace ntt {
  ParticleSpecies::ParticleSpecies(const int&            index_,
                                   const std::string&    label_,
                                   const float&          m_,
                                   const float&          ch_,
                                   const std::size_t&    maxnpart_,
                                   const ParticlePusher& pusher_)
    : m_index(index_),
      m_label(std::move(label_)),
      m_mass(m_),
      m_charge(ch_),
      m_maxnpart(maxnpart_),
      m_pusher(pusher_) {}

  ParticleSpecies::ParticleSpecies(const int&         index_,
                                   const std::string& label_,
                                   const float&       m_,
                                   const float&       ch_,
                                   const std::size_t& maxnpart_)
    : m_index(index_),
      m_label(std::move(label_)),
      m_mass(m_),
      m_charge(ch_),
      m_maxnpart(maxnpart_),
      m_pusher((m_charge == 0.0 ? ParticlePusher::PHOTON : ParticlePusher::BORIS)) {}

  // * * * * * * * * * * * * * * * * * * * *
  // PIC-specific
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Particles<Dim1, PICEngine>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
      i1 { label_ + "_i1", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    dx1_h = Kokkos::create_mirror(dx1);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }

#ifdef MINKOWSKI_METRIC
  template <>
  Particles<Dim2, PICEngine>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    i2_h  = Kokkos::create_mirror(i2);
    dx1_h = Kokkos::create_mirror(dx1);
    dx2_h = Kokkos::create_mirror(dx2);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }
#else    // axisymmetry
  template <>
  Particles<Dim2, PICEngine>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
      i1 { label_ + "_i1", maxnpart_ },
      i2 { label_ + "_i2", maxnpart_ },
      dx1 { label_ + "_dx1", maxnpart_ },
      dx2 { label_ + "_dx2", maxnpart_ },
      ux1 { label_ + "_ux1", maxnpart_ },
      ux2 { label_ + "_ux2", maxnpart_ },
      ux3 { label_ + "_ux3", maxnpart_ },
      weight { label_ + "_w", maxnpart_ },
      phi { label_ + "_phi", maxnpart_ },
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    i2_h  = Kokkos::create_mirror(i2);
    dx1_h = Kokkos::create_mirror(dx1);
    dx2_h = Kokkos::create_mirror(dx2);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }
#endif
  template <>
  Particles<Dim3, PICEngine>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
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
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    i2_h  = Kokkos::create_mirror(i2);
    i3_h  = Kokkos::create_mirror(i3);
    dx1_h = Kokkos::create_mirror(dx1);
    dx2_h = Kokkos::create_mirror(dx2);
    dx3_h = Kokkos::create_mirror(dx3);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }

  // * * * * * * * * * * * * * * * * * * * *
  // GRPIC-specific (not Cartesian)
  // * * * * * * * * * * * * * * * * * * * *
  template <>
  Particles<Dim2, TypeGRPIC>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
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
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    i2_h  = Kokkos::create_mirror(i2);
    dx1_h = Kokkos::create_mirror(dx1);
    dx2_h = Kokkos::create_mirror(dx2);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }

  template <>
  Particles<Dim3, TypeGRPIC>::Particles(const int&         index_,
                                        const std::string& label_,
                                        const float&       m_,
                                        const float&       ch_,
                                        const std::size_t& maxnpart_)
    : ParticleSpecies { index_, label_, m_, ch_, maxnpart_ },
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
      is_dead { label_ + "_a", maxnpart_ } {
    i1_h  = Kokkos::create_mirror(i1);
    i2_h  = Kokkos::create_mirror(i2);
    i3_h  = Kokkos::create_mirror(i3);
    dx1_h = Kokkos::create_mirror(dx1);
    dx2_h = Kokkos::create_mirror(dx2);
    dx3_h = Kokkos::create_mirror(dx3);
    ux1_h = Kokkos::create_mirror(ux1);
    ux2_h = Kokkos::create_mirror(ux2);
    ux3_h = Kokkos::create_mirror(ux3);
  }

  template <Dimension D, SimulationEngine S>
  Particles<D, S>::Particles(const ParticleSpecies& spec)
    : Particles(spec.index(), spec.label(), spec.mass(), spec.charge(), spec.maxnpart()) {}

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::rangeActiveParticles() -> range_t<Dim1> {
    return CreateRangePolicy<Dim1>({ 0 }, { (int)(npart()) });
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::rangeAllParticles() -> range_t<Dim1> {
    return CreateRangePolicy<Dim1>({ 0 }, { (int)(maxnpart()) });
  }

  template <Dimension D, SimulationEngine S>
  auto Particles<D, S>::CountLiving() const -> std::size_t {
    std::size_t npart_alive = 0;
    auto        is_dead_    = this->is_dead;
    Kokkos::parallel_reduce(
      "RemoveDead",
      npart(),
      Lambda(index_t & p, std::size_t & cnt) {
        if (!is_dead_(p)) {
          cnt++;
        }
      },
      npart_alive);
    return npart_alive;
  }

  template <Dimension D, SimulationEngine S>
  void Particles<D, S>::ReshuffleDead() {
    using KeyType                         = array_t<bool*>;
    using BinOp                           = BinBool<KeyType>;
    auto                            slice = std::pair<std::size_t, std::size_t>(0, npart());
    BinOp                           bin_op;
    Kokkos::BinSort<KeyType, BinOp> Sorter(Kokkos::subview(is_dead, slice), bin_op);
    Sorter.create_permute_vector();
    Sorter.sort(Kokkos::subview(is_dead, slice));
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
    if constexpr (S == TypeGRPIC) {
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
    //  !TODO: sort weights
  }

}    // namespace ntt

#ifdef PIC_ENGINE
template struct ntt::Particles<ntt::Dim1, ntt::PICEngine>;
template struct ntt::Particles<ntt::Dim2, ntt::PICEngine>;
template struct ntt::Particles<ntt::Dim3, ntt::PICEngine>;
#elif defined(GRPIC_ENGINE)
template struct ntt::Particles<ntt::Dim2, ntt::SimulationEngine::GRPIC>;
template struct ntt::Particles<ntt::Dim3, ntt::SimulationEngine::GRPIC>;
#endif

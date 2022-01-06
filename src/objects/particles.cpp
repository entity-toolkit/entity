#include "global.h"
#include "particles.h"

#include <string>
#include <cstddef>

namespace ntt {

  ParticleSpecies::ParticleSpecies(
    std::string label_, const float& m_, const float& ch_, const std::size_t& maxnpart_, const ParticlePusher& pusher_)
    : label(std::move(label_)), mass(m_), charge(ch_), maxnpart(maxnpart_), pusher(pusher_) {}

  ParticleSpecies::ParticleSpecies(std::string label_, const float& m_, const float& ch_, const std::size_t& maxnpart_)
    : label(std::move(label_)),
      mass(m_),
      charge(ch_),
      maxnpart(maxnpart_),
      pusher((charge == 0.0 ? PHOTON_PUSHER : BORIS_PUSHER)) {}

  template <>
  Particles<ONE_D>::Particles(const std::string& label_,
                              const float& m_,
                              const float& ch_,
                              const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      x1 {label_ + "_x1", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <>
  Particles<TWO_D>::Particles(const std::string& label_,
                              const float& m_,
                              const float& ch_,
                              const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      x1 {label_ + "_x1", maxnpart_},
      x2 {label_ + "_x2", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      dx2 {label_ + "_dx2", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <>
  Particles<THREE_D>::Particles(const std::string& label_,
                                const float& m_,
                                const float& ch_,
                                const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      x1 {label_ + "_x1", maxnpart_},
      x2 {label_ + "_x2", maxnpart_},
      x3 {label_ + "_x3", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      dx2 {label_ + "_dx2", maxnpart_},
      dx3 {label_ + "_dx3", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <Dimension D>
  Particles<D>::Particles(const ParticleSpecies& spec)
    : Particles(spec.get_label(), spec.get_mass(), spec.get_charge(), spec.get_maxnpart()) {}

  template <Dimension D>
  auto Particles<D>::loopParticles() -> ntt_1drange_t {
    return NTT1DRange((range_t)(0), (range_t)(get_npart()));
  }

} // namespace ntt

template struct ntt::Particles<ntt::ONE_D>;
template struct ntt::Particles<ntt::TWO_D>;
template struct ntt::Particles<ntt::THREE_D>;

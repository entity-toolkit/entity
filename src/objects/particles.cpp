#include "global.h"
#include "particles.h"

#include <string>
#include <cstddef>

namespace ntt {

  ParticleSpecies::ParticleSpecies(
    std::string label_, const float& m_, const float& ch_, const std::size_t& maxnpart_, const ParticlePusher& pusher_)
    : m_label(std::move(label_)), m_mass(m_), m_charge(ch_), m_maxnpart(maxnpart_), m_pusher(pusher_) {}

  ParticleSpecies::ParticleSpecies(std::string label_, const float& m_, const float& ch_, const std::size_t& maxnpart_)
    : m_label(std::move(label_)),
      m_mass(m_),
      m_charge(ch_),
      m_maxnpart(maxnpart_),
      m_pusher((m_charge == 0.0 ? ParticlePusher::PHOTON : ParticlePusher::BORIS)) {}

  template <>
  Particles<Dimension::ONE_D, SimulationType::PIC>::Particles(const std::string& label_,
                                                              const float& m_,
                                                              const float& ch_,
                                                              const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      i1 {label_ + "_i1", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <>
  Particles<Dimension::TWO_D, SimulationType::PIC>::Particles(const std::string& label_,
                                                              const float& m_,
                                                              const float& ch_,
                                                              const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      i1 {label_ + "_i1", maxnpart_},
      i2 {label_ + "_i2", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      dx2 {label_ + "_dx2", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <>
  Particles<Dimension::THREE_D, SimulationType::PIC>::Particles(const std::string& label_,
                                                                const float& m_,
                                                                const float& ch_,
                                                                const std::size_t& maxnpart_)
    : ParticleSpecies {label_, m_, ch_, maxnpart_},
      i1 {label_ + "_i1", maxnpart_},
      i2 {label_ + "_i2", maxnpart_},
      i3 {label_ + "_i3", maxnpart_},
      dx1 {label_ + "_dx1", maxnpart_},
      dx2 {label_ + "_dx2", maxnpart_},
      dx3 {label_ + "_dx3", maxnpart_},
      ux1 {label_ + "_ux1", maxnpart_},
      ux2 {label_ + "_ux2", maxnpart_},
      ux3 {label_ + "_ux3", maxnpart_},
      weight {label_ + "_w", maxnpart_} {}

  template <Dimension D, SimulationType S>
  Particles<D, S>::Particles(const ParticleSpecies& spec)
    : Particles(spec.label(), spec.mass(), spec.charge(), spec.maxnpart()) {}

  template <Dimension D, SimulationType S>
  auto Particles<D, S>::loopParticles() -> RangeND<Dimension::ONE_D> {
    return NTTRange<Dimension::ONE_D>({0}, {(int)(npart())});
  }

} // namespace ntt

template class ntt::Particles<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
template class ntt::Particles<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
template class ntt::Particles<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;
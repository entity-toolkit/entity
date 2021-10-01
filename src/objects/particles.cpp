#include "global.h"
#include "particles.h"

namespace ntt {

ParticleSpecies::ParticleSpecies(const std::string& label,
                                 const float& m,
                                 const float& ch,
                                 const std::size_t& maxnpart,
                                 const ParticlePusher& pusher)
    : m_label(label), m_mass(m), m_charge(ch), m_maxnpart(maxnpart), m_pusher(pusher) {}

ParticleSpecies::ParticleSpecies(const std::string& label,
                                 const float& m,
                                 const float& ch,
                                 const std::size_t& maxnpart)
    : m_label(label),
      m_mass(m),
      m_charge(ch),
      m_maxnpart(maxnpart),
      m_pusher((ch == 0.0 ? PHOTON_PUSHER : BORIS_PUSHER)) {}

ParticleSpecies::ParticleSpecies(const ParticleSpecies& spec)
    : m_label(spec.m_label),
      m_mass(spec.m_mass),
      m_charge(spec.m_charge),
      m_maxnpart(spec.m_maxnpart),
      m_pusher(spec.m_pusher) {}

template <>
Particles<One_D>::Particles(const std::string& label,
                            const float& m,
                            const float& ch,
                            const std::size_t& maxnpart)
    : ParticleSpecies{label, m, ch, maxnpart},
      m_x1{label + "_x1", maxnpart},
      m_ux1{label + "_ux1", maxnpart},
      m_ux2{label + "_ux2", maxnpart},
      m_ux3{label + "_ux3", maxnpart},
      m_weight{label + "_w", maxnpart} {}

template <>
Particles<Two_D>::Particles(const std::string& label,
                            const float& m,
                            const float& ch,
                            const std::size_t& maxnpart)
    : ParticleSpecies{label, m, ch, maxnpart},
      m_x1{label + "_x1", maxnpart},
      m_x2{label + "_x2", maxnpart},
      m_ux1{label + "_ux1", maxnpart},
      m_ux2{label + "_ux2", maxnpart},
      m_ux3{label + "_ux3", maxnpart},
      m_weight{label + "_w", maxnpart} {}

template <>
Particles<Three_D>::Particles(const std::string& label,
                              const float& m,
                              const float& ch,
                              const std::size_t& maxnpart)
    : ParticleSpecies{label, m, ch, maxnpart},
      m_x1{label + "_x1", maxnpart},
      m_x2{label + "_x2", maxnpart},
      m_x3{label + "_x3", maxnpart},
      m_ux1{label + "_ux1", maxnpart},
      m_ux2{label + "_ux2", maxnpart},
      m_ux3{label + "_ux3", maxnpart},
      m_weight{label + "_w", maxnpart} {}

template <template <typename T = std::nullptr_t> class D>
Particles<D>::Particles(const ParticleSpecies& spec)
    : Particles(spec.m_label, spec.m_mass, spec.m_charge, spec.m_maxnpart) {}

} // namespace ntt

template class ntt::Particles<ntt::One_D>;
template class ntt::Particles<ntt::Two_D>;
template class ntt::Particles<ntt::Three_D>;

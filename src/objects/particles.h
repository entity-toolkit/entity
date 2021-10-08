#ifndef OBJECTS_PARTICLES_H
#define OBJECTS_PARTICLES_H

#include "global.h"

#include <cstddef>
#include <string>

namespace ntt {

struct ParticleSpecies {
  std::string m_label;
  float m_mass, m_charge;
  std::size_t m_maxnpart;
  ParticlePusher m_pusher;

  ParticleSpecies(std::string label,
                  const float& m,
                  const float& ch,
                  const std::size_t& maxnpart,
                  const ParticlePusher& pusher);
  ParticleSpecies(std::string label, const float& m, const float& ch, const std::size_t& maxnpart);
  ParticleSpecies(const ParticleSpecies& spec) = default;
  ~ParticleSpecies() = default;

  [[nodiscard]] auto get_label() const -> std::string { return m_label; }
  [[nodiscard]] auto get_mass() const -> float { return m_mass; }
  [[nodiscard]] auto get_charge() const -> float { return m_charge; }
  [[nodiscard]] auto get_maxnpart() const -> std::size_t { return m_maxnpart; }
  [[nodiscard]] auto get_pusher() const -> ParticlePusher { return m_pusher; }
};

template <Dimension D>
struct Particles : ParticleSpecies {
  Dimension m_dim{D};

  // TODO: maybe use VPIC-style ND array
  NTTArray<real_t*> m_x1, m_x2, m_x3;
  NTTArray<real_t*> m_ux1, m_ux2, m_ux3;
  NTTArray<float*> m_weight;

  std::size_t m_npart{0};

  Particles(const std::string& label, const float& m, const float& ch, const std::size_t& maxnpart);
  Particles(const ParticleSpecies& spec);
  ~Particles() = default;

  auto loopParticles() -> ntt_1drange_t;
  [[nodiscard]] auto get_npart() const -> std::size_t { return m_npart; }
};

} // namespace ntt

#endif

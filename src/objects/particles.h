#ifndef OBJECTS_PARTICLES_H
#define OBJECTS_PARTICLES_H

#include "global.h"

#include <cstddef>
#include <string>

namespace ntt {

  struct ParticleSpecies {
  protected:
    std::string label;
    float mass, charge;
    std::size_t maxnpart;
    ParticlePusher pusher;

  public:
    ParticleSpecies(
        std::string label,
        const float& m,
        const float& ch,
        const std::size_t& maxnpart,
        const ParticlePusher& pusher);
    ParticleSpecies(std::string, const float&, const float&, const std::size_t&);
    ParticleSpecies(const ParticleSpecies&) = default;
    ~ParticleSpecies() = default;

    [[nodiscard]] auto get_label() const -> std::string { return label; }
    [[nodiscard]] auto get_mass() const -> float { return mass; }
    [[nodiscard]] auto get_charge() const -> float { return charge; }
    [[nodiscard]] auto get_maxnpart() const -> std::size_t { return maxnpart; }
    [[nodiscard]] auto get_pusher() const -> ParticlePusher { return pusher; }
  };

  template <Dimension D>
  struct Particles : ParticleSpecies {
  protected:
    std::size_t npart {0};
  public:
    Dimension m_dim {D};

    // TESTPERF: maybe use VPIC-style ND array
    NTTArray<int*> x1, x2, x3;
    NTTArray<float*> dx1, dx2, dx3;
    NTTArray<real_t*> ux1, ux2, ux3;
    NTTArray<float*> weight;

    Particles(const std::string&, const float&, const float&, const std::size_t&);
    Particles(const ParticleSpecies&);
    ~Particles() = default;

    auto loopParticles() -> ntt_1drange_t;
    [[nodiscard]] auto get_npart() const -> std::size_t { return npart; }
  };

} // namespace ntt

#endif

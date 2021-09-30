#include "global.h"
#include "particles.h"

namespace ntt {

template <template <typename T = std::nullptr_t> class D>
ParticleSpecies<D>::ParticleSpecies(const float &m, const float &ch, const std::size_t &maxnpart)
    : m_mass(m), m_charge(ch), m_maxnpart(maxnpart) {}

} // namespace ntt

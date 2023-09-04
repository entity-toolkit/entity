#include "species.h"

#include "wrapper.h"

namespace ntt {
  ParticleSpecies::ParticleSpecies(const int&            index_,
                                   const std::string&    label_,
                                   const float&          m_,
                                   const float&          ch_,
                                   const std::size_t&    maxnpart_,
                                   const ParticlePusher& pusher_,
                                   const unsigned short& npld_) :
    m_index { index_ },
    m_label { std::move(label_) },
    m_mass { m_ },
    m_charge { ch_ },
    m_maxnpart { maxnpart_ },
    m_pusher { pusher_ },
    m_npld { npld_ } {}
} // namespace ntt
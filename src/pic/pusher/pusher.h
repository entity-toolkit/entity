#ifndef PIC_PUSHER_H
#define PIC_PUSHER_H

#include "global.h"
#include "meshblock.h"
#include "particles.h"

namespace ntt {

template <Dimension D>
class Pusher {
public:
  Meshblock<D> m_meshblock;
  Particles<D> m_particles;
  real_t coeff;
  real_t dt;

  Pusher(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : m_meshblock(m_meshblock_), m_particles(m_particles_), coeff(coeff_), dt(dt_) {}
};

} // namespace ntt

template class ntt::Pusher<ntt::ONE_D>;
template class ntt::Pusher<ntt::TWO_D>;
template class ntt::Pusher<ntt::THREE_D>;

#endif

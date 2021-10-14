#ifndef PIC_PUSHER_PHOTON_H
#define PIC_PUSHER_PHOTON_H

#include "global.h"
#include "meshblock.h"
#include "pusher.h"

#include <cmath>

namespace ntt {

template <Dimension D>
class PhotonPusher : public Pusher<D> {
  using index_t = typename Pusher<D>::index_t;

public:
  PhotonPusher(
      const Meshblock<D>& m_meshblock_,
      const Particles<D>& m_particles_,
      const real_t& coeff_,
      const real_t& dt_)
      : Pusher<D> {m_meshblock_, m_particles_, coeff_, dt_} {}

  void velocityUpdate(
      const index_t&, real_t&, real_t&, real_t&, real_t&, real_t&, real_t&) const override {}
  Inline void operator()(const index_t p) const {
    Pusher<D>::positionUpdate(p);
  }
};

} // namespace ntt

template class ntt::PhotonPusher<ntt::ONE_D>;
template class ntt::PhotonPusher<ntt::TWO_D>;
template class ntt::PhotonPusher<ntt::THREE_D>;

#endif

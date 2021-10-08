#ifndef PIC_PUSHER_H
#define PIC_PUSHER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

template <Dimension D>
class Pusher {
protected:
  MeshblockND<D> m_meshblock;
  std::size_t m_sp;
  using size_type = NTTArray<real_t*>::size_type;

public:
  Pusher(const MeshblockND<D>& m_meshblock_, const std::size_t& m_sp_) : m_meshblock(m_meshblock_), m_sp{m_sp_} {}
};

}

#endif

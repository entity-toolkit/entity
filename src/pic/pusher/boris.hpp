#ifndef PIC_PUSHER_BORIS_H
#define PIC_PUSHER_BORIS_H

#include "global.h"
#include "meshblock.h"
#include "pusher.h"

namespace ntt {

class Boris1D : public Pusher<ONE_D> {
public:
  Boris1D(const Meshblock1D& m_meshblock_, const std::size_t& m_sp_)
      : Pusher<ONE_D>{m_meshblock_, m_sp_} {}
  Inline void operator()(const size_type i) const {}
};

class Boris2D : public Pusher<TWO_D> {
public:
  Boris2D(const Meshblock2D& m_meshblock_, const std::size_t& m_sp_)
      : Pusher<TWO_D>{m_meshblock_, m_sp_} {}
  Inline void operator()(const size_type i) const {}
};

class Boris3D : public Pusher<THREE_D> {
public:
  Boris3D(const Meshblock3D& m_meshblock_, const std::size_t& m_sp_)
      : Pusher<THREE_D>{m_meshblock_, m_sp_} {}
  Inline void operator()(const size_type i) const {}
};

} // namespace ntt

#endif

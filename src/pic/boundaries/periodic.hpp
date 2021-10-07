#ifndef PIC_BOUNDARIES_PERIODIC_H
#define PIC_BOUNDARIES_PERIODIC_H

#include "global.h"
#include "boundaries.h"

namespace ntt {

// 1d
// ... x1
class BC1D_PeriodicX1m : public BC1D {
public:
  BC1D_PeriodicX1m(const Meshblock1D& m_mblock_, const std::size_t& nxI_) : BC1D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i) const {
    m_mblock.ex1(i) = m_mblock.ex1(i + nxI);
    m_mblock.ex2(i) = m_mblock.ex2(i + nxI);
    m_mblock.ex3(i) = m_mblock.ex3(i + nxI);
    m_mblock.bx1(i) = m_mblock.bx1(i + nxI);
    m_mblock.bx2(i) = m_mblock.bx2(i + nxI);
    m_mblock.bx3(i) = m_mblock.bx3(i + nxI);
  }
};
class BC1D_PeriodicX1p : public BC1D {
public:
  BC1D_PeriodicX1p(const Meshblock1D& m_mblock_, const std::size_t& nxI_) : BC1D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i) const {
    m_mblock.ex1(i) = m_mblock.ex1(i - nxI);
    m_mblock.ex2(i) = m_mblock.ex2(i - nxI);
    m_mblock.ex3(i) = m_mblock.ex3(i - nxI);
    m_mblock.bx1(i) = m_mblock.bx1(i - nxI);
    m_mblock.bx2(i) = m_mblock.bx2(i - nxI);
    m_mblock.bx3(i) = m_mblock.bx3(i - nxI);
  }
};

// 2d
// ... x1
class BC2D_PeriodicX1m : public BC2D {
public:
  BC2D_PeriodicX1m(const Meshblock2D& m_mblock_, const std::size_t& nxI_) : BC2D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i + nxI, j);
    m_mblock.ex2(i, j) = m_mblock.ex2(i + nxI, j);
    m_mblock.ex3(i, j) = m_mblock.ex3(i + nxI, j);
    m_mblock.bx1(i, j) = m_mblock.bx1(i + nxI, j);
    m_mblock.bx2(i, j) = m_mblock.bx2(i + nxI, j);
    m_mblock.bx3(i, j) = m_mblock.bx3(i + nxI, j);
  }
};
class BC2D_PeriodicX1p : public BC2D {
public:
  BC2D_PeriodicX1p(const Meshblock2D& m_mblock_, const std::size_t& nxI_) : BC2D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i - nxI, j);
    m_mblock.ex2(i, j) = m_mblock.ex2(i - nxI, j);
    m_mblock.ex3(i, j) = m_mblock.ex3(i - nxI, j);
    m_mblock.bx1(i, j) = m_mblock.bx1(i - nxI, j);
    m_mblock.bx2(i, j) = m_mblock.bx2(i - nxI, j);
    m_mblock.bx3(i, j) = m_mblock.bx3(i - nxI, j);
  }
};
// ... x2
class BC2D_PeriodicX2m : public BC2D {
public:
  BC2D_PeriodicX2m(const Meshblock2D& m_mblock_, const std::size_t& nxI_) : BC2D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i, j + nxI);
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j + nxI);
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j + nxI);
    m_mblock.bx1(i, j) = m_mblock.bx1(i, j + nxI);
    m_mblock.bx2(i, j) = m_mblock.bx2(i, j + nxI);
    m_mblock.bx3(i, j) = m_mblock.bx3(i, j + nxI);
  }
};
class BC2D_PeriodicX2p : public BC2D {
public:
  BC2D_PeriodicX2p(const Meshblock2D& m_mblock_, const std::size_t& nxI_) : BC2D{m_mblock_, nxI_} {}
  Inline void operator()(const size_type i, const size_type j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i, j - nxI);
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j - nxI);
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j - nxI);
    m_mblock.bx1(i, j) = m_mblock.bx1(i, j - nxI);
    m_mblock.bx2(i, j) = m_mblock.bx2(i, j - nxI);
    m_mblock.bx3(i, j) = m_mblock.bx3(i, j - nxI);
  }
};

} // namespace ntt

#endif

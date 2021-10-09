#ifndef PIC_FLDS_BOUNDARIES_PERIODIC_H
#define PIC_FLDS_BOUNDARIES_PERIODIC_H

#include "global.h"
#include "field_boundaries.h"

namespace ntt {

// 1d
// ... x1
class FldBC1D_PeriodicX1m : public FldBC<ONE_D> {
public:
  FldBC1D_PeriodicX1m(const Meshblock<ONE_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<ONE_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i) const {
    m_mblock.ex1(i) = m_mblock.ex1(i + nxI);
    m_mblock.ex2(i) = m_mblock.ex2(i + nxI);
    m_mblock.ex3(i) = m_mblock.ex3(i + nxI);
    m_mblock.bx1(i) = m_mblock.bx1(i + nxI);
    m_mblock.bx2(i) = m_mblock.bx2(i + nxI);
    m_mblock.bx3(i) = m_mblock.bx3(i + nxI);
  }
};
class FldBC1D_PeriodicX1p : public FldBC<ONE_D> {
public:
  FldBC1D_PeriodicX1p(const Meshblock<ONE_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<ONE_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i) const {
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
class FldBC2D_PeriodicX1m : public FldBC<TWO_D> {
public:
  FldBC2D_PeriodicX1m(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<TWO_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i, const index_t j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i + nxI, j);
    m_mblock.ex2(i, j) = m_mblock.ex2(i + nxI, j);
    m_mblock.ex3(i, j) = m_mblock.ex3(i + nxI, j);
    m_mblock.bx1(i, j) = m_mblock.bx1(i + nxI, j);
    m_mblock.bx2(i, j) = m_mblock.bx2(i + nxI, j);
    m_mblock.bx3(i, j) = m_mblock.bx3(i + nxI, j);
  }
};
class FldBC2D_PeriodicX1p : public FldBC<TWO_D> {
public:
  FldBC2D_PeriodicX1p(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<TWO_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i, const index_t j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i - nxI, j);
    m_mblock.ex2(i, j) = m_mblock.ex2(i - nxI, j);
    m_mblock.ex3(i, j) = m_mblock.ex3(i - nxI, j);
    m_mblock.bx1(i, j) = m_mblock.bx1(i - nxI, j);
    m_mblock.bx2(i, j) = m_mblock.bx2(i - nxI, j);
    m_mblock.bx3(i, j) = m_mblock.bx3(i - nxI, j);
  }
};
// ... x2
class FldBC2D_PeriodicX2m : public FldBC<TWO_D> {
public:
  FldBC2D_PeriodicX2m(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<TWO_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i, const index_t j) const {
    m_mblock.ex1(i, j) = m_mblock.ex1(i, j + nxI);
    m_mblock.ex2(i, j) = m_mblock.ex2(i, j + nxI);
    m_mblock.ex3(i, j) = m_mblock.ex3(i, j + nxI);
    m_mblock.bx1(i, j) = m_mblock.bx1(i, j + nxI);
    m_mblock.bx2(i, j) = m_mblock.bx2(i, j + nxI);
    m_mblock.bx3(i, j) = m_mblock.bx3(i, j + nxI);
  }
};
class FldBC2D_PeriodicX2p : public FldBC<TWO_D> {
public:
  FldBC2D_PeriodicX2p(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
      : FldBC<TWO_D> {m_mblock_, nxI_} {}
  Inline void operator()(const index_t i, const index_t j) const {
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

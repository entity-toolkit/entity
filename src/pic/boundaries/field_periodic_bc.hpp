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
      m_mblock.em_fields(i, fld::ex1) = m_mblock.em_fields(i + nxI, fld::ex1);
      m_mblock.em_fields(i, fld::ex2) = m_mblock.em_fields(i + nxI, fld::ex2);
      m_mblock.em_fields(i, fld::ex3) = m_mblock.em_fields(i + nxI, fld::ex3);
      m_mblock.em_fields(i, fld::bx1) = m_mblock.em_fields(i + nxI, fld::bx1);
      m_mblock.em_fields(i, fld::bx2) = m_mblock.em_fields(i + nxI, fld::bx2);
      m_mblock.em_fields(i, fld::bx3) = m_mblock.em_fields(i + nxI, fld::bx3);
    }
  };
  class FldBC1D_PeriodicX1p : public FldBC<ONE_D> {
  public:
    FldBC1D_PeriodicX1p(const Meshblock<ONE_D>& m_mblock_, const std::size_t& nxI_)
        : FldBC<ONE_D> {m_mblock_, nxI_} {}
    Inline void operator()(const index_t i) const {
      m_mblock.em_fields(i, fld::ex1) = m_mblock.em_fields(i - nxI, fld::ex1);
      m_mblock.em_fields(i, fld::ex2) = m_mblock.em_fields(i - nxI, fld::ex2);
      m_mblock.em_fields(i, fld::ex3) = m_mblock.em_fields(i - nxI, fld::ex3);
      m_mblock.em_fields(i, fld::bx1) = m_mblock.em_fields(i - nxI, fld::bx1);
      m_mblock.em_fields(i, fld::bx2) = m_mblock.em_fields(i - nxI, fld::bx2);
      m_mblock.em_fields(i, fld::bx3) = m_mblock.em_fields(i - nxI, fld::bx3);
    }
  };

  // 2d
  // ... x1
  class FldBC2D_PeriodicX1m : public FldBC<TWO_D> {
  public:
    FldBC2D_PeriodicX1m(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
        : FldBC<TWO_D> {m_mblock_, nxI_} {}
    Inline void operator()(const index_t i, const index_t j) const {
      m_mblock.em_fields(i, j, fld::ex1) = m_mblock.em_fields(i + nxI, j, fld::ex1);
      m_mblock.em_fields(i, j, fld::ex2) = m_mblock.em_fields(i + nxI, j, fld::ex2);
      m_mblock.em_fields(i, j, fld::ex3) = m_mblock.em_fields(i + nxI, j, fld::ex3);
      m_mblock.em_fields(i, j, fld::bx1) = m_mblock.em_fields(i + nxI, j, fld::bx1);
      m_mblock.em_fields(i, j, fld::bx2) = m_mblock.em_fields(i + nxI, j, fld::bx2);
      m_mblock.em_fields(i, j, fld::bx3) = m_mblock.em_fields(i + nxI, j, fld::bx3);
    }
  };
  class FldBC2D_PeriodicX1p : public FldBC<TWO_D> {
  public:
    FldBC2D_PeriodicX1p(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
        : FldBC<TWO_D> {m_mblock_, nxI_} {}
    Inline void operator()(const index_t i, const index_t j) const {
      m_mblock.em_fields(i, j, fld::ex1) = m_mblock.em_fields(i - nxI, j, fld::ex1);
      m_mblock.em_fields(i, j, fld::ex2) = m_mblock.em_fields(i - nxI, j, fld::ex2);
      m_mblock.em_fields(i, j, fld::ex3) = m_mblock.em_fields(i - nxI, j, fld::ex3);
      m_mblock.em_fields(i, j, fld::bx1) = m_mblock.em_fields(i - nxI, j, fld::bx1);
      m_mblock.em_fields(i, j, fld::bx2) = m_mblock.em_fields(i - nxI, j, fld::bx2);
      m_mblock.em_fields(i, j, fld::bx3) = m_mblock.em_fields(i - nxI, j, fld::bx3);
    }
  };
  // ... x2
  class FldBC2D_PeriodicX2m : public FldBC<TWO_D> {
  public:
    FldBC2D_PeriodicX2m(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
        : FldBC<TWO_D> {m_mblock_, nxI_} {}
    Inline void operator()(const index_t i, const index_t j) const {
      m_mblock.em_fields(i, j, fld::ex1) = m_mblock.em_fields(i, j + nxI, fld::ex1);
      m_mblock.em_fields(i, j, fld::ex2) = m_mblock.em_fields(i, j + nxI, fld::ex2);
      m_mblock.em_fields(i, j, fld::ex3) = m_mblock.em_fields(i, j + nxI, fld::ex3);
      m_mblock.em_fields(i, j, fld::bx1) = m_mblock.em_fields(i, j + nxI, fld::bx1);
      m_mblock.em_fields(i, j, fld::bx2) = m_mblock.em_fields(i, j + nxI, fld::bx2);
      m_mblock.em_fields(i, j, fld::bx3) = m_mblock.em_fields(i, j + nxI, fld::bx3);
    }
  };
  class FldBC2D_PeriodicX2p : public FldBC<TWO_D> {
  public:
    FldBC2D_PeriodicX2p(const Meshblock<TWO_D>& m_mblock_, const std::size_t& nxI_)
        : FldBC<TWO_D> {m_mblock_, nxI_} {}
    Inline void operator()(const index_t i, const index_t j) const {
      m_mblock.em_fields(i, j, fld::ex1) = m_mblock.em_fields(i, j - nxI, fld::ex1);
      m_mblock.em_fields(i, j, fld::ex2) = m_mblock.em_fields(i, j - nxI, fld::ex2);
      m_mblock.em_fields(i, j, fld::ex3) = m_mblock.em_fields(i, j - nxI, fld::ex3);
      m_mblock.em_fields(i, j, fld::bx1) = m_mblock.em_fields(i, j - nxI, fld::bx1);
      m_mblock.em_fields(i, j, fld::bx2) = m_mblock.em_fields(i, j - nxI, fld::bx2);
      m_mblock.em_fields(i, j, fld::bx3) = m_mblock.em_fields(i, j - nxI, fld::bx3);
    }
  };

} // namespace ntt

#endif

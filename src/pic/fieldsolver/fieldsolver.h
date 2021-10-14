#ifndef PIC_FIELDSOLVER_H
#define PIC_FIELDSOLVER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

template <Dimension D>
class FieldSolver {
protected:
  Meshblock<D> m_mblock;

public:
  FieldSolver(const Meshblock<D>& m_mblock_)
      : m_mblock(m_mblock_) {}
};

} // namespace ntt

#endif

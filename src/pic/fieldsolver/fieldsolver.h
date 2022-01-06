#ifndef PIC_FIELDSOLVER_H
#define PIC_FIELDSOLVER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

  template <Dimension D>
  class FieldSolver {
  protected:
    Meshblock<D> mblock;

  public:
    FieldSolver(const Meshblock<D>& mblock_) : mblock(mblock_) {}
  };

} // namespace ntt

#endif

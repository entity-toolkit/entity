#ifndef PIC_FIELDSOLVER_H
#define PIC_FIELDSOLVER_H

#include "global.h"
#include "meshblock.h"

namespace ntt {

template<Dimension D>
class FieldSolver {
protected:
  MeshblockND<D> m_mblock;
  real_t coeff;
  using size_type = NTTArray<real_t*>::size_type;

public:
  FieldSolver(const MeshblockND<D>& m_mblock_, const real_t& coeff_)
      : m_mblock(m_mblock_), coeff(coeff_) {}
};

} // namespace ntt

#endif

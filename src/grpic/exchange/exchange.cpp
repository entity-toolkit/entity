#include "wrapper.h"

#include "io/output.h"
#include "grpic.h"
#include "meshblock/meshblock.h"

namespace ntt {
  template <Dimension D>
  void GRPIC<D>::Exchange(const GhostCells&) {}
}    // namespace ntt

template void ntt::GRPIC<ntt::Dim2>::Exchange(const ntt::GhostCells&);
template void ntt::GRPIC<ntt::Dim3>::Exchange(const ntt::GhostCells&);

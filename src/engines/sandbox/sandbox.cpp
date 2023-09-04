#include "sandbox.h"

#include "wrapper.h"

namespace ntt {
  template <Dimension D>
  void SANDBOX<D>::Run() {}
} // namespace ntt

template class ntt::SANDBOX<ntt::Dim1>;
template class ntt::SANDBOX<ntt::Dim2>;
template class ntt::SANDBOX<ntt::Dim3>;

#include "global.h"
#include "grid.h"

#include <vector>

namespace ntt {

template <Dimension D>
Grid<D>::Grid(std::vector<std::size_t> res) : m_resolution {std::move(res)} {}

template <>
auto Grid<ONE_D>::loopActiveCells() -> ntt_1drange_t {
  return NTT1DRange(static_cast<range_t>(get_imin()), static_cast<range_t>(get_imax()));
}
template <>
auto Grid<TWO_D>::loopActiveCells() -> ntt_2drange_t {
  return NTT2DRange({get_imin(), get_jmin()}, {get_imax(), get_jmax()});
}
template <>
auto Grid<THREE_D>::loopActiveCells() -> ntt_3drange_t {
  return NTT3DRange({get_imin(), get_jmin(), get_kmin()}, {get_imax(), get_jmax(), get_kmax()});
}

} // namespace ntt

template struct ntt::Grid<ntt::ONE_D>;
template struct ntt::Grid<ntt::TWO_D>;
template struct ntt::Grid<ntt::THREE_D>;

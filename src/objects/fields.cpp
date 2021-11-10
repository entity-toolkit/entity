#include "global.h"
#include "fields.h"

#include <vector>

namespace ntt {

template <>
Fields<ONE_D>::Fields(std::vector<std::size_t> res)
    : Grid<ONE_D>(res), em_fields {"EM", res[0] + 2 * N_GHOSTS}, j_fields {"J", res[0] + 2 * N_GHOSTS} {}

template <>
Fields<TWO_D>::Fields(std::vector<std::size_t> res)
    : Grid<TWO_D>(res),
      em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
      j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {}

template <>
Fields<THREE_D>::Fields(std::vector<std::size_t> res)
    : Grid<THREE_D>(res),
      em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
      j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {}

template <>
auto Fields<ONE_D>::loopActiveCells() -> ntt_1drange_t {
  return NTT1DRange(static_cast<range_t>(get_imin()), static_cast<range_t>(get_imax()));
}
template <>
auto Fields<TWO_D>::loopActiveCells() -> ntt_2drange_t {
  return NTT2DRange({get_imin(), get_jmin()}, {get_imax(), get_jmax()});
}
template <>
auto Fields<THREE_D>::loopActiveCells() -> ntt_3drange_t {
  return NTT3DRange({get_imin(), get_jmin(), get_kmin()}, {get_imax(), get_jmax(), get_kmax()});
}

} // namespace ntt

template struct ntt::Fields<ntt::ONE_D>;
template struct ntt::Fields<ntt::TWO_D>;
template struct ntt::Fields<ntt::THREE_D>;

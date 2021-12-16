#include "global.h"
#include "fields.h"

#include <vector>

namespace ntt {

  template <>
  Fields<ONE_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS}, j_fields {"J", res[0] + 2 * N_GHOSTS} {}

  template <>
  Fields<TWO_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS},
        j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS} {}

  template <>
  Fields<THREE_D>::Fields(std::vector<std::size_t> res)
      : em_fields {"EM", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS},
        j_fields {"J", res[0] + 2 * N_GHOSTS, res[1] + 2 * N_GHOSTS, res[2] + 2 * N_GHOSTS} {}

} // namespace ntt

template struct ntt::Fields<ntt::ONE_D>;
template struct ntt::Fields<ntt::TWO_D>;
template struct ntt::Fields<ntt::THREE_D>;

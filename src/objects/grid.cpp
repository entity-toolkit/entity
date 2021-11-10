#include "global.h"
#include "grid.h"

#include <vector>

namespace ntt {

template <Dimension D>
Grid<D>::Grid(std::vector<std::size_t> res) : m_resolution {std::move(res)} {}

} // namespace ntt

template struct ntt::Grid<ntt::ONE_D>;
template struct ntt::Grid<ntt::TWO_D>;
template struct ntt::Grid<ntt::THREE_D>;

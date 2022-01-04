#include "global.h"
#include "grid.h"

#include <vector>
#include <algorithm>


namespace ntt {

  template <Dimension D>
  Grid<D>::Grid(std::vector<real_t> ext,
                std::vector<std::size_t> res)
    : m_extent {std::move(ext)},
      m_resolution {std::move(res)},
      grid_x1 {"grid_x1", res.size() > 0 ? res[0] + 1 : 1},
      grid_x2 {"grid_x2", res.size() > 1 ? res[1] + 1 : 1},
      grid_x3 {"grid_x3", res.size() > 2 ? res[2] + 1 : 1} {
    using index_t = NTTArray<real_t*>::size_type;
    Kokkos::parallel_for(grid_x1.extent(0),
        Lambda(index_t n) {
          grid_x1(n) = static_cast<real_t>(n) / std::max(ONE, static_cast<real_t>(grid_x1.extent(0) - 1));
    });
    Kokkos::parallel_for(grid_x2.extent(0),
        Lambda(index_t n) {
          grid_x2(n) = static_cast<real_t>(n) / std::max(ONE, static_cast<real_t>(grid_x2.extent(0) - 1));
    });
    Kokkos::parallel_for(grid_x3.extent(0),
        Lambda(index_t n) {
          grid_x3(n) = static_cast<real_t>(n) / std::max(ONE, static_cast<real_t>(grid_x3.extent(0) - 1));
    });
  }

  // # # # # # # # # # # # # # # # #
  // custom range
  template <>
  auto Grid<ONE_D>::loopCells(const long int& i1, const long int& i2) -> ntt_1drange_t {
    return NTT1DRange(  (range_t)(N_GHOSTS + i1),
                      (range_t)(get_imax() + i2));
  }
  template <>
  auto Grid<TWO_D>::loopCells(const long int& i1, const long int& i2,
                              const long int& j1, const long int& j2) -> ntt_2drange_t {
    return NTT2DRange({  N_GHOSTS + i1,   N_GHOSTS + j1},
                      {get_imax() + i2, get_jmax() + j2});
  }
  template <>
  auto Grid<THREE_D>::loopCells(const long int& i1, const long int& i2,
                                const long int& j1, const long int& j2,
                                const long int& k1, const long int& k2) -> ntt_3drange_t {
    return NTT3DRange({  N_GHOSTS + i1,     N_GHOSTS + j1,    N_GHOSTS + k1},
                      {get_imax() + i2,   get_jmax() + j2,  get_kmax() + k2});
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Grid<ONE_D>::loopActiveCells() -> ntt_1drange_t {
    return loopCells(0, 0);
  }
  template <>
  auto Grid<TWO_D>::loopActiveCells() -> ntt_2drange_t {
    return loopCells(0, 0,
                     0, 0);
  }
  template <>
  auto Grid<THREE_D>::loopActiveCells() -> ntt_3drange_t {
    return loopCells(0, 0,
                     0, 0,
                     0, 0);
  }

  // # # # # # # # # # # # # # # # #
  template <>
  auto Grid<ONE_D>::loopAllCells() -> ntt_1drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Grid<TWO_D>::loopAllCells() -> ntt_2drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS,
                     -N_GHOSTS, N_GHOSTS);
  }
  template <>
  auto Grid<THREE_D>::loopAllCells() -> ntt_3drange_t {
    return loopCells(-N_GHOSTS, N_GHOSTS,
                     -N_GHOSTS, N_GHOSTS,
                     -N_GHOSTS, N_GHOSTS);
  }

} // namespace ntt

template struct ntt::Grid<ntt::ONE_D>;
template struct ntt::Grid<ntt::TWO_D>;
template struct ntt::Grid<ntt::THREE_D>;

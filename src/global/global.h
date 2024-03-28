/**
 * @file global.h
 * @brief Global constants, macros, and aliases
 * @implements
 *   - N_GHOSTS
 *   - COORD
 *   - Dimension, Dim, DimensionTag
 *   - prtldx_t, real_t
 *   - tuple_t, list_t, coord_t, vec_t
 *   - index_t, range_tuple_t
 *   - ntt::GlobalInitialize, ntt::GlobalFinalize
 * @cpp:
 *   - global.cpp
 * @namespaces:
 *   - ntt::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_GLOBAL_H
#define GLOBAL_GLOBAL_H

inline constexpr unsigned int N_GHOSTS = 2;

// Coordinate shift to account for ghost cells
#define COORD(I)                                                               \
  (static_cast<real_t>(static_cast<int>((I)) - static_cast<int>(N_GHOSTS)))

#include <utility>

enum Dimension : unsigned short {
  _1D = 1,
  _2D = 2,
  _3D = 3
};

using Dim = Dimension;

template <Dimension D>
struct DimensionTag {};

// Defining precision-based constants and types
using prtldx_t = float;
#if defined(SINGLE_PRECISION)
using real_t = float;
#else
using real_t = double;
#endif

// ND list alias
template <typename T, Dimension D>
using tuple_t = T[D];

// list alias of size N
template <typename T, unsigned short N>
using list_t = T[N];

// ND coordinate alias
template <Dimension D>
using coord_t = tuple_t<real_t, D>;

// ND vector alias
template <Dimension D>
using vec_t = tuple_t<real_t, D>;

using index_t = const std::size_t;

using range_tuple_t = std::pair<std::size_t, std::size_t>;

namespace ntt {

  void GlobalInitialize(int argc, char* argv[]);

  void GlobalFinalize();

} // namespace ntt

#endif // GLOBAL_GLOBAL_H
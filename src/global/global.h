/**
 * @file global.h
 * @brief Global constants, macros, aliases and enums
 * @implements
 *   - ntt::N_GHOSTS
 *   - macro COORD
 *   - macro HERE
 *   - enum Dimension, enum Dim
 *   - enum ntt::em
 *   - enum ntt::cur
 *   - enum ntt::ParticleTag    // dead, alive
 *   - type PrepareOutputFlags
 *   - enum PrepareOutput
 *   - enum CellLayer           // allLayer, activeLayer, minGhostLayer,
 *                                 minActiveLayer, maxActiveLayer, maxGhostLayer
 *   - type box_region_t
 *   - files::LogFile, files::ErrFile, files::InfoFile
 *   - type prtldx_t
 *   - type real_t
 *   - type tuple_t
 *   - type list_t
 *   - type coord_t
 *   - type vec_t
 *   - type index_t
 *   - type range_tuple_t
 *   - ntt::GlobalInitialize -> void
 *   - ntt::GlobalFinalize -> void
 * @cpp:
 *   - global.cpp
 * @namespaces:
 *   - ntt::
 *   - files::
 * @macros:
 *   - MPI_ENABLED
 * @note
 * CellLayer enum:
 *
 * min/max layers have N_GHOSTS cells
 *
 * allLayer:                 .* *|* * * * * * * * *\* *.
 * activeLayer:              .   |* * * * * * * * *\   .
 * minGhostLayer:            .* *|                 \   .
 * minActiveLayer:           .   |* *              \   .
 * maxActiveLayer:           .   |              * *\   .
 * maxGhostLayer:            .   |                 \* *.
 *
 * @example
 * Usage of the CellLayer enum:
 *
 * 1. box_region_t<Dim2>{ CellLayer::minGhostLayer, CellLayer::maxGhostLayer }
 * results in a region [ [ i1min - N_GHOSTS, i1min ),
 *                       [ i2max, i2max + N_GHOSTS ) ]
 * i.e.,
 *    . . . . . . . . . .
 *    .* *              .
 *    .* *              .
 *    .   ^= = = = =^   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   ^- - - - -^   .
 *    .                 .
 *    .                 .
 *    . . . . . . . . . .
 *
 * 2. box_region_t<Dim3>{ CellLayer::activeLayer,
 *                     CellLayer::maxActiveLayer,
 *                     CellLayer::allLayer }
 * results in a region [ [ i1min, i1max ),
 *                       [ i2max - N_GHOSTS, i2max ),
 *                       [ i3min - N_GHOSTS, i3max + N_GHOSTS ) ]
 * i.e., in x1 & x2 slice
 *    . . . . . . . . . .
 *    .                 .
 *    .                 .
 *    .   ^= = = = =^   .
 *    .   |* * * * *\   .
 *    .   |* * * * *\   .
 *    .   |         \   .
 *    .   |         \   .
 *    .   ^- - - - -^   .
 *    .                 .
 *    .                 .
 *    . . . . . . . . . .
 */

#ifndef GLOBAL_GLOBAL_H
#define GLOBAL_GLOBAL_H

#include <utility>
#include <vector>

#pragma nv_diag_suppress 20011

#define HERE __FILE__, __func__, __LINE__

namespace files {
  enum {
    LogFile = 1,
    ErrFile,
    InfoFile
  };
} // namespace files

namespace ntt {

  inline constexpr unsigned int N_GHOSTS = 2;
// Coordinate shift to account for ghost cells
#define COORD(I)                                                               \
  (static_cast<real_t>(static_cast<int>((I)) - static_cast<int>(N_GHOSTS)))

  enum em {
    ex1 = 0,
    ex2 = 1,
    ex3 = 2,
    dx1 = 0,
    dx2 = 1,
    dx3 = 2,
    bx1 = 3,
    bx2 = 4,
    bx3 = 5,
    hx1 = 3,
    hx2 = 4,
    hx3 = 5
  };

  enum cur {
    jx1 = 0,
    jx2 = 1,
    jx3 = 2
  };

  enum ParticleTag : short {
    dead = 0,
    alive
  };

  void GlobalInitialize(int argc, char* argv[]);

  void GlobalFinalize();

} // namespace ntt

/* global scope enums & aliases --------------------------------------------- */

enum Dimension : unsigned short {
  _1D = 1,
  _2D = 2,
  _3D = 3
};

enum class CellLayer {
  allLayer,
  activeLayer,
  minGhostLayer,
  minActiveLayer,
  maxActiveLayer,
  maxGhostLayer
};

enum class Idx {
  U,   // contravariant
  D,   // covariant
  T,   // tetrad
  XYZ, // Cartesian
  Sph, // spherical
  PU,  // physical contravariant
  PD,  // physical covariant
};

enum class Crd {
  Cd,  // code units
  Ph,  // physical units
  XYZ, // Cartesian
  Sph, // spherical
};

template <Dimension D>
using box_region_t = CellLayer[D];

namespace PrepareOutput {
  enum PrepareOutputFlags_ {
    None                        = 0,
    InterpToCellCenterFromEdges = 1 << 0,
    InterpToCellCenterFromFaces = 1 << 1,
    ConvertToHat                = 1 << 2,
    ConvertToPhysCntrv          = 1 << 3,
    ConvertToPhysCov            = 1 << 4,
  };
} // namespace PrepareOutput

typedef int PrepareOutputFlags;

using Dim = Dimension;

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
using idx_t   = unsigned short;

using range_tuple_t = std::pair<std::size_t, std::size_t>;

template <typename T>
using boundaries_t = std::vector<std::pair<T, T>>;

#endif // GLOBAL_GLOBAL_H
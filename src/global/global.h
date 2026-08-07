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
 *   - enum Idx                // U, D, T, XYZ, Sph, PU, PD
 *   - enum Crd                // Cd, Ph, XYZ, Sph
 *   - enum in                 // x1, x2, x3
 *   - enum bc_in                // Px1, Mx1, Px2, Mx2, Px3, Mx3
 *   - type box_region_t
 *   - files::LogFile, files::ErrFile, files::InfoFile
 *   - type prtldx_t
 *   - type real_t
 *   - type tuple_t
 *   - type list_t
 *   - type coord_t
 *   - type vec_t
 *   - type duration_t
 *   - type simtime_t
 *   - type timestep_t
 *   - type ncells_t
 *   - type npart_t
 *   - type timestamp_t
 *   - type cellidx_t
 *   - type prtlidx_t
 *   - type idx_t
 *   - type spidx_t
 *   - type dim_t
 *   - type path_t
 *   - type cell_range_t
 *   - type prtl_slice_t
 *   - type boundaries_t
 *   - ntt::GlobalInitialize -> void
 *   - ntt::GlobalFinalize -> void
 *   - enum ntt::DiagFlags
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

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <utility>
#include <vector>

#pragma nv_diag_suppress 20011
#pragma nv_diag_suppress 20013

#define HERE __FILE__, __func__, __LINE__

namespace files {
  enum : uint8_t {
    LogFile = 1,
    ErrFile,
    InfoFile
  };
} // namespace files

namespace ntt {

#if !defined(SHAPE_ORDER)
  #define SHAPE_ORDER 0
  inline constexpr uint32_t N_GHOSTS = 2;
#else  // SHAPE_ORDER
  inline constexpr uint32_t N_GHOSTS = static_cast<uint32_t>((SHAPE_ORDER + 1) / 2) +
                                       1;
#endif // SHAPE_ORDER

// Coordinate shift to account for ghost cells
#define COORD(I)                                                               \
  (static_cast<real_t>(static_cast<int>((I)) - static_cast<int>(N_GHOSTS)))

  enum em : uint8_t {
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

  enum cur : uint8_t {
    jx1 = 0,
    jx2 = 1,
    jx3 = 2
  };

  enum pldi : uint8_t {
    spcCtr = 0,
    domIdx = 1
  };

  enum ParticleTag : short { // NOLINT
    dead = 0,
    alive
  };

  void GlobalInitialize(int argc, char* argv[]);

  void GlobalFinalize();

} // namespace ntt

/* global scope enums & aliases --------------------------------------------- */

enum Dimension : uint8_t {
  _1D = 1,
  _2D = 2,
  _3D = 3,
  _4D = 4
};

enum class CellLayer : uint8_t {
  allLayer,
  activeLayer,
  minGhostLayer,
  minActiveLayer,
  maxActiveLayer,
  maxGhostLayer
};

enum class Idx : uint8_t {
  U,   // contravariant
  D,   // covariant
  T,   // tetrad
  XYZ, // Cartesian
  Sph, // spherical
  PU,  // physical contravariant
  PD,  // physical covariant
};

enum class Crd : uint8_t {
  Cd,  // code units
  Ph,  // physical units
  XYZ, // Cartesian
  Sph, // spherical
};

enum class in : uint8_t {
  x1 = 0,
  x2 = 1,
  x3 = 2,
};

enum class bc_in : int8_t {
  Mx1 = -1,
  Px1 = 1,
  Mx2 = -2,
  Px2 = 2,
  Mx3 = -3,
  Px3 = 3,
};

template <Dimension D>
using box_region_t = CellLayer[D];

/* config flags ------------------------------------------------------------- */

namespace PrepareOutput {
  enum PrepareOutputFlags_ : uint8_t {
    None                        = 0,
    InterpToCellCenterFromEdges = 1 << 0,
    InterpToCellCenterFromFaces = 1 << 1,
    ConvertToHat                = 1 << 2,
    ConvertToPhysCntrv          = 1 << 3,
    ConvertToPhysCov            = 1 << 4,
  };
} // namespace PrepareOutput

using PrepareOutputFlags = uint8_t;

namespace Timer {
  enum TimerFlags_ : uint8_t {
    None              = 0,
    PrintTotal        = 1 << 0,
    PrintTitle        = 1 << 1,
    AutoConvert       = 1 << 2,
    PrintOutput       = 1 << 3,
    PrintParticleSort = 1 << 4,
    PrintCheckpoint   = 1 << 5,
    PrintNormed       = 1 << 6,
    PrintAscent       = 1 << 7,
    Default           = PrintNormed | PrintTotal | PrintTitle | AutoConvert,
  };
} // namespace Timer

using TimerFlags = uint8_t;

namespace Diag {
  enum DiagFlags_ : uint8_t {
    None     = 0,
    Progress = 1 << 0,
    Timers   = 1 << 1,
    Species  = 1 << 2,
    Colorful = 1 << 3,
    Default  = Progress | Timers | Species | Colorful,
  };

} // namespace Diag

using DiagFlags = uint8_t;

namespace Comm {
  enum CommTags_ : uint16_t {
    None = 0,
    E    = 1 << 0,
    B    = 1 << 1,
    J    = 1 << 2,
    D    = 1 << 3,
    D0   = 1 << 4,
    B0   = 1 << 5,
    H    = 1 << 6,
    Bckp = 1 << 7,
    Buff = 1 << 8,
  };
} // namespace Comm

using CommTags = uint16_t;

namespace WriteMode {
  enum WriteModeTags_ : uint8_t {
    None      = 0,
    Fields    = 1 << 0,
    Particles = 1 << 1,
    Spectra   = 1 << 2,
    Stats     = 1 << 3,
  };
} // namespace WriteMode

using WriteModeTags = uint8_t;

namespace BC {
  enum BCTags_ : uint16_t {
    None = 0,
    Ex1  = 1 << 0,
    Ex2  = 1 << 1,
    Ex3  = 1 << 2,
    Bx1  = 1 << 3,
    Bx2  = 1 << 4,
    Bx3  = 1 << 5,
    Dx1  = 1 << 0,
    Dx2  = 1 << 1,
    Dx3  = 1 << 2,
    Hx1  = 1 << 3,
    Hx2  = 1 << 4,
    Hx3  = 1 << 5,
    Jx1  = 1 << 6,
    Jx2  = 1 << 7,
    Jx3  = 1 << 8,
    B    = Bx1 | Bx2 | Bx3,
    E    = Ex1 | Ex2 | Ex3,
    D    = Dx1 | Dx2 | Dx3,
    H    = Hx1 | Hx2 | Hx3,
    J    = Jx1 | Jx2 | Jx3,
  };
} // namespace BC

using BCTags = uint16_t;

namespace Inj {
  enum InjTags_ : uint8_t {
    None        = 0,
    AssumeEmpty = 1 << 0,
  };
} // namespace Inj

using InjTags = uint8_t;

/* aliases ------------------------------------------------------------------ */

using Dim = Dimension;

// Defining precision-based constants and types
using prtldx_t = float;
#if defined(SINGLE_PRECISION)
using real_t = float;
#else
using real_t = double;
#endif

namespace Range {
  constexpr std::pair<real_t, real_t> All = {
    -std::numeric_limits<real_t>::infinity(),
    std::numeric_limits<real_t>::infinity()
  };
  constexpr real_t Min = -std::numeric_limits<real_t>::infinity();
  constexpr real_t Max = std::numeric_limits<real_t>::infinity();
}; // namespace Range

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

// time/duration
using duration_t = double;
using simtime_t  = double;
using timestep_t = uint32_t;
using ncells_t   = uint32_t;
using npart_t    = uint32_t;

// walltime
using timestamp_t = std::chrono::time_point<std::chrono::system_clock>;

// index/number
using cellidx_t = const ncells_t;
using prtlidx_t = const npart_t;
using idx_t     = uint8_t;
using spidx_t   = uint8_t;
using dim_t     = uint8_t;

// utility
using path_t = std::filesystem::path;

using cell_range_t = std::pair<ncells_t, ncells_t>;
using prtl_slice_t = std::pair<npart_t, npart_t>;

template <typename T>
using boundaries_t = std::vector<std::pair<T, T>>;

#endif // GLOBAL_GLOBAL_H

#include "arch/directions.h"

#include "global.h"

#include "utils/error.h"

#include <algorithm>
#include <string>

auto main() -> int {
  using namespace ntt;

  {
    using dir1D = dir::Directions<Dim::_1D>;
    raise::ErrorIf(dir1D::all.size() != 2, "Wrong number of ::all in 1D", HERE);
    raise::ErrorIf(dir1D::unique.size() != 1, "Wrong number of ::unique in 1D", HERE);

    short i0 = 0;
    for (const auto& dir : dir1D::all) {
      auto iold  = i0;
      i0        += dir[0];
      raise::ErrorIf(i0 == iold, "Wrong ::all in 1D", HERE);
    }
    raise::ErrorIf(i0 != 0, "Wrong ::all in 1D", HERE);

    for (auto d = 0u; d < dir1D::unique.size(); ++d) {
      raise::ErrorIf(
        std::find(dir1D::all.begin(), dir1D::all.end(), dir1D::unique[d]) ==
          dir1D::all.end(),
        "Wrong ::unique in 1D",
        HERE);
      for (auto dd = 0u; dd < dir1D::unique.size(); ++dd) {
        raise::ErrorIf(
          (d != dd) & ((dir1D::unique[d] == dir1D::unique[dd]) ||
                       (dir1D::unique[d] == -dir1D::unique[dd])),
          "Wrong ::unique in 1D",
          HERE);
      }
    }
  }

  {
    using dir2D = dir::Directions<Dim::_2D>;
    raise::ErrorIf(dir2D::all.size() != 8, "Wrong number of ::all in 2D", HERE);
    raise::ErrorIf(dir2D::unique.size() != 4, "Wrong number of ::unique in 2D", HERE);

    short i0 = 0, j0 = 0;
    for (const auto& dir : dir2D::all) {
      auto iold = i0, jold = j0;
      i0 += dir[0];
      j0 += dir[1];
      raise::ErrorIf((i0 == iold) && (j0 == jold), "Wrong ::all in 2D", HERE);
    }
    raise::ErrorIf((i0 != 0) || (j0 != 0), "Wrong ::all in 2D", HERE);

    for (auto d = 0u; d < dir2D::unique.size(); ++d) {
      raise::ErrorIf(
        std::find(dir2D::all.begin(), dir2D::all.end(), dir2D::unique[d]) ==
          dir2D::all.end(),
        "Wrong ::unique in 2D",
        HERE);
      for (auto dd = 0u; dd < dir2D::unique.size(); ++dd) {
        raise::ErrorIf(
          (d != dd) & ((dir2D::unique[d] == dir2D::unique[dd]) ||
                       (dir2D::unique[d] == -dir2D::unique[dd])),
          "Wrong ::unique in 2D",
          HERE);
      }
    }
  }

  {
    using dir3D = dir::Directions<Dim::_3D>;
    raise::ErrorIf(dir3D::all.size() != 26, "Wrong number of ::all in 3D", HERE);
    raise::ErrorIf(dir3D::unique.size() != 13, "Wrong number of ::unique in 3D", HERE);

    short i0 = 0, j0 = 0, k0 = 0;
    for (const auto& dir : dir3D::all) {
      auto iold = i0, jold = j0, kold = k0;
      i0 += dir[0];
      j0 += dir[1];
      k0 += dir[2];
      raise::ErrorIf((i0 == iold) && (j0 == jold) && (k0 == kold),
                     "Wrong ::all in 3D",
                     HERE);
    }
    raise::ErrorIf((i0 != 0) || (j0 != 0) || (k0 != 0), "Wrong ::all in 3D", HERE);

    for (auto d = 0u; d < dir3D::unique.size(); ++d) {
      raise::ErrorIf(
        std::find(dir3D::all.begin(), dir3D::all.end(), dir3D::unique[d]) ==
          dir3D::all.end(),
        "Wrong ::unique in 3D",
        HERE);
      for (auto dd = 0u; dd < dir3D::unique.size(); ++dd) {
        raise::ErrorIf(
          (d != dd) & ((dir3D::unique[d] == dir3D::unique[dd]) ||
                       (dir3D::unique[d] == -dir3D::unique[dd])),
          "Wrong ::unique in 3D",
          HERE);
      }
    }
  }

  return 0;
}

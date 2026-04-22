#include "arch/directions.h"

#include "global.h"

#include <algorithm>
#include <stdexcept>
#include <string>

void errorIf(bool condition, const std::string& message) {
  if (condition) {
    throw std::runtime_error(message);
  }
}

auto main() -> int {
  using namespace ntt;

  {
    using dir1D = dir::Directions<Dim::_1D>;
    errorIf(dir1D::all.size() != 2, "Wrong number of ::all in 1D");
    errorIf(dir1D::unique.size() != 1, "Wrong number of ::unique in 1D");

    short i0 = 0;
    for (const auto& dir : dir1D::all) {
      auto iold  = i0;
      i0        += dir[0];
      errorIf(i0 == iold, "Wrong ::all in 1D");
    }
    errorIf(i0 != 0, "Wrong ::all in 1D");

    for (auto d = 0u; d < dir1D::unique.size(); ++d) {
      errorIf(std::find(dir1D::all.begin(), dir1D::all.end(), dir1D::unique[d]) ==
                dir1D::all.end(),
              "Wrong ::unique in 1D");
      for (auto dd = 0u; dd < dir1D::unique.size(); ++dd) {
        errorIf((d != dd) & ((dir1D::unique[d] == dir1D::unique[dd]) ||
                             (dir1D::unique[d] == -dir1D::unique[dd])),
                "Wrong ::unique in 1D");
      }
    }
  }

  {
    using dir2D = dir::Directions<Dim::_2D>;
    errorIf(dir2D::all.size() != 8, "Wrong number of ::all in 2D");
    errorIf(dir2D::unique.size() != 4, "Wrong number of ::unique in 2D");

    short i0 = 0, j0 = 0;
    for (const auto& dir : dir2D::all) {
      auto iold = i0, jold = j0;
      i0 += dir[0];
      j0 += dir[1];
      errorIf((i0 == iold) && (j0 == jold), "Wrong ::all in 2D");
    }
    errorIf((i0 != 0) || (j0 != 0), "Wrong ::all in 2D");

    for (auto d = 0u; d < dir2D::unique.size(); ++d) {
      errorIf(std::find(dir2D::all.begin(), dir2D::all.end(), dir2D::unique[d]) ==
                dir2D::all.end(),
              "Wrong ::unique in 2D");
      for (auto dd = 0u; dd < dir2D::unique.size(); ++dd) {
        errorIf((d != dd) & ((dir2D::unique[d] == dir2D::unique[dd]) ||
                             (dir2D::unique[d] == -dir2D::unique[dd])),
                "Wrong ::unique in 2D");
      }
    }
  }

  {
    using dir3D = dir::Directions<Dim::_3D>;
    errorIf(dir3D::all.size() != 26, "Wrong number of ::all in 3D");
    errorIf(dir3D::unique.size() != 13, "Wrong number of ::unique in 3D");

    short i0 = 0, j0 = 0, k0 = 0;
    for (const auto& dir : dir3D::all) {
      auto iold = i0, jold = j0, kold = k0;
      i0 += dir[0];
      j0 += dir[1];
      k0 += dir[2];
      errorIf((i0 == iold) && (j0 == jold) && (k0 == kold), "Wrong ::all in 3D");
    }
    errorIf((i0 != 0) || (j0 != 0) || (k0 != 0), "Wrong ::all in 3D");

    for (auto d = 0u; d < dir3D::unique.size(); ++d) {
      errorIf(std::find(dir3D::all.begin(), dir3D::all.end(), dir3D::unique[d]) ==
                dir3D::all.end(),
              "Wrong ::unique in 3D");
      for (auto dd = 0u; dd < dir3D::unique.size(); ++dd) {
        errorIf((d != dd) & ((dir3D::unique[d] == dir3D::unique[dd]) ||
                             (dir3D::unique[d] == -dir3D::unique[dd])),
                "Wrong ::unique in 3D");
      }
    }
  }

  return 0;
}
#include "framework/containers/fields.h"

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <Dimension D, ntt::SimEngine::type S>
void testFields(const std::vector<std::size_t>& res) {
  using namespace ntt;
  raise::ErrorIf(res.size() != D, "Resolution vector has wrong size", HERE);
  auto       f   = Fields<D, S> { res };
  const auto sx1 = res[0] + 2 * N_GHOSTS;
  raise::ErrorIf(f.em.extent(0) != sx1, "EM field has wrong size [0]", HERE);
  raise::ErrorIf(f.bckp.extent(0) != sx1, "Backup field has wrong size [0]", HERE);
  raise::ErrorIf(f.cur.extent(0) != sx1, "Current field has wrong size [0]", HERE);
  raise::ErrorIf(f.buff.extent(0) != sx1, "Buffer field has wrong size [0]", HERE);

  if constexpr (D == Dim::_1D) {
    // 1D
    raise::ErrorIf(f.em.extent(1) != 6, "EM field has wrong size [1]", HERE);
    raise::ErrorIf(f.bckp.extent(1) != 6, "Backup field has wrong size [1]", HERE);
    raise::ErrorIf(f.cur.extent(1) != 3, "Current field has wrong size [1]", HERE);
    raise::ErrorIf(f.buff.extent(1) != 3, "Buffer field has wrong size [1]", HERE);
  } else {
    // 2D or 3D
    const auto sx2 = res[1] + 2 * N_GHOSTS;
    raise::ErrorIf(f.em.extent(1) != sx2, "EM field has wrong size [1]", HERE);
    raise::ErrorIf(f.bckp.extent(1) != sx2, "Backup field has wrong size [1]", HERE);
    raise::ErrorIf(f.cur.extent(1) != sx2, "Current field has wrong size [1]", HERE);
    raise::ErrorIf(f.buff.extent(1) != sx2, "Buffer field has wrong size [1]", HERE);
    if constexpr (S == SimEngine::GRPIC) {
      raise::ErrorIf(f.aux.extent(0) != sx1, "Aux field has wrong size [0]", HERE);
      raise::ErrorIf(f.aux.extent(1) != sx2, "Aux field has wrong size [1]", HERE);
      raise::ErrorIf(f.em0.extent(0) != sx1, "EM0 field has wrong size [0]", HERE);
      raise::ErrorIf(f.em0.extent(1) != sx2, "EM0 field has wrong size [1]", HERE);
      raise::ErrorIf(f.cur0.extent(0) != sx1, "CUR0 field has wrong size [0]", HERE);
      raise::ErrorIf(f.cur0.extent(1) != sx2, "CUR0 field has wrong size [1]", HERE);
    }
  }

  if constexpr (D == Dim::_2D) {
    // 2D
    raise::ErrorIf(f.em.extent(2) != 6, "EM field has wrong size [2]", HERE);
    raise::ErrorIf(f.bckp.extent(2) != 6, "Backup field has wrong size [2]", HERE);
    raise::ErrorIf(f.cur.extent(2) != 3, "Current field has wrong size [2]", HERE);
    raise::ErrorIf(f.buff.extent(2) != 3, "Buffer field has wrong size [2]", HERE);
    if constexpr (S == SimEngine::GRPIC) {
      raise::ErrorIf(f.aux.extent(2) != 6, "Aux field has wrong size [2]", HERE);
      raise::ErrorIf(f.em0.extent(2) != 6, "EM0 field has wrong size [2]", HERE);
      raise::ErrorIf(f.cur0.extent(2) != 3, "CUR0 field has wrong size [2]", HERE);
    }
  } else if constexpr (D == Dim::_3D) {
    // 3D
    const auto sx3 = res[2] + 2 * N_GHOSTS;
    raise::ErrorIf(f.em.extent(2) != sx3, "EM field has wrong size [2]", HERE);
    raise::ErrorIf(f.bckp.extent(2) != sx3, "Backup field has wrong size [2]", HERE);
    raise::ErrorIf(f.cur.extent(2) != sx3, "Current field has wrong size [2]", HERE);
    raise::ErrorIf(f.buff.extent(2) != sx3, "Buffer field has wrong size [2]", HERE);

    raise::ErrorIf(f.em.extent(3) != 6, "EM field has wrong size [3]", HERE);
    raise::ErrorIf(f.bckp.extent(3) != 6, "Backup field has wrong size [3]", HERE);
    raise::ErrorIf(f.cur.extent(3) != 3, "Current field has wrong size [3]", HERE);
    raise::ErrorIf(f.buff.extent(3) != 3, "Buffer field has wrong size [3]", HERE);

    if constexpr (S == SimEngine::GRPIC) {
      raise::ErrorIf(f.aux.extent(2) != sx3, "Aux field has wrong size [2]", HERE);
      raise::ErrorIf(f.em0.extent(2) != sx3, "EM0 field has wrong size [2]", HERE);
      raise::ErrorIf(f.cur0.extent(2) != sx3, "CUR0 field has wrong size [2]", HERE);
      raise::ErrorIf(f.aux.extent(3) != 6, "Aux field has wrong size [3]", HERE);
      raise::ErrorIf(f.em0.extent(3) != 6, "EM0 field has wrong size [3]", HERE);
      raise::ErrorIf(f.cur0.extent(3) != 3, "CUR0 field has wrong size [3]", HERE);
    }
  }
}

auto main(int argc, char** argv) -> int {
  using namespace ntt;
  GlobalInitialize(argc, argv);
  try {
    testFields<Dim::_1D, SimEngine::type::SRPIC>({ 10 });
    testFields<Dim::_2D, SimEngine::type::SRPIC>({ 10, 20 });
    testFields<Dim::_3D, SimEngine::type::SRPIC>({ 10, 20, 30 });
    testFields<Dim::_2D, SimEngine::type::GRPIC>({ 10, 20 });
    testFields<Dim::_3D, SimEngine::type::GRPIC>({ 10, 20, 30 });
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    GlobalFinalize();
    return 1;
  }
  GlobalFinalize();
  return 0;
}
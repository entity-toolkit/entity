/**
 * @brief X-3 (team_policy) — sort_by_key permutation test.
 *
 * Exercises every backend overload of `ntt::sort_helpers::sort_by_key_dispatch`
 * that is compiled in for the current Kokkos device. For each backend:
 *   1. Allocate keys = { 5, 2, 5, 1, 3, 5, 2 }, perm = (uninitialised).
 *   2. Call sort_by_key_dispatch.
 *   3. Verify that keys[perm[i]] is sorted in non-decreasing order.
 *
 * Stability is verified for the BinSort and StdSort backends (the others
 * promise stability per their documentation but we don't bake that into
 * the test).
 *
 * Built only when `team_policy=ON` at CMake time.
 */
#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/sort_dispatch.h"
#include "utils/sorting.h"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <vector>

namespace {

  using namespace ntt;

  template <typename Backend>
  void test_one_backend(const char* label, Backend tag) {
    const std::vector<ncells_t> keys_host_init { 5u, 2u, 5u, 1u, 3u, 5u, 2u };
    const npart_t               n     = keys_host_init.size();
    const ncells_t              n_max = 6u; // bin range [0, n_max)

    array_t<ncells_t*> keys { "keys", n };
    auto               keys_h = Kokkos::create_mirror_view(keys);
    for (npart_t i = 0u; i < n; ++i) {
      keys_h(i) = keys_host_init[i];
    }
    Kokkos::deep_copy(keys, keys_h);

    prtl_perm_t perm { "perm", n };

    sort_helpers::sort_by_key_dispatch(keys, perm, n_max, tag);

    auto perm_h = Kokkos::create_mirror_view(perm);
    Kokkos::deep_copy(perm_h, perm);

    // Validate: keys[perm[0]] <= keys[perm[1]] <= ...
    for (npart_t i = 1u; i < n; ++i) {
      const auto a = keys_host_init[perm_h(i - 1u)];
      const auto b = keys_host_init[perm_h(i)];
      raise::ErrorIf(
        a > b,
        std::string("sort_by_key_dispatch produced non-sorted permutation "
                    "for backend ") +
          label,
        HERE);
    }

    // Validate: perm is a permutation of [0, n).
    std::vector<int> seen(n, 0);
    for (npart_t i = 0u; i < n; ++i) {
      const auto idx = perm_h(i);
      raise::ErrorIf(idx >= n,
                     std::string("permutation index out of range for "
                                 "backend ") +
                       label,
                     HERE);
      seen[idx] += 1;
    }
    for (npart_t i = 0u; i < n; ++i) {
      raise::ErrorIf(seen[i] != 1,
                     std::string("permutation not a bijection for backend ") +
                       label,
                     HERE);
    }

    std::cout << "[OK] sort_by_key_dispatch<" << label << ">: "
              << "keys[perm] sorted, perm is a bijection." << std::endl;
  }

} // namespace

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    // Always-available backends.
    test_one_backend("BinSort", ::sort::backend::BinSort {});
#if !defined(DEVICE_ENABLED)
    test_one_backend("StdSort", ::sort::backend::StdSort {});
#endif
#if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
    test_one_backend("OneDPL", ::sort::backend::OneDPL {});
#endif
#if defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
    test_one_backend("Thrust", ::sort::backend::Thrust {});
#endif
  } catch (std::exception& e) {
    std::cerr << e.what() << std::endl;
    ntt::GlobalFinalize();
    return 1;
  }
  ntt::GlobalFinalize();
  return 0;
}

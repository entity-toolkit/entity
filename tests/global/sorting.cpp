#include "utils/sorting.h"

#include "global.h"

#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

auto main(int argc, char* argv[]) -> int {
  Kokkos::initialize(argc, argv);
  try {
    constexpr std::size_t n          = 100;
    auto                  keys_short = Kokkos::View<short*>("keys", n);
    auto                  keys_bool  = Kokkos::View<bool*>("bool", n);
    auto                  values     = Kokkos::View<int*>("values", n);
    // init keys and values
    Kokkos::parallel_for(
      n,
      KOKKOS_LAMBDA(const std::size_t i) {
        keys_short(i) = static_cast<short>(i % 6);
        keys_bool(i)  = static_cast<bool>(i % 2);
        values(i)     = static_cast<int>(i);
      });

    // init sorters
    using KeyType_short = Kokkos::View<short*>;
    using KeyType_bool  = Kokkos::View<bool*>;
    using BinOp_short   = sort::BinTag<KeyType_short>;
    using BinOp_bool    = sort::BinBool<KeyType_bool>;
    BinOp_short bin_op_short(6);
    BinOp_bool  bin_op_bool;

    Kokkos::BinSort<KeyType_short, BinOp_short> sorter_short(keys_short,
                                                             bin_op_short,
                                                             false);
    Kokkos::BinSort<KeyType_bool, BinOp_bool>   sorter_bool(keys_bool,
                                                          bin_op_bool,
                                                          false);
    sorter_short.create_permute_vector();

    // sort with short
    sorter_short.sort(keys_short);
    sorter_short.sort(keys_bool);
    sorter_short.sort(values);

    Kokkos::parallel_for(
      n,
      KOKKOS_LAMBDA(const std::size_t i) {
        auto should_raise = false;
        if (i < 17) {
          should_raise = (values(i) % 6 != 1);
          should_raise = should_raise || (keys_bool(i) != 1);
          should_raise = should_raise || (keys_short(i) != 1);
        } else if (i < 34) {
          should_raise = (values(i) % 6 != 0);
          should_raise = should_raise || (keys_bool(i) != 0);
          should_raise = should_raise || (keys_short(i) != 0);
        } else if (i < 51) {
          should_raise = (values(i) % 6 != 2);
          should_raise = should_raise || (keys_bool(i) != 0);
          should_raise = should_raise || (keys_short(i) != 2);
        } else if (i < 68) {
          should_raise = (values(i) % 6 != 3);
          should_raise = should_raise || (keys_bool(i) != 1);
          should_raise = should_raise || (keys_short(i) != 3);
        } else if (i < 84) {
          should_raise = (values(i) % 6 != 4);
          should_raise = should_raise || (keys_bool(i) != 0);
          should_raise = should_raise || (keys_short(i) != 4);
        } else {
          should_raise = (values(i) % 6 != 5);
          should_raise = should_raise || (keys_bool(i) != 1);
          should_raise = should_raise || (keys_short(i) != 5);
        }
        if (should_raise) {
          raise::KernelError(HERE, "Short sort failed");
        }
      });

    sorter_bool.create_permute_vector();

    sorter_bool.sort(keys_short);
    sorter_bool.sort(keys_bool);
    sorter_bool.sort(values);

    Kokkos::parallel_for(
      n,
      KOKKOS_LAMBDA(const std::size_t i) {
        auto should_raise = false;
        if (i < 50) {
          should_raise = (values(i) % 2 != 0);
          should_raise = should_raise || (keys_bool(i) != 0);
          should_raise = should_raise || (keys_short(i) % 2 != 0);
        } else {
          should_raise = (values(i) % 2 != 1);
          should_raise = should_raise || (keys_bool(i) != 1);
          should_raise = should_raise || (keys_short(i) % 2 != 1);
        }
        if (should_raise) {
          raise::KernelError(HERE, "Short sort failed");
        }
      });
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    Kokkos::finalize();
    return 1;
  }
  Kokkos::finalize();
  return 0;
}

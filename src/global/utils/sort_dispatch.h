/**
 * @file utils/sort_dispatch.h
 * @brief Backend-dispatched sort_by_key for team_policy SortSpatially.
 * @implements
 *   - sort_helpers::sort_by_key_dispatch -> void  (BinSort, OneDPL, Thrust, StdSort)
 * @namespaces:
 *   - ntt::sort_helpers::
 * @macros:
 *   - TEAM_POLICY
 *   - SYCL_ENABLED, ONEDPL_ENABLED  (oneDPL overload)
 *   - CUDA_ENABLED, THRUST_ENABLED  (Thrust overload)
 *
 * @note Each overload produces a permutation `perm` of size N such that
 *       keys[perm[0]] <= keys[perm[1]] <= ... in stable order.
 *       Always-available overloads: BinSort (uses Kokkos::BinSort) and
 *       StdSort (host-side std::stable_sort fallback). The vendor-library
 *       overloads (OneDPL on SYCL, cub radix sort on CUDA, rocprim radix
 *       sort on HIP) are conditional on the respective build flags.
 * @note The CUDA/HIP overloads bound the radix sort to the significant
 *       key bits (`significant_bits(n_bins)`) instead of the full 32, so
 *       only ceil(log2(n_bins)) bits are sorted — fewer radix passes than
 *       a full-width `thrust::sort_by_key`. Scratch is transient (freed at
 *       scope exit); no persistent buffer is retained (cf. 787aa045).
 */

#ifndef GLOBAL_UTILS_SORT_DISPATCH_H
#define GLOBAL_UTILS_SORT_DISPATCH_H

#if !defined(TEAM_POLICY)
  #error "sort_dispatch.h is only meaningful when TEAM_POLICY is defined"
#endif

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/sorting.h"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

// Entity's Kokkos alias macros (arch/kokkos_aliases.h) define bare words such
// as `Function`, `Inline`, `Lambda` and `ClassLambda`. These collide with
// template-parameter and member names used inside the vendor sort headers
// (rocPRIM, cub, oneDPL) and corrupt their parsing (e.g. rocPRIM's
// `template<class Tuple, class Function, ...>`). Suspend the aliases across the
// vendor includes only, then restore them for the rest of the translation unit.
#pragma push_macro("Function")
#pragma push_macro("Inline")
#pragma push_macro("Lambda")
#pragma push_macro("ClassLambda")
#undef Function
#undef Inline
#undef Lambda
#undef ClassLambda

#if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/execution>
#endif
#if defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
  #include <cub/device/device_radix_sort.cuh>
#endif
#if defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
  #include <rocprim/rocprim.hpp>
#endif

#pragma pop_macro("ClassLambda")
#pragma pop_macro("Lambda")
#pragma pop_macro("Inline")
#pragma pop_macro("Function")

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace ntt::sort_helpers {

  // Number of low-order bits needed to represent keys in [0, n_bins).
  // The radix sort only needs to scan these bits — bounding `end_bit` to
  // ceil(log2(n_bins)) instead of 32 cuts the number of passes (e.g. 18
  // bits when total_tiles ~ 176K). Returns at least 1.
  inline unsigned int significant_bits(ncells_t n_bins) {
    unsigned int bits = 0u;
    while (bits < 32u &&
           (static_cast<std::uint64_t>(1u) << bits) <
             static_cast<std::uint64_t>(n_bins)) {
      ++bits;
    }
    return (bits == 0u) ? 1u : bits;
  }

  // Always-available legacy fallback: Kokkos::BinSort. n_bins must be an
  // upper bound on distinct key values.
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t                  n_bins,
                                   ::sort::backend::BinSort) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    using sorter_op_t = Kokkos::BinOp1D<array_t<ncells_t*>>;
    using sorter_t    = Kokkos::BinSort<array_t<ncells_t*>, sorter_op_t>;
    auto bin_op       = sorter_op_t { static_cast<int>(n_bins), 0u, n_bins };
    auto sorter       = sorter_t { keys, bin_op, false };
    sorter.create_permute_vector();
    auto perm_v = perm;
    Kokkos::parallel_for(
      "PermInitIota",
      n,
      KOKKOS_LAMBDA(const npart_t i) { perm_v(i) = i; });
    Kokkos::fence("sort_by_key_dispatch BinSort: pre-sort");
    sorter.sort(perm);
    Kokkos::fence("sort_by_key_dispatch BinSort: post-sort");
  }

#if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t /*n_bins*/,
                                   ::sort::backend::OneDPL) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    auto*  keys_ptr = keys.data();
    auto*  perm_ptr = perm.data();
    auto   exec     = Kokkos::DefaultExecutionSpace();
    auto   perm_v   = perm;
    Kokkos::parallel_for(
      "PermInitIota",
      n,
      KOKKOS_LAMBDA(const npart_t i) { perm_v(i) = i; });
    // Drain Kokkos's queue so oneDPL's policy sees the iota'd perm even
    // if oneDPL submits to a different SYCL queue internally.
    exec.fence("sort_by_key_dispatch OneDPL: pre-sort");
    auto queue  = exec.sycl_queue();
    auto policy = oneapi::dpl::execution::make_device_policy(queue);
    oneapi::dpl::sort_by_key(policy, keys_ptr, keys_ptr + n, perm_ptr);
    exec.fence("sort_by_key_dispatch OneDPL: post-sort");
  }
#endif

#if defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t                  n_bins,
                                   ::sort::backend::Thrust) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    auto exec   = Kokkos::DefaultExecutionSpace();
    auto perm_v = perm;
    Kokkos::parallel_for(
      "PermInitIota",
      n,
      KOKKOS_LAMBDA(const npart_t i) { perm_v(i) = i; });

    // Radix sort bounded to the significant key bits. The _out buffers and
    // temp storage are transient (freed at scope exit).
    array_t<ncells_t*> keys_out("tile_keys_sorted", n);
    prtl_perm_t        perm_out("tile_perm_sorted", n);
    const int          end_bit = static_cast<int>(significant_bits(n_bins));

    exec.fence("sort_by_key_dispatch Thrust: pre-sort");
    auto stream = exec.cuda_stream();

    // DoubleBuffer radix sort: cub ping-pongs between the supplied
    // (current, alternate) buffer pairs, so `temp_bytes` holds only the
    // histograms (~MB) instead of an internal N-sized alternate (~8*N bytes).
    // Nearly halves the sort's transient memory vs the plain out-of-place
    // form, which matters at high npart where the N-sized temp can fail to
    // allocate (device OOM at scale).
    cub::DoubleBuffer<ncells_t> d_keys(keys.data(), keys_out.data());
    cub::DoubleBuffer<npart_t>  d_perm(perm.data(), perm_out.data());

    std::size_t temp_bytes = 0;
    auto        err = cub::DeviceRadixSort::SortPairs(nullptr,
                                              temp_bytes,
                                              d_keys,
                                              d_perm,
                                              n,
                                              0,
                                              end_bit,
                                              stream);
    raise::ErrorIf(err != cudaSuccess,
                   "cub::DeviceRadixSort::SortPairs (size query) failed",
                   HERE);
    array_t<char*> temp("cub_radix_temp",
                        (temp_bytes == 0u) ? std::size_t { 1 } : temp_bytes);
    err = cub::DeviceRadixSort::SortPairs(temp.data(),
                                          temp_bytes,
                                          d_keys,
                                          d_perm,
                                          n,
                                          0,
                                          end_bit,
                                          stream);
    raise::ErrorIf(err != cudaSuccess,
                   "cub::DeviceRadixSort::SortPairs failed",
                   HERE);
    exec.fence("sort_by_key_dispatch Thrust: post-sort");

    // Publish results from whichever buffer cub left as Current() (depends on
    // the pass count): copy sorted keys back into `keys`' storage if they
    // ended up in the alternate, and point `perm` at its current buffer.
    if (d_keys.Current() != keys.data()) {
      auto keys_nc = keys; // non-const handle aliasing the same storage
      Kokkos::deep_copy(keys_nc, keys_out);
    }
    if (d_perm.Current() == perm_out.data()) {
      perm = perm_out;
    }
  }
#endif

#if defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
  // HIP analogue of the CUDA cub overload, using rocprim's radix sort
  // (which ships with rocThrust). Same bounded-bit, out-of-place,
  // transient-scratch scheme.
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t                  n_bins,
                                   ::sort::backend::Rocthrust) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    auto exec   = Kokkos::DefaultExecutionSpace();
    auto perm_v = perm;
    Kokkos::parallel_for(
      "PermInitIota",
      n,
      KOKKOS_LAMBDA(const npart_t i) { perm_v(i) = i; });

    array_t<ncells_t*>  keys_out("tile_keys_sorted", n);
    prtl_perm_t         perm_out("tile_perm_sorted", n);
    const unsigned int  end_bit = significant_bits(n_bins);

    exec.fence("sort_by_key_dispatch Rocthrust: pre-sort");
    auto stream = exec.hip_stream();

    // double_buffer radix sort: rocprim ping-pongs between the supplied
    // (current, alternate) buffer pairs, so `temp_storage` holds only the
    // histograms (~MB) instead of an internal N-sized alternate (~8*N bytes,
    // the dominant `rocprim_radix_temp`). This nearly halves the sort's
    // transient memory vs the plain out-of-place form, which matters at high
    // npart where that N-sized temp can fail to allocate (device OOM at scale).
    rocprim::double_buffer<ncells_t> d_keys(keys.data(), keys_out.data());
    rocprim::double_buffer<npart_t>  d_perm(perm.data(), perm_out.data());

    std::size_t temp_bytes = 0;
    auto        err = rocprim::radix_sort_pairs(nullptr,
                                         temp_bytes,
                                         d_keys,
                                         d_perm,
                                         static_cast<std::size_t>(n),
                                         0u,
                                         end_bit,
                                         stream);
    raise::ErrorIf(err != hipSuccess,
                   "rocprim::radix_sort_pairs (size query) failed",
                   HERE);
    array_t<char*> temp("rocprim_radix_temp",
                        (temp_bytes == 0u) ? std::size_t { 1 } : temp_bytes);
    err = rocprim::radix_sort_pairs(temp.data(),
                                    temp_bytes,
                                    d_keys,
                                    d_perm,
                                    static_cast<std::size_t>(n),
                                    0u,
                                    end_bit,
                                    stream);
    raise::ErrorIf(err != hipSuccess,
                   "rocprim::radix_sort_pairs failed",
                   HERE);
    exec.fence("sort_by_key_dispatch Rocthrust: post-sort");

    // Publish results from whichever buffer rocprim left as `current()`
    // (depends on the pass count). If the sorted keys ended up in the
    // alternate, copy them back into `keys`' storage so downstream
    // (compute_tile_offsets) sees them; point `perm` at its current buffer.
    if (d_keys.current() != keys.data()) {
      auto keys_nc = keys; // non-const handle aliasing the same storage
      Kokkos::deep_copy(keys_nc, keys_out);
    }
    if (d_perm.current() == perm_out.data()) {
      perm = perm_out;
    }
  }
#endif

  // Host fallback: indirect-sort via std::stable_sort.
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t /*n_bins*/,
                                   ::sort::backend::StdSort) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    auto keys_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
                                                       keys);
    auto perm_h = Kokkos::create_mirror_view(perm);
    std::iota(perm_h.data(), perm_h.data() + n, npart_t { 0u });
    std::stable_sort(perm_h.data(),
                     perm_h.data() + n,
                     [&](npart_t a, npart_t b) {
                       return keys_h(a) < keys_h(b);
                     });
    Kokkos::deep_copy(perm, perm_h);
  }

} // namespace ntt::sort_helpers

#endif // GLOBAL_UTILS_SORT_DISPATCH_H

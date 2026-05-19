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
 *       overloads (OneDPL on SYCL, Thrust on CUDA) are conditional on the
 *       respective build flags.
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

#if defined(SYCL_ENABLED) && defined(ONEDPL_ENABLED)
  #include <oneapi/dpl/algorithm>
  #include <oneapi/dpl/execution>
#endif
#if defined(CUDA_ENABLED) && defined(THRUST_ENABLED)
  #include <thrust/device_ptr.h>
  #include <thrust/sequence.h>
  #include <thrust/sort.h>
#endif
#if defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
  #include <thrust/device_ptr.h>
  #include <thrust/execution_policy.h>
  #include <thrust/sequence.h>
  #include <thrust/sort.h>
#endif

#include <algorithm>
#include <numeric>

namespace ntt::sort_helpers {

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
                                   ncells_t /*n_bins*/,
                                   ::sort::backend::Thrust) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    Kokkos::fence("sort_by_key_dispatch Thrust: pre-sort");
    thrust::device_ptr<ncells_t> kp(keys.data());
    thrust::device_ptr<npart_t>  pp(perm.data());
    thrust::sequence(pp, pp + n);
    thrust::sort_by_key(kp, kp + n, pp);
    Kokkos::fence("sort_by_key_dispatch Thrust: post-sort");
  }
#endif

#if defined(HIP_ENABLED) && defined(ROCTHRUST_ENABLED)
  // rocThrust exposes the same thrust:: API as CUDA Thrust; with hipcc
  // device_ptr-based algorithms dispatch to the HIP backend. Mirrors
  // the CUDA Thrust overload.
  inline void sort_by_key_dispatch(const array_t<ncells_t*>& keys,
                                   prtl_perm_t&              perm,
                                   ncells_t /*n_bins*/,
                                   ::sort::backend::Rocthrust) {
    const auto n = static_cast<npart_t>(keys.extent(0));
    if (n == 0u) {
      return;
    }
    Kokkos::fence("sort_by_key_dispatch Rocthrust: pre-sort");
    thrust::device_ptr<ncells_t> kp(keys.data());
    thrust::device_ptr<npart_t>  pp(perm.data());
    thrust::sequence(pp, pp + n);
    thrust::sort_by_key(kp, kp + n, pp);
    Kokkos::fence("sort_by_key_dispatch Rocthrust: post-sort");
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

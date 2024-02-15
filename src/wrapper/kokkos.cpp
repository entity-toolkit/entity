#include "wrapper.h"

#include "definitions.h"

#include <Kokkos_Core.hpp>

namespace ntt {
  auto WaitAndSynchronize(bool debug_only) -> void {
    if (debug_only) {
#ifndef DEBUG
      return;
#endif
    }
    Kokkos::fence();
  }

  template <>
  auto CreateRangePolicy<Dim1>(const tuple_t<std::size_t, Dim1>& i1,
                               const tuple_t<std::size_t, Dim1>& i2)
    -> range_t<Dim1> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    return Kokkos::RangePolicy<AccelExeSpace>(i1min, i1max);
  }

  template <>
  auto CreateRangePolicy<Dim2>(const tuple_t<std::size_t, Dim2>& i1,
                               const tuple_t<std::size_t, Dim2>& i2)
    -> range_t<Dim2> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, AccelExeSpace>({ i1min, i2min },
                                                                 { i1max, i2max });
  }

  template <>
  auto CreateRangePolicy<Dim3>(const tuple_t<std::size_t, Dim3>& i1,
                               const tuple_t<std::size_t, Dim3>& i2)
    -> range_t<Dim3> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    index_t i3min = i1[2];
    index_t i3max = i2[2];
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, AccelExeSpace>(
      { i1min, i2min, i3min },
      { i1max, i2max, i3max });
  }

  template <>
  auto CreateRangePolicyOnHost<Dim1>(const tuple_t<std::size_t, Dim1>& i1,
                                     const tuple_t<std::size_t, Dim1>& i2)
    -> range_h_t<Dim1> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    return Kokkos::RangePolicy<HostExeSpace>(i1min, i1max);
  }

  template <>
  auto CreateRangePolicyOnHost<Dim2>(const tuple_t<std::size_t, Dim2>& i1,
                                     const tuple_t<std::size_t, Dim2>& i2)
    -> range_h_t<Dim2> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    return Kokkos::MDRangePolicy<Kokkos::Rank<2>, HostExeSpace>({ i1min, i2min },
                                                                { i1max, i2max });
  }

  template <>
  auto CreateRangePolicyOnHost<Dim3>(const tuple_t<std::size_t, Dim3>& i1,
                                     const tuple_t<std::size_t, Dim3>& i2)
    -> range_h_t<Dim3> {
    index_t i1min = i1[0];
    index_t i1max = i2[0];
    index_t i2min = i1[1];
    index_t i2max = i2[1];
    index_t i3min = i1[2];
    index_t i3max = i2[2];
    return Kokkos::MDRangePolicy<Kokkos::Rank<3>, HostExeSpace>(
      { i1min, i2min, i3min },
      { i1max, i2max, i3max });
  }

} // namespace ntt
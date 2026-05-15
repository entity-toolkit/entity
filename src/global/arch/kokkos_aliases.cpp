#include "arch/kokkos_aliases.h"

#include "global.h"

#include <Kokkos_Core.hpp>

template <>
auto CreateParticleRangePolicy<Dim::_1D>(const tuple_t<npart_t, Dim::_1D>& p1,
                                         const tuple_t<npart_t, Dim::_1D>& p2)
  -> range_t<Dim::_1D> {
  prtlidx_t p1min = p1[0];
  prtlidx_t p1max = p2[0];
  return Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(p1min, p1max);
}

template <>
auto CreateParticleRangePolicy<Dim::_2D>(const tuple_t<npart_t, Dim::_2D>& p1,
                                         const tuple_t<npart_t, Dim::_2D>& p2)
  -> range_t<Dim::_2D> {
  prtlidx_t p1min = p1[0];
  prtlidx_t p1max = p2[0];
  prtlidx_t p2min = p1[1];
  prtlidx_t p2max = p2[1];
  return Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
    { p1min, p2min },
    { p1max, p2max });
}

template <>
auto CreateRangePolicy<Dim::_1D>(const tuple_t<ncells_t, Dim::_1D>& i1,
                                 const tuple_t<ncells_t, Dim::_1D>& i2)
  -> range_t<Dim::_1D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  return Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(i1min, i1max);
}

template <>
auto CreateRangePolicy<Dim::_2D>(const tuple_t<ncells_t, Dim::_2D>& i1,
                                 const tuple_t<ncells_t, Dim::_2D>& i2)
  -> range_t<Dim::_2D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  cellidx_t i2min = i1[1];
  cellidx_t i2max = i2[1];
  return Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultExecutionSpace>(
    { i1min, i2min },
    { i1max, i2max });
}

template <>
auto CreateRangePolicy<Dim::_3D>(const tuple_t<ncells_t, Dim::_3D>& i1,
                                 const tuple_t<ncells_t, Dim::_3D>& i2)
  -> range_t<Dim::_3D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  cellidx_t i2min = i1[1];
  cellidx_t i2max = i2[1];
  cellidx_t i3min = i1[2];
  cellidx_t i3max = i2[2];
  return Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultExecutionSpace>(
    { i1min, i2min, i3min },
    { i1max, i2max, i3max });
}

template <>
auto CreateRangePolicyOnHost<Dim::_1D>(const tuple_t<ncells_t, Dim::_1D>& i1,
                                       const tuple_t<ncells_t, Dim::_1D>& i2)
  -> range_h_t<Dim::_1D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  return Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(i1min, i1max);
}

template <>
auto CreateRangePolicyOnHost<Dim::_2D>(const tuple_t<ncells_t, Dim::_2D>& i1,
                                       const tuple_t<ncells_t, Dim::_2D>& i2)
  -> range_h_t<Dim::_2D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  cellidx_t i2min = i1[1];
  cellidx_t i2max = i2[1];
  return Kokkos::MDRangePolicy<Kokkos::Rank<2>, Kokkos::DefaultHostExecutionSpace>(
    { i1min, i2min },
    { i1max, i2max });
}

template <>
auto CreateRangePolicyOnHost<Dim::_3D>(const tuple_t<ncells_t, Dim::_3D>& i1,
                                       const tuple_t<ncells_t, Dim::_3D>& i2)
  -> range_h_t<Dim::_3D> {
  cellidx_t i1min = i1[0];
  cellidx_t i1max = i2[0];
  cellidx_t i2min = i1[1];
  cellidx_t i2max = i2[1];
  cellidx_t i3min = i1[2];
  cellidx_t i3max = i2[2];
  return Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::DefaultHostExecutionSpace>(
    { i1min, i2min, i3min },
    { i1max, i2max, i3max });
}

auto WaitAndSynchronize(bool debug_only) -> void {
  if (debug_only) {
#ifndef DEBUG
    return;
#endif
  }
  Kokkos::fence();
}

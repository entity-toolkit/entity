/**
 * @file output/utils/readers.h
 * @brief
 * Defines generic reader functions.
 * @implements
 *   - out::ReadVariable<> -> void
 *   - out::ReadVariableAll<> -> void
 *   - out::Read1DArray<> -> void
 *   - out::Read2DArray<> -> void
 *   - out::ReadNDField<> -> void
 * @cpp:
 *   - readers.cpp
 * @namespaces:
 *   - out::
 */

#ifndef OUTPUT_UTILS_READERS_H
#define OUTPUT_UTILS_READERS_H

#include "arch/kokkos_aliases.h"

#include <adios2.h>

#include <string>
#include <vector>

namespace out {

  template <typename T>
  void ReadVariable(adios2::IO&, adios2::Engine&, const std::string&, T&, std::size_t);

  /**
   * @brief Read the first `count` elements of a per-domain variable at once.
   * @note One selection instead of `count` single-element reads: each read
   *       lands in a different writer's block, so per-element reads cost a
   *       separate filesystem round-trip apiece.
   */
  template <typename T>
  void ReadVariableAll(adios2::IO&,
                       adios2::Engine&,
                       const std::string&,
                       std::vector<T>&,
                       std::size_t);

  template <typename T>
  void Read1DArray(adios2::IO&,
                   adios2::Engine&,
                   const std::string&,
                   array_t<T*>&,
                   std::size_t,
                   std::size_t);

  template <typename T>
  void Read2DArray(adios2::IO&,
                   adios2::Engine&,
                   const std::string&,
                   array_t<T**>&,
                   unsigned short,
                   std::size_t,
                   std::size_t);

  template <Dimension D, int N>
  void ReadNDField(adios2::IO&,
                   adios2::Engine&,
                   const std::string&,
                   ndfield_t<D, N>&,
                   const adios2::Box<adios2::Dims>&);

} // namespace out

#endif // OUTPUT_UTILS_READERS_H

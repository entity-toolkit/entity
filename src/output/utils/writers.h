/**
 * @file output/utils/writers.h
 * @brief
 * Defines generic writer functions.
 * @implements
 *  - out::WriteVariable<> -> void
 *  - out::Write1DArray<> -> void
 *  - out::Write2DArray<> -> void
 *  - out::WriteNDField<> -> void
 * @cpp:
 *   - writers.cpp
 * @namespaces:
 *   - out::
 */

#ifndef OUTPUT_UTILS_WRITERS_H
#define OUTPUT_UTILS_WRITERS_H

#include "arch/kokkos_aliases.h"

#include <adios2.h>

#include <string>

namespace out {

  template <typename T>
  void WriteVariable(adios2::IO&,
                     adios2::Engine&,
                     const std::string&,
                     const T&,
                     std::size_t,
                     std::size_t);

  template <typename T>
  void Write1DArray(adios2::IO&,
                    adios2::Engine&,
                    const std::string&,
                    const array_t<T*>&,
                    std::size_t,
                    std::size_t,
                    std::size_t);

  template <typename T>
  void Write2DArray(adios2::IO&,
                    adios2::Engine&,
                    const std::string&,
                    const array_t<T**>&,
                    unsigned short,
                    std::size_t,
                    std::size_t,
                    std::size_t);

  template <Dimension D, int N>
  void WriteNDField(adios2::IO&,
                    adios2::Engine&,
                    const std::string&,
                    const ndfield_t<D, N>&);

} // namespace out

#endif // OUTPUT_UTILS_WRITERS_H

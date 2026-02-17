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

  // template <typename T>
  // void Write1DSubArray(adios2::IO&,
  //                      adios2::Engine&,
  //                      const std::string&,
  //                      const subarray1d_t<T>&,
  //                      std::size_t,
  //                      std::size_t,
  //                      std::size_t);

  template <typename T, typename S>
  void Write1DSubArray(adios2::IO&        io,
                       adios2::Engine&    writer,
                       const std::string& name,
                       const S&           data,
                       std::size_t        local_size,
                       std::size_t        global_size,
                       std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(name);
    var.SetShape({ global_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { local_size }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    if (!data_h.span_is_contiguous()) {
      array_h_t<T*> data_contig_h { "data_contig_h", local_size };
      Kokkos::deep_copy(data_contig_h, data_h);
      writer.Put(var, data_contig_h.data(), adios2::Mode::Sync);
    } else {
      writer.Put(var, data_h.data(), adios2::Mode::Sync);
    }
  }

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

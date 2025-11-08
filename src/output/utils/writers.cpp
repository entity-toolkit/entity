#include "output/utils/writers.h"

#include "arch/kokkos_aliases.h"

#include <adios2.h>

#include <string>

namespace out {

  template <typename T>
  void WriteVariable(adios2::IO&        io,
                     adios2::Engine&    writer,
                     const std::string& name,
                     const T&           data,
                     std::size_t        global_size,
                     std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(name);
    var.SetShape({ global_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { 1 }));
    writer.Put(var, &data);
  }

  template <typename T>
  void Write1DArray(adios2::IO&        io,
                    adios2::Engine&    writer,
                    const std::string& name,
                    const array_t<T*>& data,
                    std::size_t        local_size,
                    std::size_t        global_size,
                    std::size_t        local_offset) {
    const auto slice = range_tuple_t(0, local_size);
    auto       var   = io.InquireVariable<T>(name);
    var.SetShape({ global_size });
    var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { local_size }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice);
    writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

  template <typename T>
  void Write2DArray(adios2::IO&         io,
                    adios2::Engine&     writer,
                    const std::string&  name,
                    const array_t<T**>& data,
                    unsigned short      dim2_size,
                    std::size_t         local_size,
                    std::size_t         global_size,
                    std::size_t         local_offset) {
    const auto slice = range_tuple_t(0, local_size);
    auto       var   = io.InquireVariable<T>(name);

    var.SetShape({ global_size, dim2_size });
    var.SetSelection(
      adios2::Box<adios2::Dims>({ local_offset, 0 }, { local_size, dim2_size }));

    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    auto data_sub = Kokkos::subview(data_h, slice, range_tuple_t(0, dim2_size));
    writer.Put(var, data_sub.data(), adios2::Mode::Sync);
  }

  template <Dimension D, int N>
  void WriteNDField(adios2::IO&            io,
                    adios2::Engine&        writer,
                    const std::string&     name,
                    const ndfield_t<D, N>& data) {
    auto data_h = Kokkos::create_mirror_view(data);
    Kokkos::deep_copy(data_h, data);
    writer.Put(io.InquireVariable<real_t>(name), data_h.data(), adios2::Mode::Sync);
  }

#define ARRAY_WRITERS(T)                                                       \
  template void WriteVariable(adios2::IO&,                                     \
                              adios2::Engine&,                                 \
                              const std::string&,                              \
                              const T&,                                        \
                              std::size_t,                                     \
                              std::size_t);                                    \
  template void Write1DArray<T>(adios2::IO&,                                   \
                                adios2::Engine&,                               \
                                const std::string&,                            \
                                const array_t<T*>&,                            \
                                std::size_t,                                   \
                                std::size_t,                                   \
                                std::size_t);                                  \
  template void Write2DArray<T>(adios2::IO&,                                   \
                                adios2::Engine&,                               \
                                const std::string&,                            \
                                const array_t<T**>&,                           \
                                unsigned short,                                \
                                std::size_t,                                   \
                                std::size_t,                                   \
                                std::size_t);

  ARRAY_WRITERS(short)
  ARRAY_WRITERS(unsigned short)
  ARRAY_WRITERS(int)
  ARRAY_WRITERS(unsigned int)
  ARRAY_WRITERS(unsigned long int)
  ARRAY_WRITERS(double)
  ARRAY_WRITERS(float)
#undef ARRAY_WRITERS

#define NDFIELD_WRITERS(D, N)                                                  \
  template void WriteNDField<D, N>(adios2::IO&,                                \
                                   adios2::Engine&,                            \
                                   const std::string&,                         \
                                   const ndfield_t<D, N>&);
  NDFIELD_WRITERS(Dim::_1D, 3)
  NDFIELD_WRITERS(Dim::_1D, 6)
  NDFIELD_WRITERS(Dim::_2D, 3)
  NDFIELD_WRITERS(Dim::_2D, 6)
  NDFIELD_WRITERS(Dim::_3D, 3)
  NDFIELD_WRITERS(Dim::_3D, 6)
#undef NDFIELD_WRITERS

} // namespace out

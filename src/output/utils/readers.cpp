#include "output/utils/readers.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include <adios2.h>

#include <string>

namespace out {

  template <typename T>
  void ReadVariable(adios2::IO&        io,
                    adios2::Engine&    reader,
                    const std::string& quantity,
                    T&                 data,
                    std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(quantity);
    if (var) {
      T read_data;
      var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { 1 }));
      reader.Get(var, &read_data, adios2::Mode::Sync);
      data = read_data;
    } else {
      raise::Error(fmt::format("Variable: %s not found", quantity.c_str()), HERE);
    }
  }

  template <typename T>
  void Read1DArray(adios2::IO&        io,
                   adios2::Engine&    reader,
                   const std::string& quantity,
                   array_t<T*>&       data,
                   std::size_t        local_size,
                   std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(quantity);
    if (var) {
      var.SetSelection(adios2::Box<adios2::Dims>({ local_offset }, { local_size }));
      const auto slice  = range_tuple_t(0, local_size);
      auto       data_h = Kokkos::create_mirror_view(data);
      reader.Get(var, Kokkos::subview(data_h, slice).data(), adios2::Mode::Sync);
      Kokkos::deep_copy(Kokkos::subview(data, slice),
                        Kokkos::subview(data_h, slice));
    } else {
      raise::Error(fmt::format("Variable: %s not found", quantity.c_str()), HERE);
    }
  }

  template <typename T>
  void Read2DArray(adios2::IO&        io,
                   adios2::Engine&    reader,
                   const std::string& quantity,
                   array_t<T**>&      data,
                   unsigned short     dim2_size,
                   std::size_t        local_size,
                   std::size_t        local_offset) {
    auto var = io.InquireVariable<T>(quantity);
    if (var) {
      var.SetSelection(adios2::Box<adios2::Dims>({ local_offset, 0 },
                                                 { local_size, dim2_size }));
      const auto slice  = range_tuple_t(0, local_size);
      auto       data_h = Kokkos::create_mirror_view(data);
      reader.Get(var,
                 Kokkos::subview(data_h, slice, range_tuple_t(0, dim2_size)).data(),
                 adios2::Mode::Sync);
      Kokkos::deep_copy(data, data_h);
    } else {
      raise::Error(fmt::format("Variable: %s not found", quantity.c_str()), HERE);
    }
  }

  template <Dimension D, int N>
  void ReadNDField(adios2::IO&                      io,
                   adios2::Engine&                  reader,
                   const std::string&               quantity,
                   ndfield_t<D, N>&                 data,
                   const adios2::Box<adios2::Dims>& range) {
    auto var = io.InquireVariable<real_t>(quantity);
    if (var) {
      var.SetSelection(range);

      auto data_h = Kokkos::create_mirror_view(data);
      reader.Get(var, data_h.data(), adios2::Mode::Sync);
      Kokkos::deep_copy(data, data_h);
    } else {
      raise::Error(fmt::format("Variable: %s not found", quantity.c_str()), HERE);
    }
  }

#define ARRAY_READERS(T)                                                       \
  template void ReadVariable(adios2::IO&,                                      \
                             adios2::Engine&,                                  \
                             const std::string&,                               \
                             T&,                                               \
                             std::size_t);                                     \
  template void Read1DArray<T>(adios2::IO&,                                    \
                               adios2::Engine&,                                \
                               const std::string&,                             \
                               array_t<T*>&,                                   \
                               std::size_t,                                    \
                               std::size_t);                                   \
  template void Read2DArray<T>(adios2::IO&,                                    \
                               adios2::Engine&,                                \
                               const std::string&,                             \
                               array_t<T**>&,                                  \
                               unsigned short,                                 \
                               std::size_t,                                    \
                               std::size_t);

  ARRAY_READERS(short)
  ARRAY_READERS(unsigned short)
  ARRAY_READERS(int)
  ARRAY_READERS(unsigned int)
  ARRAY_READERS(unsigned long int)
  ARRAY_READERS(double)
  ARRAY_READERS(float)
#undef ARRAY_READERS

#define NDFIELD_READERS(D, N)                                                  \
  template void ReadNDField<D, N>(adios2::IO&,                                 \
                                  adios2::Engine&,                             \
                                  const std::string&,                          \
                                  ndfield_t<D, N>&,                            \
                                  const adios2::Box<adios2::Dims>&);
  NDFIELD_READERS(Dim::_1D, 3)
  NDFIELD_READERS(Dim::_1D, 6)
  NDFIELD_READERS(Dim::_2D, 3)
  NDFIELD_READERS(Dim::_2D, 6)
  NDFIELD_READERS(Dim::_3D, 3)
  NDFIELD_READERS(Dim::_3D, 6)
#undef NDFIELD_READERS

} // namespace out

#include "checkpoint/reader.h"

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"

#include <Kokkos_Core.hpp>
#include <adios2.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <string>
#include <utility>
#include <vector>

namespace checkpoint {

  template <Dimension D, int N>
  void ReadFields(adios2::IO&                      io,
                  adios2::Engine&                  reader,
                  const std::string&               field,
                  const adios2::Box<adios2::Dims>& range,
                  ndfield_t<D, N>&                 array) {
    logger::Checkpoint(fmt::format("Reading field: %s", field.c_str()), HERE);
    auto field_var = io.InquireVariable<real_t>(field);
    if (field_var) {
      field_var.SetSelection(range);

      auto array_h = Kokkos::create_mirror_view(array);
      reader.Get(field_var, array_h.data(), adios2::Mode::Sync);
      Kokkos::deep_copy(array, array_h);
    } else {
      raise::Error(fmt::format("Field variable: %s not found", field.c_str()),
                   HERE);
    }
  }

  auto ReadParticleCount(adios2::IO&     io,
                         adios2::Engine& reader,
                         spidx_t         s,
                         std::size_t     local_dom,
                         std::size_t ndomains) -> std::pair<npart_t, npart_t> {
    logger::Checkpoint(fmt::format("Reading particle count for: %d", s + 1), HERE);
    auto npart_var = io.InquireVariable<npart_t>(fmt::format("s%d_npart", s + 1));
    if (npart_var) {
      raise::ErrorIf(npart_var.Shape()[0] != ndomains,
                     "npart_var.Shape()[0] != ndomains",
                     HERE);
      raise::ErrorIf(npart_var.Shape().size() != 1,
                     "npart_var.Shape().size() != 1",
                     HERE);
      npart_var.SetSelection(adios2::Box<adios2::Dims>({ local_dom }, { 1 }));
      npart_t npart;
      reader.Get(npart_var, &npart, adios2::Mode::Sync);
      const auto loc_npart = npart;
#if !defined(MPI_ENABLED)
      npart_t offset_npart = 0;
#else
      std::vector<npart_t> glob_nparts(ndomains);
      MPI_Allgather(&loc_npart,
                    1,
                    mpi::get_type<npart_t>(),
                    glob_nparts.data(),
                    1,
                    mpi::get_type<npart_t>(),
                    MPI_COMM_WORLD);
      npart_t offset_npart = 0;
      for (auto d { 0u }; d < local_dom; ++d) {
        offset_npart += glob_nparts[d];
      }
#endif
      return { loc_npart, offset_npart };
    } else {
      raise::Error("npart_var is not found", HERE);
      return { 0, 0 };
    }
  }

  template <typename T>
  void ReadParticleData(adios2::IO&        io,
                        adios2::Engine&    reader,
                        const std::string& quantity,
                        spidx_t            s,
                        array_t<T*>&       array,
                        npart_t            count,
                        npart_t            offset) {
    logger::Checkpoint(
      fmt::format("Reading quantity: s%d_%s", s + 1, quantity.c_str()),
      HERE);
    auto var = io.InquireVariable<T>(
      fmt::format("s%d_%s", s + 1, quantity.c_str()));
    if (var) {
      var.SetSelection(adios2::Box<adios2::Dims>({ offset }, { count }));
      const auto slice   = range_tuple_t(0, count);
      auto       array_h = Kokkos::create_mirror_view(array);
      reader.Get(var, Kokkos::subview(array_h, slice).data(), adios2::Mode::Sync);
      Kokkos::deep_copy(Kokkos::subview(array, slice),
                        Kokkos::subview(array_h, slice));
    } else {
      raise::Error(
        fmt::format("Variable: s%d_%s not found", s + 1, quantity.c_str()),
        HERE);
    }
  }

  void ReadParticlePayloads(adios2::IO&        io,
                            adios2::Engine&    reader,
                            spidx_t            s,
                            array_t<real_t**>& array,
                            std::size_t        nplds,
                            npart_t            count,
                            npart_t            offset) {
    logger::Checkpoint(fmt::format("Reading quantity: s%d_plds", s + 1), HERE);
    auto var = io.InquireVariable<real_t>(fmt::format("s%d_plds", s + 1));
    if (var) {
      var.SetSelection(adios2::Box<adios2::Dims>({ offset, 0 }, { count, nplds }));
      const auto slice   = range_tuple_t(0, count);
      auto       array_h = Kokkos::create_mirror_view(array);
      reader.Get(var,
                 Kokkos::subview(array_h, slice, range_tuple_t(0, nplds)).data(),
                 adios2::Mode::Sync);
      Kokkos::deep_copy(array, array_h);
    } else {
      raise::Error(fmt::format("Variable: s%d_plds not found", s + 1), HERE);
    }
  }

#define CHECKPOINT_FIELDS(D, N)                                                \
  template void ReadFields<D, N>(adios2::IO&,                                  \
                                 adios2::Engine&,                              \
                                 const std::string&,                           \
                                 const adios2::Box<adios2::Dims>&,             \
                                 ndfield_t<D, N>&);
  CHECKPOINT_FIELDS(Dim::_1D, 3)
  CHECKPOINT_FIELDS(Dim::_2D, 3)
  CHECKPOINT_FIELDS(Dim::_3D, 3)
  CHECKPOINT_FIELDS(Dim::_1D, 6)
  CHECKPOINT_FIELDS(Dim::_2D, 6)
  CHECKPOINT_FIELDS(Dim::_3D, 6)
#undef CHECKPOINT_FIELDS

#define CHECKPOINT_PARTICLE_DATA(T)                                            \
  template void ReadParticleData<T>(adios2::IO&,                               \
                                    adios2::Engine&,                           \
                                    const std::string&,                        \
                                    spidx_t,                                   \
                                    array_t<T*>&,                              \
                                    npart_t,                                   \
                                    npart_t);
  CHECKPOINT_PARTICLE_DATA(int)
  CHECKPOINT_PARTICLE_DATA(float)
  CHECKPOINT_PARTICLE_DATA(double)
  CHECKPOINT_PARTICLE_DATA(short)
#undef CHECKPOINT_PARTICLE_DATA

} // namespace checkpoint

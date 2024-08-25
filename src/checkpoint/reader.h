/**
 * @file checkpoint/reader.h
 * @brief Function for reading field & particle data from checkpoint files
 * @implements
 *   - checkpoint::ReadFields -> void
 *   - checkpoint::ReadParticleData -> void
 * @cpp:
 *   - reader.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_READER_H
#define CHECKPOINT_READER_H

#include "arch/kokkos_aliases.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <string>
#include <utility>

namespace checkpoint {

  template <Dimension D, int N>
  void ReadFields(adios2::IO&,
                  adios2::Engine&,
                  const std::string&,
                  const adios2::Box<adios2::Dims>&,
                  ndfield_t<D, N>&);

  auto ReadParticleCount(adios2::IO&,
                         adios2::Engine&,
                         unsigned short,
                         std::size_t,
                         std::size_t) -> std::pair<std::size_t, std::size_t>;

  template <typename T>
  void ReadParticleData(adios2::IO&,
                        adios2::Engine&,
                        const std::string&,
                        unsigned short,
                        array_t<T*>&,
                        std::size_t,
                        std::size_t);

} // namespace checkpoint

#endif // CHECKPOINT_READER_H

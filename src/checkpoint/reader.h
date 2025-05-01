/**
 * @file checkpoint/reader.h
 * @brief Function for reading field & particle data from checkpoint files
 * @implements
 *   - checkpoint::ReadFields -> void
 *   - checkpoint::ReadParticleData -> void
 *   - checkpoint::ReadParticleCount -> std::pair<npart_t, npart_t>
 * @cpp:
 *   - reader.cpp
 * @namespaces:
 *   - checkpoint::
 */

#ifndef CHECKPOINT_READER_H
#define CHECKPOINT_READER_H

#include "arch/kokkos_aliases.h"

#include <adios2.h>

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
                         spidx_t,
                         std::size_t,
                         std::size_t) -> std::pair<npart_t, npart_t>;

  template <typename T>
  void ReadParticleData(adios2::IO&,
                        adios2::Engine&,
                        const std::string&,
                        spidx_t,
                        array_t<T*>&,
                        npart_t,
                        npart_t);

  void ReadParticlePayloads(adios2::IO&,
                            adios2::Engine&,
                            spidx_t,
                            array_t<real_t**>&,
                            std::size_t,
                            npart_t,
                            npart_t);

} // namespace checkpoint

#endif // CHECKPOINT_READER_H

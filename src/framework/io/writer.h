#ifndef IO_WRITER_H
#define IO_WRITER_H

#include "wrapper.h"

#include "meshblock.h"
#include "output.h"
#include "sim_params.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <map>
#include <string>

namespace ntt {

  template <Dimension D, SimulationEngine S>
  class Writer {
#ifdef OUTPUT_ENABLED
    adios2::ADIOS                m_adios;
    adios2::IO                   m_io;
    adios2::Engine               m_writer;
    adios2::Mode                 m_mode { adios2::Mode::Write };

    std::vector<OutputField>     m_fields;
    std::vector<OutputParticles> m_particles;
#endif

  public:
    Writer() = default;
    ~Writer();

    void Initialize(const SimulationParams&, const Meshblock<D, S>&);
    void WriteAll(const SimulationParams&, Meshblock<D, S>&, const real_t&, const std::size_t&);

    void WriteFields(const SimulationParams&,
                     Meshblock<D, S>&,
                     const real_t&,
                     const std::size_t&);

    void WriteParticles(const SimulationParams&,
                        Meshblock<D, S>&,
                        const real_t&,
                        const std::size_t&);
  };

}    // namespace ntt

#endif    // IO_WRITER_H
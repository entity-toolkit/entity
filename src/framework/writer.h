#ifndef FRAMEWORK_OUTPUT_WRITER_H
#define FRAMEWORK_OUTPUT_WRITER_H

#include "wrapper.h"
#include "meshblock.h"
#include "sim_params.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#include <map>
#include <string>

namespace ntt {

  template <Dimension D, SimulationType S>
  class Writer {
    adios2::ADIOS  m_adios;
    adios2::IO     m_io;
    adios2::Engine writer;

    std::map<std::string, adios2::Variable<real_t>> m_vars_r;
    std::map<std::string, adios2::Variable<int>>    m_vars_i;

  public:
    Writer(const SimulationParams&, const Meshblock<D, S>&);
    ~Writer();

    void WriteFields(const Meshblock<D, S>&, const real_t&, const std::size_t&);
  };

} // namespace ntt

#endif // FRAMEWORK_OUTPUT_WRITER_H
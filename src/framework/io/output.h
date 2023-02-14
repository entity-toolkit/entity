#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace ntt {
  class OutputField {
    std::string m_name;
    FieldID     m_id;

  public:
    std::vector<int> comp {};
    std::vector<int> species {};

    OutputField() = default;
    OutputField(const std::string& name, const FieldID& id) : m_name(name), m_id(id) {}
    ~OutputField() = default;

    void setName(const std::string& name) {
      m_name = name;
    }
    void setId(const FieldID& id) {
      m_id = id;
    }

    [[nodiscard]] auto name() const -> std::string {
      return m_name;
    }
    [[nodiscard]] auto id() const -> FieldID {
      return m_id;
    }

    void show(std::ostream& os = std::cout) const;

#ifdef OUTPUT_ENABLED
    template <Dimension D, SimulationEngine S>
    void put(adios2::Engine&,
             const adios2::Variable<real_t>&,
             const SimulationParams&,
             Meshblock<D, S>&) const;
#endif
  };

  auto InterpretInputField(const std::string&) -> std::vector<OutputField>;
}    // namespace ntt

#endif    // IO_OUTPUT_H
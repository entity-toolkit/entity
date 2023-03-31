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
  inline auto StringizeFieldID(const FieldID& id) -> std::string {
    switch (id) {
    case FieldID::E:
      return "E";
    case FieldID::D:
      return "D";
    case FieldID::B:
      return "B";
    case FieldID::H:
      return "H";
    case FieldID::J:
      return "J";
    case FieldID::T:
      return "T";
    case FieldID::Rho:
      return "Rho";
    case FieldID::N:
      return "N";
    case FieldID::Nppc:
      return "Nppc";
    default:
      return "UNKNOWN";
    }
  }

  class OutputField {
    std::string m_name;
    FieldID     m_id;

  public:
    std::vector<std::vector<int>> comp {};
    std::vector<int>              species {};

    OutputField() = default;
    OutputField(const std::string& name, const FieldID& id) : m_name(name), m_id(id) {}
    ~OutputField() = default;

    void setName(const std::string& name) {
      m_name = name;
    }
    void setId(const FieldID& id) {
      m_id = id;
    }

    [[nodiscard]] auto name(const int& i) const -> std::string {
      std::string myname { m_name };
      for (auto& cc : comp[i]) {
#ifdef MINKOWSKI_METRIC
        myname += (cc == 0 ? "t" : (cc == 1 ? "x" : (cc == 2 ? "y" : "z")));
#else
        myname += std::to_string(cc);
#endif
      }
      if (species.size() > 0) {
        myname += "_";
        for (auto& s : species) {
          myname += std::to_string(s);
          myname += "_";
        }
        myname.pop_back();
      }
      return myname;
    }

    [[nodiscard]] auto id() const -> FieldID {
      return m_id;
    }

#ifdef OUTPUT_ENABLED
    template <Dimension D, SimulationEngine S>
    void put(adios2::IO&, adios2::Engine&, const SimulationParams&, Meshblock<D, S>&) const;
#endif
  };

  auto InterpretInputField(const std::string&) -> OutputField;
}    // namespace ntt

#endif    // IO_OUTPUT_H
#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include "wrapper.h"

#include <string>
#include <utility>

namespace ntt {
  enum class FieldID {
    E,      // Electric fields
    B,      // Magnetic fields
    J,      // Current density
    T,      // Particle distribution moments
    Rho,    // Particle density
    N       // Particle number density
  };

  inline auto StringizeFieldID(const FieldID& id) -> std::string {
    switch (id) {
    case FieldID::E:
      return "E";
    case FieldID::B:
      return "B";
    case FieldID::J:
      return "J";
    case FieldID::T:
      return "T";
    case FieldID::Rho:
      return "Rho";
    case FieldID::N:
      return "N";
    default:
      return "UNKNOWN";
    }
  }

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

    void show() const {
      std::cout << "OutputField: " << m_name << " (" << StringizeFieldID(m_id) << ")\n";
      if (comp.size() == 2) {
        std::cout << "  comp: " << comp[0] << " " << comp[1] << "\n";
      } else if (comp.size() == 1) {
        std::cout << "  comp: " << comp[0] << "\n";
      }
      std::cout << "  species: ";
      for (const auto& s : species) {
        std::cout << s << " ";
      }
      std::cout << "\n";
    }
  };

  auto InterpretInputField(const std::string&) -> std::vector<OutputField>;
}    // namespace ntt

#endif    // IO_OUTPUT_H
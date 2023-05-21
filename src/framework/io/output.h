#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include "wrapper.h"

#include "meshblock.h"
#include "sim_params.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

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
      NTTHostError("Unknown field ID");
      return "UNKNOWN";
    }
  }

  inline auto StringizePrtlID(const PrtlID& id) -> std::string {
    switch (id) {
    case PrtlID::X:
      return "X";
    case PrtlID::U:
      return "U";
    case PrtlID::W:
      return "W";
    default:
      NTTHostError("Unknown particle ID");
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

  class OutputParticles {
    std::string      m_name;
    std::vector<int> m_species_id;
    PrtlID           m_id;

  public:
    OutputParticles() = default;
    OutputParticles(const std::string&      name,
                    const std::vector<int>& species_id,
                    const PrtlID&           id)
      : m_name(name), m_species_id { species_id }, m_id(id) {}
    ~OutputParticles() = default;

    void setName(const std::string& name) {
      m_name = name;
    }
    void setSpeciesID(const std::vector<int>& species_id) {
      std::copy(species_id.begin(), species_id.end(), std::back_inserter(m_species_id));
    }
    void setId(const PrtlID& id) {
      m_id = id;
    }

    [[nodiscard]] auto name(const int& i) const -> std::string {
      return m_name + "p_" + std::to_string(i);
    }

    [[nodiscard]] auto speciesID() const -> std::vector<int> {
      return m_species_id;
    }

    [[nodiscard]] auto id() const -> PrtlID {
      return m_id;
    }

#ifdef OUTPUT_ENABLED
    template <Dimension D, SimulationEngine S>
    void put(adios2::IO&, adios2::Engine&, const SimulationParams&, Meshblock<D, S>&) const;
#endif
  };

  auto InterpretInputForFieldOutput(const std::string&) -> OutputField;
  auto InterpretInputForParticleOutput(const std::string&) -> OutputParticles;
}    // namespace ntt

#endif    // IO_OUTPUT_H
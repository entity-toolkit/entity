#ifndef IO_OUTPUT_H
#define IO_OUTPUT_H

#include "wrapper.h"

#include "sim_params.h"

#include "communications/metadomain.h"
#include "meshblock/meshblock.h"

#ifdef OUTPUT_ENABLED
  #include <adios2.h>
  #include <adios2/cxx11/KokkosView.h>
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
      case FieldID::A:
        return "A";
      case FieldID::Rho:
        return "Rho";
      case FieldID::Charge:
        return "Charge";
      case FieldID::divE:
        return "divE";
      case FieldID::divD:
        return "divD";
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
    const std::string m_name;
    const FieldID     m_id;

    PrepareOutputFlags prepare_flag { PrepareOutput_None };
    PrepareOutputFlags interp_flag { PrepareOutput_None };

  public:
    std::vector<std::vector<int>> comp {};
    std::vector<int>              species {};
    std::vector<int>              address {};
    bool                          ghosts { false };

    OutputField(const std::string& name, const FieldID& id) :
      m_name(name),
      m_id(id) {}

    ~OutputField() = default;

    [[nodiscard]]
    auto is_moment() const -> bool {
      return (id() == FieldID::T || id() == FieldID::Rho || id() == FieldID::Nppc ||
              id() == FieldID::N || id() == FieldID::Charge);
    }

    [[nodiscard]]
    auto is_field() const -> bool {
      return (id() == FieldID::E || id() == FieldID::B || id() == FieldID::D ||
              id() == FieldID::H);
    }

    [[nodiscard]]
    auto is_divergence() const -> bool {
      return (id() == FieldID::divE || id() == FieldID::divD);
    }

    [[nodiscard]]
    auto is_current() const -> bool {
      return (id() == FieldID::J);
    }

    [[nodiscard]]
    auto is_efield() const -> bool {
      return (id() == FieldID::E || id() == FieldID::D);
    }

    [[nodiscard]]
    auto is_gr_aux_field() const -> bool {
      return (id() == FieldID::E || id() == FieldID::H);
    }

    [[nodiscard]]
    auto is_vpotential() const -> bool {
      return (id() == FieldID::A);
    }

    [[nodiscard]]
    auto name(const int& i) const -> std::string;

    [[nodiscard]]
    auto id() const -> FieldID {
      return m_id;
    }

    void initialize(const SimulationEngine& S);

    template <Dimension D, SimulationEngine S>
    void compute(const SimulationParams& params, Meshblock<D, S>& mblock) const;

#ifdef OUTPUT_ENABLED
    template <Dimension D, SimulationEngine S>
    void put(adios2::IO&, adios2::Engine&, Meshblock<D, S>&) const;
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
                    const PrtlID&           id) :
      m_name(name),
      m_species_id { species_id },
      m_id(id) {}

    ~OutputParticles() = default;

    void setName(const std::string& name) {
      m_name = name;
    }

    void setSpeciesID(const std::vector<int>& species_id) {
      std::copy(species_id.begin(),
                species_id.end(),
                std::back_inserter(m_species_id));
    }

    void setId(const PrtlID& id) {
      m_id = id;
    }

    [[nodiscard]]
    auto name(const int& i) const -> std::string {
      return m_name + "p_" + std::to_string(i);
    }

    [[nodiscard]]
    auto speciesID() const -> std::vector<int> {
      return m_species_id;
    }

    [[nodiscard]]
    auto id() const -> PrtlID {
      return m_id;
    }

#ifdef OUTPUT_ENABLED
    template <Dimension D, SimulationEngine S>
    void put(adios2::IO&,
             adios2::Engine&,
             const SimulationParams&,
             const Metadomain<D>&,
             Meshblock<D, S>&) const;
#endif
  };

  auto InterpretInputForFieldOutput(const std::string&) -> OutputField;
  auto InterpretInputForParticleOutput(const std::string&) -> OutputParticles;
} // namespace ntt

#endif // IO_OUTPUT_H
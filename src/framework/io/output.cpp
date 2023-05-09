#include "output.h"

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <string>
#include <vector>

namespace ntt {
  namespace {
    auto InterpretInputForFieldOutput_helper(const FieldID&                       fid,
                                             const std::vector<std::vector<int>>& comps,
                                             const std::vector<int>& species) -> OutputField {
      OutputField of;
      of.setId(fid);
      for (auto ci : comps) {
        std::vector<int> component;
        for (auto c : ci) {
          component.push_back(c);
        }
        of.comp.push_back(component);
      }
      for (auto s : species) {
        of.species.push_back(s);
      }
      of.setName(StringizeFieldID(fid));
      return of;
    }

    auto InterpretInputForFieldOutput_getcomponents(const std::vector<std::string>& comps)
      -> std::vector<std::vector<int>> {
      NTTHostErrorIf(comps.size() > 2, "Invalid field name");
      std::vector<int> comps_int;
      for (auto& c : comps) {
        TestValidOption(c, { "t", "x", "y", "z", "0", "1", "2", "3", "i", "j" });
        if (c == "t") {
          comps_int.push_back(0);
        } else if (c == "x") {
          comps_int.push_back(1);
        } else if (c == "y") {
          comps_int.push_back(2);
        } else if (c == "z") {
          comps_int.push_back(3);
        } else if (c == "i") {
          comps_int.push_back(-1);
        } else if (c == "j") {
          comps_int.push_back(-2);
        } else {
          comps_int.push_back(std::stoi(c));
        }
      }
      std::vector<std::vector<int>> comps_ints;
      if (comps_int.size() == 1) {
        auto c = comps_int[0];
        if (c < 0) {
          for (int i { 0 }; i < 3; ++i) {
            comps_ints.push_back({ i + 1 });
          }
        } else {
          NTTHostErrorIf((c > 3) || (c == 0), "Invalid field name");
          comps_ints.push_back({ c });
        }
      } else {
        auto c1 = comps_int[0];
        auto c2 = comps_int[1];
        if (c1 < 0) {
          for (int i { 0 }; i < 3; ++i) {
            if (c2 < 0) {
              for (int j { i }; j < 3; ++j) {
                comps_ints.push_back({ i + 1, j + 1 });
              }
            } else {
              comps_ints.push_back({ i + 1, c2 });
            }
          }
        } else {
          if (c2 < 0) {
            for (int j { 0 }; j < 3; ++j) {
              comps_ints.push_back({ c1, j + 1 });
            }
          } else {
            comps_ints.push_back({ c1, c2 });
          }
        }
      }
      return comps_ints;
    }

    auto InterpretInput_getspecies(const std::string& input_quantities) -> std::vector<int> {
      std::vector<int> species;
      if (input_quantities.find("_") < input_quantities.size()) {
        auto species_str
          = SplitString(input_quantities.substr(input_quantities.find("_") + 1), "_");
        for (const auto& specie : species_str) {
          species.push_back(std::stoi(specie));
        }
      }
      return species;
    }
  }    // namespace

  auto InterpretInputForFieldOutput(const std::string& fld) -> OutputField {
    FieldID                       id;
    std::vector<std::vector<int>> comps   = { {} };
    std::vector<int>              species = {};
    if (fld.find("T") == 0) {
      id = FieldID::T;
    } else if (fld.find("Rho") == 0) {
      id = FieldID::Rho;
    } else if (fld.find("Nppc") == 0) {
      id = FieldID::Nppc;
    } else if (fld.find("N") == 0) {
      id = FieldID::N;
    } else if (fld.find("E") == 0) {
      id = FieldID::E;
    } else if (fld.find("B") == 0) {
      id = FieldID::B;
    } else if (fld.find("D") == 0) {
      id = FieldID::D;
    } else if (fld.find("H") == 0) {
      id = FieldID::H;
    } else if (fld.find("J") == 0) {
      id = FieldID::J;
    } else {
      NTTHostError("Invalid field name");
    }
    auto is_moment
      = (id == FieldID::T || id == FieldID::Rho || id == FieldID::Nppc || id == FieldID::N);
    auto is_field = (id == FieldID::E || id == FieldID::B || id == FieldID::D
                     || id == FieldID::H || id == FieldID::J);
    if (is_moment) {
      species = InterpretInput_getspecies(fld);
    } else if (is_field) {
      comps = InterpretInputForFieldOutput_getcomponents({ fld.substr(1, 1) });
    }
    if (id == FieldID::T) {
      comps
        = InterpretInputForFieldOutput_getcomponents({ fld.substr(1, 1), fld.substr(2, 1) });
    }
    return InterpretInputForFieldOutput_helper(id, comps, species);
  }

  auto InterpretInputForParticleOutput(const std::string& prtl) -> OutputParticles {
    PrtlID id;
    if (prtl.find("X") == 0) {
      id = PrtlID::X;
    } else if (prtl.find("U") == 0) {
      id = PrtlID::U;
    } else if (prtl.find("W") == 0) {
      id = PrtlID::W;
    } else {
      NTTHostError("Invalid particle quantity ");
    }
    return OutputParticles(StringizePrtlID(id), InterpretInput_getspecies(prtl), id);
  }

}    // namespace ntt
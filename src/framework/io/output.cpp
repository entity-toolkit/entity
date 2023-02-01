#include "output.h"

#include "wrapper.h"

#include "utils.h"

#include <iostream>
#include <string>
#include <vector>

namespace ntt {
  namespace {
    auto InterpretInputField_helper(const FieldID&          fid,
                                    const std::vector<int>& comps,
                                    const std::vector<int>& species) -> OutputField {
      OutputField of;
      of.setId(fid);
      std::string fldname { StringizeFieldID(fid) };
      for (auto c : comps) {
        of.comp.push_back(c);
#ifdef MINKOWSKI_METRIC
        fldname += (c == 0 ? "t" : (c == 1 ? "x" : (c == 2 ? "y" : "z")));
#else
        fldname += std::to_string(c);
#endif
      }
      if (species.size() > 0) {
        fldname += "_";
        for (auto s : species) {
          of.species.push_back(s);
          fldname += std::to_string(s);
          fldname += "_";
        }
        fldname.pop_back();
      }
      of.setName(fldname);
      return of;
    }

    auto InterpretInputField_getcomponents(const std::vector<std::string>& comps)
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

    auto InterpretInputField_getspecies(const std::string& fld) -> std::vector<int> {
      std::vector<int> species;
      if (fld.find("_") < fld.size()) {
        auto species_str = SplitString(fld.substr(fld.find("_") + 1), "_");
        for (const auto& specie : species_str) {
          species.push_back(std::stoi(specie));
        }
      }
      return species;
    }
  }    // namespace

  auto InterpretInputField(const std::string& fld) -> std::vector<OutputField> {
    std::vector<OutputField> ofs;
    if (fld.find("T") == 0) {
      const auto id      = FieldID::T;
      auto       species = InterpretInputField_getspecies(fld);
      auto comps = InterpretInputField_getcomponents({ fld.substr(1, 1), fld.substr(2, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, species));
      }
    } else if (fld.find("Rho") == 0) {
      const auto id      = FieldID::Rho;
      auto       species = InterpretInputField_getspecies(fld);
      ofs.emplace_back(InterpretInputField_helper(id, {}, species));
    } else if (fld.find("N") == 0) {
      const auto id      = FieldID::N;
      auto       species = InterpretInputField_getspecies(fld);
      ofs.emplace_back(InterpretInputField_helper(id, {}, species));
    } else if (fld.find("E") == 0) {
      const auto id    = FieldID::E;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else if (fld.find("B") == 0) {
      const auto id    = FieldID::B;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else if (fld.find("J") == 0) {
      const auto id    = FieldID::J;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else {
      NTTHostError("Invalid field name");
    }
    return ofs;
  }
}    // namespace ntt
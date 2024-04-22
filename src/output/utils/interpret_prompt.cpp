#include "output/utils/interpret_prompt.h"

#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"

#include <string>
#include <vector>

namespace out {

  auto InterpretSpecies(const std::string& in) -> std::vector<unsigned short> {
    std::vector<unsigned short> species;
    if (in.find("_") < in.size()) {
      auto species_str = fmt::splitString(in.substr(in.find("_") + 1), "_");
      for (const auto& specie : species_str) {
        species.push_back((unsigned short)(std::stoi(specie)));
      }
    }
    return species;
  }

  auto InterpretComponents(const std::vector<std::string>& comps)
    -> std::vector<std::vector<unsigned short>> {
    raise::ErrorIf(comps.size() > 2, "Invalid field name", HERE);
    std::vector<short> comps_int;
    for (auto& c : comps) {
      raise::ErrorIf(c != "t" && c != "x" && c != "y" && c != "z" && c != "0" &&
                       c != "1" && c != "2" && c != "3" && c != "i" && c != "j",
                     "Invalid field name",
                     HERE);
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
    std::vector<std::vector<unsigned short>> comps_ints;
    if (comps_int.size() == 1) {
      auto c = comps_int[0];
      if (c < 0) {
        for (int i { 0 }; i < 3; ++i) {
          comps_ints.push_back({ static_cast<unsigned short>(i + 1) });
        }
      } else {
        raise::ErrorIf((c > 3) || (c == 0), "Invalid field name", HERE);
        comps_ints.push_back({ static_cast<unsigned short>(c) });
      }
    } else {
      auto c1 = comps_int[0];
      auto c2 = comps_int[1];
      if (c1 < 0) {
        for (int i { 0 }; i < 3; ++i) {
          if (c2 < 0) {
            if (c2 == c1) {
              comps_ints.push_back({ static_cast<unsigned short>(i + 1),
                                     static_cast<unsigned short>(i + 1) });
            } else {
              for (int j { i }; j < 3; ++j) {
                comps_ints.push_back({ static_cast<unsigned short>(i + 1),
                                       static_cast<unsigned short>(j + 1) });
              }
            }
          } else {
            comps_ints.push_back({ static_cast<unsigned short>(i + 1),
                                   static_cast<unsigned short>(c2) });
          }
        }
      } else {
        if (c2 < 0) {
          for (int j { 0 }; j < 3; ++j) {
            comps_ints.push_back({ static_cast<unsigned short>(c1),
                                   static_cast<unsigned short>(j + 1) });
          }
        } else {
          comps_ints.push_back({ static_cast<unsigned short>(c1),
                                 static_cast<unsigned short>(c2) });
        }
      }
    }
    return comps_ints;
  }

} // namespace out
#include "output/fields.h"

#include "enums.h"

#include "utils/error.h"

#include <iostream>
#include <stdexcept>
#include <string>

auto main() -> int {
  using namespace out;
  using namespace ntt;
  try {
    {
      const auto e = OutputField(SimEngine::SRPIC, "E");
      raise::ErrorIf(not e.is_field(), "E should be a field", HERE);
      raise::ErrorIf(not e.is_efield(), "E should be a field", HERE);
      raise::ErrorIf(e.is_moment(), "E should not be a moment", HERE);
      raise::ErrorIf(e.id() != FldsID::E, "E should have ID FldsID::E", HERE);
      raise::ErrorIf(e.species.size() != 0, "E should have no species", HERE);
      raise::ErrorIf(e.comp.size() != 3, "E should have 3 components", HERE);
      raise::ErrorIf(e.name() != "fEi", "E should have name `Ei`", HERE);
      raise::ErrorIf(e.name(0) != "fE1", "E should have name `E1`", HERE);
      raise::ErrorIf(e.name(1) != "fE2", "E should have name `E2`", HERE);
      raise::ErrorIf(e.name(2) != "fE3", "E should have name `E3`", HERE);
      raise::ErrorIf(!(e.prepare_flag & PrepareOutput::ConvertToHat),
                     "E in SRPIC should be converted to hat",
                     HERE);
      raise::ErrorIf(!(e.interp_flag & PrepareOutput::InterpToCellCenterFromEdges),
                     "E should be interpolated to cell center from edges",
                     HERE);
    }

    {
      const auto rho = OutputField(SimEngine::SRPIC, "Rho_1_3");
      raise::ErrorIf(not rho.is_moment(), "Rho should be a moment", HERE);
      raise::ErrorIf(rho.is_field(), "Rho should not be a field", HERE);
      raise::ErrorIf(rho.id() != FldsID::Rho, "Rho should have ID FldsID::Rho", HERE);
      raise::ErrorIf(rho.name() != "fRho_1_3", "Rho should have name `Rho_1_3`", HERE);
      raise::ErrorIf(rho.comp.size() != 0, "Rho should have 0 components", HERE);
      raise::ErrorIf(rho.prepare_flag != PrepareOutput::None,
                     "Rho should not have any prepare flags",
                     HERE);
      raise::ErrorIf(rho.interp_flag != PrepareOutput::None,
                     "Rho should not have any interp flags",
                     HERE);
      raise::ErrorIf(not(rho.species == std::vector<unsigned short> { 1, 3 }),
                     "Rho should have species 1 and 3",
                     HERE);
    }

    {
      const auto t = OutputField(SimEngine::GRPIC, "Tti_2_3");
      raise::ErrorIf(not t.is_moment(), "T should be a moment", HERE);
      raise::ErrorIf(t.is_field(), "T should not be a field", HERE);
      raise::ErrorIf(t.id() != FldsID::T, "T should have ID FldsID::T", HERE);
      raise::ErrorIf(t.name() != "fTti_2_3", "T should have name `Tti_2_3`", HERE);
      raise::ErrorIf(t.name(0) != "fT01_2_3", "T should have name `T01_2_3`", HERE);
      raise::ErrorIf(t.name(1) != "fT02_2_3", "T should have name `T02_2_3`", HERE);
      raise::ErrorIf(t.name(2) != "fT03_2_3", "T should have name `T03_2_3`", HERE);
      raise::ErrorIf(t.comp.size() != 3, "T should have 3 component", HERE);
      raise::ErrorIf(t.comp[0].size() != 2,
                     "T.comp[0] should have 2 components",
                     HERE);
      raise::ErrorIf(t.comp[1].size() != 2,
                     "T.comp[1] should have 2 components",
                     HERE);
      raise::ErrorIf(t.comp[2].size() != 2,
                     "T.comp[2] should have 2 components",
                     HERE);
      raise::ErrorIf(t.comp[0] != std::vector<unsigned short> { 0, 1 },
                     "T.comp[0] should be {0, 1}",
                     HERE);
      raise::ErrorIf(t.comp[1] != std::vector<unsigned short> { 0, 2 },
                     "T.comp[1] should be {0, 2}",
                     HERE);
      raise::ErrorIf(t.comp[2] != std::vector<unsigned short> { 0, 3 },
                     "T.comp[2] should be {0, 3}",
                     HERE);
      raise::ErrorIf(t.species.size() != 2, "T should have 2 species", HERE);
      raise::ErrorIf(t.species[0] != 2, "T should have specie 2", HERE);
      raise::ErrorIf(t.species[1] != 3, "T should have specie 3", HERE);
    }
    {
      const auto t = OutputField(SimEngine::GRPIC, "Tij");
      raise::ErrorIf(t.comp.size() != 6, "T should have 6 component", HERE);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}

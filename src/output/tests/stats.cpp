#include "output/stats.h"

#include "enums.h"

#include "utils/error.h"

#include <iostream>
#include <stdexcept>
#include <string>

auto main() -> int {
  using namespace stats;
  using namespace ntt;
  try {
    {
      const auto e = OutputStats("E^2");
      raise::ErrorIf(e.is_vector(), "E^2 should not be a vector quantity", HERE);
      raise::ErrorIf(e.is_moment(), "E^2 should not be a moment", HERE);
      raise::ErrorIf(e.id() != StatsID::E2, "E^2 should have ID StatsID::E2", HERE);
      raise::ErrorIf(e.species.size() != 0, "E^2 should have no species", HERE);
      raise::ErrorIf(e.comp.size() != 0, "E^2 should have no components", HERE);
      raise::ErrorIf(e.name() != "E^2", "E^2 should have name `E^2`", HERE);
    }

    {
      const auto e = OutputStats("ExB");
      raise::ErrorIf(not e.is_vector(), "ExB should be a vector quantity", HERE);
      raise::ErrorIf(e.is_moment(), "ExB should not be a moment", HERE);
      raise::ErrorIf(e.id() != StatsID::ExB, "ExB should have ID StatsID::ExB", HERE);
      raise::ErrorIf(e.species.size() != 0, "ExB should have no species", HERE);
      raise::ErrorIf(e.comp.size() != 3, "ExB should have 3 components", HERE);
      raise::ErrorIf(e.name() != "ExBi", "ExB should have name `ExBi`", HERE);
    }

    {
      const auto e = OutputStats("J.E");
      raise::ErrorIf(e.is_vector(), "J.E should not be a vector quantity", HERE);
      raise::ErrorIf(e.is_moment(), "J.E should not be a moment", HERE);
      raise::ErrorIf(e.id() != StatsID::JdotE,
                     "J.E should have ID StatsID::JdotE",
                     HERE);
      raise::ErrorIf(e.species.size() != 0, "J.E should have no species", HERE);
      raise::ErrorIf(e.comp.size() != 0, "J.E should have no components", HERE);
      raise::ErrorIf(e.name() != "J.E", "J.E should have name `J.E`", HERE);
    }

    {
      const auto rho = OutputStats("Rho_1_3");
      raise::ErrorIf(not rho.is_moment(), "Rho should be a moment", HERE);
      raise::ErrorIf(rho.id() != StatsID::Rho,
                     "Rho should have ID StatsID::Rho",
                     HERE);
      raise::ErrorIf(rho.name() != "Rho_1_3", "Rho should have name `Rho_1_3`", HERE);
      raise::ErrorIf(rho.comp.size() != 0, "Rho should have 0 components", HERE);
      raise::ErrorIf(not(rho.species == std::vector<spidx_t> { 1, 3 }),
                     "Rho should have species 1 and 3",
                     HERE);
    }

    {
      const auto t = OutputStats("Tti_2_3");
      raise::ErrorIf(not t.is_moment(), "T should be a moment", HERE);
      raise::ErrorIf(t.is_vector(), "T should not be a vector quantity", HERE);
      raise::ErrorIf(t.id() != StatsID::T, "T should have ID StatsID::T", HERE);
      raise::ErrorIf(t.name() != "Tti_2_3", "T should have name `Tti_2_3`", HERE);
      raise::ErrorIf(t.name(0) != "T01_2_3", "T should have name `T01_2_3`", HERE);
      raise::ErrorIf(t.name(1) != "T02_2_3", "T should have name `T02_2_3`", HERE);
      raise::ErrorIf(t.name(2) != "T03_2_3", "T should have name `T03_2_3`", HERE);
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
      const auto t = OutputStats("Tij");
      raise::ErrorIf(t.comp.size() != 6, "T should have 6 component", HERE);
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  return 0;
}

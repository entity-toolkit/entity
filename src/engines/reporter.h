#ifndef ENGINES_REPORTER_H
#define ENGINES_REPORTER_H

#include "enums.h"

#include "utils/reporter.h"

#include "archetypes/traits.h"
#include "framework/parameters/parameters.h"

#include <string>
#include <vector>

namespace ntt {

  auto ReportSimulationConfig(const SimulationParams&,
                              SimEngine,
                              Metric,
                              real_t,
                              simtime_t,
                              timestep_t,
                              const std::vector<unsigned int>&,
                              unsigned int) -> std::string;

  template <class PG, class Dom>
  inline auto ReportPgenConfig(const PG& pgen, const std::string& pgen_name)
    -> std::string {
    std::string report  = "";
    report             += "\n";
    reporter::AddCategory(report, 4, "Problem generator");
    reporter::AddParam(report, 6, "Name", "%s", pgen_name.c_str());
    reporter::AddSubcategory(report, 6, "Methods defined");

    const auto BoolToOnOff = [](bool toggle) -> const char* {
      return toggle ? "ON" : "OFF";
    };

    reporter::AddParam(report,
                       8,
                       "InitFlds",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasInitFlds<PG>));
    reporter::AddParam(report,
                       8,
                       "InitPrtls",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasInitPrtls<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "CustomPostStep",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasCustomPostStep<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "ExternalFields",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasExternalFields<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "ext_current",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasExtCurrent<PG>));
    reporter::AddParam(report,
                       8,
                       "AtmFields",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasAtmFields<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFields",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasMatchFields<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX1",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasMatchFieldsInX1<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX2",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasMatchFieldsInX2<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX3",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasMatchFieldsInX3<PG>));
    reporter::AddParam(report,
                       8,
                       "FixFieldsConst",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasFixFieldsConst<PG>));
    reporter::AddParam(report,
                       8,
                       "EmissionPolicy",
                       "%s",
                       BoolToOnOff(arch::traits::pgen::HasEmissionPolicy<PG, Dom>));
    reporter::AddParam(
      report,
      8,
      "CustomFieldOutput",
      "%s",
      BoolToOnOff(arch::traits::pgen::HasCustomFieldOutput<PG, Dom>));
    reporter::AddParam(
      report,
      8,
      "CustomStatOutput",
      "%s",
      BoolToOnOff(arch::traits::pgen::HasCustomStatOutput<PG, Dom>));
    return report;
  }

} // namespace ntt

#endif // ENGINES_REPORTER_H

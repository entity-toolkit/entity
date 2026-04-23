/**
 * @file engines/reporter.h
 * @brief Functions for reporting simulation and problem generator configuration
 * @implements
 *   - ntt::ReportSimulationConfig -> std::string
 *   - ntt::ReportPgenConfig<> -> std::string
 * @namespaces:
 *   - ntt::
 */

#ifndef ENGINES_REPORTER_H
#define ENGINES_REPORTER_H

#include "enums.h"

#include "traits/pgen.h"
#include "utils/reporter.h"

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
  inline auto ReportPgenConfig(const PG& /*pgen*/, const std::string& pgen_name)
    -> std::string {
    std::string report;
    report += "\n";
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
                       BoolToOnOff(::traits::pgen::HasInitFlds<PG>));
    reporter::AddParam(report,
                       8,
                       "InitPrtls",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasInitPrtls<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "CustomPostStep",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasCustomPostStep<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "ExternalFields",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasExternalFields<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "ext_current",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasExtCurrent<PG>));
    reporter::AddParam(report,
                       8,
                       "AtmFields",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasAtmFields<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFields",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasMatchFields<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX1",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasMatchFieldsInX1<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX2",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasMatchFieldsInX2<PG>));
    reporter::AddParam(report,
                       8,
                       "MatchFieldsInX3",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasMatchFieldsInX3<PG>));
    reporter::AddParam(report,
                       8,
                       "FixFieldsConst",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasFixFieldsConst<PG>));
    reporter::AddParam(report,
                       8,
                       "EmissionPolicy",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasEmissionPolicy<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "CustomFieldOutput",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasCustomFieldOutput<PG, Dom>));
    reporter::AddParam(report,
                       8,
                       "CustomStatOutput",
                       "%s",
                       BoolToOnOff(::traits::pgen::HasCustomStatOutput<PG, Dom>));
    return report;
  }

} // namespace ntt

#endif // ENGINES_REPORTER_H

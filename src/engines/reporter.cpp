#include "engines/reporter.h"

#include "enums.h"
#include "global.h"

#include "utils/formatting.h"
#include "utils/reporter.h"

#include "framework/parameters/extra.h"
#include "framework/parameters/parameters.h"

#include <string>
#include <vector>

namespace ntt {

  auto ReportSimulationConfig(const SimulationParams&          params,
                              SimEngine                        S,
                              Metric                           M,
                              real_t                           dt,
                              simtime_t                        runtime,
                              timestep_t                       max_steps,
                              const std::vector<unsigned int>& ndomains_per_dim,
                              unsigned int ndomains) -> std::string {
    std::string report;
    /*
     * Simulation configs
     */
    reporter::AddCategory(report, 4, "Configuration");
    reporter::AddParam(report,
                       4,
                       "Name",
                       "%s",
                       params.template get<std::string>("simulation.name").c_str());
    reporter::AddParam(report, 4, "Engine", "%s", SimEngine(S).to_string());
    reporter::AddParam(report, 4, "Metric", "%s", M.to_string());
#if SHAPE_ORDER == 0
    reporter::AddParam(report, 4, "Deposit", "%s", "zigzag");
#else
    reporter::AddParam(report, 4, "Deposit", "%s", "esirkepov");
    reporter::AddParam(report, 4, "Interpolation order", "%i", SHAPE_ORDER);
#endif
    reporter::AddParam(report, 4, "Timestep [dt]", "%.3e", dt);
    reporter::AddParam(report, 4, "Runtime", "%.3e [%d steps]", runtime, max_steps);
    report += "\n";
    reporter::AddCategory(report, 4, "Global domain");
    reporter::AddParam(
      report,
      4,
      "Resolution",
      "%s",
      params.template stringize<ncells_t>("grid.resolution").c_str());
    reporter::AddParam(report,
                       4,
                       "Extent",
                       "%s",
                       params.template stringize<real_t>("grid.extent").c_str());
    reporter::AddParam(report,
                       4,
                       "Fiducial cell size [dx0]",
                       "%.3e",
                       params.template get<real_t>("scales.dx0"));
    reporter::AddSubcategory(report, 4, "Boundary conditions");
    reporter::AddParam(
      report,
      6,
      "Fields",
      "%s",
      params.template stringize<FldsBC>("grid.boundaries.fields").c_str());
    reporter::AddParam(
      report,
      6,
      "Particles",
      "%s",
      params.template stringize<PrtlBC>("grid.boundaries.particles").c_str());
    reporter::AddParam(report,
                       4,
                       "Domain decomposition",
                       "%s [%d total]",
                       fmt::formatVector(ndomains_per_dim).c_str(),
                       ndomains);
    report += "\n";
    reporter::AddCategory(report, 4, "Nominal parameters");
    reporter::AddParam(report,
                       4,
                       "Particles per cell [ppc0]",
                       "%.1f",
                       params.template get<real_t>("particles.ppc0"));
    reporter::AddParam(report,
                       4,
                       "Larmor radius [larmor0]",
                       "%.3e [%.3f dx0]",
                       params.template get<real_t>("scales.larmor0"),
                       params.template get<real_t>("scales.larmor0") /
                         params.template get<real_t>("scales.dx0"));
    reporter::AddParam(report,
                       4,
                       "Larmor frequency [omegaB0 * dt]",
                       "%.3e",
                       params.template get<real_t>("scales.omegaB0") *
                         params.template get<real_t>("algorithms.timestep.dt"));
    reporter::AddParam(report,
                       4,
                       "Skin depth [skindepth0]",
                       "%.3e [%.3f dx0]",
                       params.template get<real_t>("scales.skindepth0"),
                       params.template get<real_t>("scales.skindepth0") /
                         params.template get<real_t>("scales.dx0"));
    reporter::AddParam(report,
                       4,
                       "Plasma frequency [omp0 * dt]",
                       "%.3e",
                       params.template get<real_t>("algorithms.timestep.dt") /
                         params.template get<real_t>("scales.skindepth0"));
    reporter::AddParam(report,
                       4,
                       "Magnetization [sigma0]",
                       "%.3e",
                       params.template get<real_t>("scales.sigma0"));

    if (params.contains("radiation.emission.compton.photon_species")) {
      reporter::AddCategory(report, 4, "- Compton emission");
      reporter::AddParam(report,
                         6,
                         "Nominal probability",
                         "%.3e",
                         params.template get<real_t>(
                           "radiation.emission.compton.nominal_probability"));
      reporter::AddParam(report,
                         6,
                         "Nominal photon energy",
                         "%.3e",
                         params.template get<real_t>(
                           "radiation.emission.compton.nominal_photon_energy"));
    }
    if (params.contains("radiation.emission.synchrotron.photon_species")) {
      reporter::AddCategory(report, 4, "- Synchrotron emission");
      reporter::AddParam(
        report,
        6,
        "Nominal probability",
        "%.3e",
        params.template get<real_t>(
          "radiation.emission.synchrotron.nominal_probability"));
      reporter::AddParam(
        report,
        6,
        "Nominal photon energy",
        "%.3e",
        params.template get<real_t>(
          "radiation.emission.synchrotron.nominal_photon_energy"));
    }

    report += "\n";
    const auto two_body_interactions =
      params.template get<std::vector<::ntt::params::TwoBodyInteractionParams>>(
        "two_body.interaction");
    if (not two_body_interactions.empty()) {
      reporter::AddCategory(report, 4, "Two-body interactions");
      reporter::AddParam(
        report,
        6,
        "Thomson optical depth",
        "%.3e",
        params.template get<real_t>("two_body.thomson_optical_depth"));
      for (const auto& interaction : two_body_interactions) {
        reporter::AddSubcategory(
          report,
          6,
          TwoBodyInteraction::to_string(interaction.type).c_str());
        std::vector<int> group1(interaction.group1.size());
        std::vector<int> group2(interaction.group2.size());
        for (size_t g1 = 0; g1 < interaction.group1.size(); ++g1) {
          group1[g1] = interaction.group1[g1];
        }
        for (size_t g2 = 0; g2 < interaction.group2.size(); ++g2) {
          group2[g2] = interaction.group2[g2];
        }
        reporter::AddParam(report,
                           8,
                           "group #1 species",
                           "%s (recoil: %s)",
                           fmt::formatVector(group1).c_str(),
                           interaction.recoil1 ? "ON" : "OFF");
        reporter::AddParam(report,
                           8,
                           "group #2 species",
                           "%s (recoil: %s)",
                           fmt::formatVector(group2).c_str(),
                           interaction.recoil2 ? "ON" : "OFF");
        reporter::AddParam(report, 8, "tile size [cells]", "%u", interaction.tile_size);
        reporter::AddParam(report, 8, "interval [steps]", "%u", interaction.interval);
      }
    }
    return report;
  }

} // namespace ntt

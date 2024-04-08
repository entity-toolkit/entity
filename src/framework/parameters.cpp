#include "framework/parameters.h"

#include "defaults.h"
#include "enums.h"

#include "arch/kokkos_aliases.h"
#include "framework/species.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/log.h"
#include "utils/numeric.h"
// metrics
#include "metrics/kerr_schild.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "metrics/kerr_schild_0.h"
// < metrics

#include <toml.hpp>

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace ntt {

  template <typename M>
  auto get_dx0_V0(
    const std::vector<unsigned int>&              resolution,
    const std::vector<std::pair<real_t, real_t>>& extent,
    const std::map<std::string, real_t>& params) -> std::pair<real_t, real_t> {
    const auto      metric = M(resolution, extent, params);
    const auto      dx0    = metric.dxMin();
    coord_t<M::Dim> x_corner { ZERO };
    for (unsigned short d { 0 }; d < (unsigned short)(M::Dim); ++d) {
      x_corner[d] = HALF;
    }
    const auto V0 = metric.sqrt_det_h(x_corner);
    return { dx0, V0 };
  }

  SimulationParams::SimulationParams(const toml::value& data) :
    raw_data { data } {
    /* [simulation] --------------------------------------------------------- */
    set("simulation.name", toml::find<std::string>(raw_data, "simulation", "name"));
    set("simulation.runtime",
        toml::find<real_t>(raw_data, "simulation", "runtime"));

    const auto engine = fmt::toLower(
      toml::find<std::string>(raw_data, "simulation", "engine"));
    const auto engine_enum = SimEngine::pick(engine.c_str());
    set("simulation.engine", engine_enum);

    /* [grid] --------------------------------------------------------------- */
    const auto res = toml::find<std::vector<unsigned int>>(raw_data,
                                                           "grid",
                                                           "resolution");
    raise::ErrorIf(res.size() < 1 || res.size() > 3,
                   "invalid `grid.resolution`",
                   HERE);
    set("grid.resolution", res);
    const auto dim = static_cast<Dimension>(res.size());
    set("grid.dim", dim);

    auto extent = toml::find<std::vector<std::vector<real_t>>>(raw_data,
                                                               "grid",
                                                               "extent");
    raise::ErrorIf(extent.size() < 1 || extent.size() > 3,
                   "invalid `grid.extent`",
                   HERE);
    promiseToDefine("grid.extent");

    /* [grid.metric] -------------------------------------------------------- */
    const auto metric = fmt::toLower(
      toml::find<std::string>(raw_data, "grid", "metric", "metric"));
    const auto metric_enum = Metric::pick(metric.c_str());
    promiseToDefine("grid.metric.metric");
    std::string coord;
    if (metric == "minkowski") {
      raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                     "minkowski metric is only supported for SRPIC",
                     HERE);
      coord = "cart";
    } else if (metric[0] == 'q') {
      // quasi-spherical geometry
      raise::ErrorIf(dim == Dim::_1D,
                     "not enough dimensions for qspherical geometry",
                     HERE);
      raise::ErrorIf(dim == Dim::_3D,
                     "3D not implemented for qspherical geometry",
                     HERE);
      coord = "qsph";
      set("grid.metric.qsph_r0",
          toml::find_or(raw_data, "grid", "metric", "qsph_r0", defaults::qsph::r0));
      set("grid.metric.qsph_h",
          toml::find_or(raw_data, "grid", "metric", "qsph_h", defaults::qsph::h));
    } else {
      // spherical geometry
      raise::ErrorIf(dim == Dim::_1D,
                     "not enough dimensions for spherical geometry",
                     HERE);
      raise::ErrorIf(dim == Dim::_3D,
                     "3D not implemented for spherical geometry",
                     HERE);
      coord = "sph";
    }
    if ((engine_enum == SimEngine::GRPIC) &&
        (metric_enum != Metric::Kerr_Schild_0)) {
      const auto ks_a = toml::find_or(raw_data,
                                      "grid",
                                      "metric",
                                      "ks_a",
                                      defaults::ks::a);
      set("grid.metric.ks_a", ks_a);
      set("grid.metric.ks_rh", ONE + math::sqrt(ONE - SQR(ks_a)));
    }
    const auto coord_enum = Coord::pick(coord.c_str());
    set("grid.metric.coord", coord_enum);

    /* [grid.boundaraies] --------------------------------------------------- */
    auto flds_bc = toml::find<std::vector<std::vector<std::string>>>(
      raw_data,
      "grid",
      "boundaries",
      "fields");
    raise::ErrorIf(flds_bc.size() < 1 || flds_bc.size() > 3,
                   "invalid `grid.boundaries.fields`",
                   HERE);
    promiseToDefine("grid.boundaries.fields");
    for (const auto& bcs : flds_bc) {
      for (const auto& bc : bcs) {
        if (fmt::toLower(bc) == "absorb") {
          promiseToDefine("grid.boundaries.absorb_d");
          promiseToDefine("grid.boundaries.absorb_coeff");
        }
      }
    }

    auto prtl_bc = toml::find<std::vector<std::vector<std::string>>>(
      raw_data,
      "grid",
      "boundaries",
      "particles");
    raise::ErrorIf(prtl_bc.size() < 1 || prtl_bc.size() > 3,
                   "invalid `grid.boundaries.particles`",
                   HERE);
    promiseToDefine("grid.boundaries.particles");
    for (const auto& bcs : prtl_bc) {
      for (const auto& bc : bcs) {
        if (fmt::toLower(bc) == "absorb") {
          promiseToDefine("grid.boundaries.absorb_d");
          promiseToDefine("grid.boundaries.absorb_coeff");
        }
      }
    }

    /* [scales] ------------------------------------------------------------- */
    const auto larmor0 = toml::find<real_t>(raw_data, "scales", "larmor0");
    const auto skindepth0 = toml::find<real_t>(raw_data, "scales", "skindepth0");
    raise::ErrorIf(larmor0 <= ZERO || skindepth0 <= ZERO,
                   "larmor0 and skindepth0 must be positive",
                   HERE);
    set("scales.larmor0", larmor0);
    set("scales.skindepth0", skindepth0);
    promiseToDefine("scales.dx0");
    promiseToDefine("scales.V0");
    promiseToDefine("scales.n0");
    promiseToDefine("scales.q0");
    set("scales.sigma0", SQR(skindepth0 / larmor0));
    set("scales.B0", ONE / larmor0);
    set("scales.omegaB0", ONE / larmor0);

    /* [algorithms] --------------------------------------------------------- */
    set("algorithms.current_filters",
        toml::find_or(raw_data,
                      "algorithms",
                      "current_filters",
                      defaults::current_filters));

    /* [algorithms.toggles] ------------------------------------------------- */
    set("algorithms.toggles.fieldsolver",
        toml::find_or(raw_data, "algorithms", "toggles", "fieldsolver", true));
    set("algorithms.toggles.deposit",
        toml::find_or(raw_data, "algorithms", "toggles", "deposit", true));
    set("algorithms.toggles.extforce",
        toml::find_or(raw_data, "algorithms", "toggles", "extforce", false));

    /* [algorithms.timestep] ------------------------------------------------ */
    set("algorithms.timestep.CFL",
        toml::find_or(raw_data, "algorithms", "timestep", "CFL", defaults::cfl));
    set("algorithms.timestep.correction",
        toml::find_or(raw_data,
                      "algorithms",
                      "timestep",
                      "correction",
                      defaults::correction));

    /* [algorithms.gr] ------------------------------------------------------ */
    if (engine_enum == SimEngine::GRPIC) {
      set("algorithms.gr.pusher_eps",
          toml::find_or(raw_data,
                        "algorithms",
                        "gr",
                        "pusher_eps",
                        defaults::gr::pusher_eps));
      set("algorithms.gr.pusher_niter",
          toml::find_or(raw_data,
                        "algorithms",
                        "gr",
                        "pusher_niter",
                        defaults::gr::pusher_niter));
    }

    /* [particles] ---------------------------------------------------------- */
    const auto ppc0 = toml::find<real_t>(raw_data, "particles", "ppc0");
    set("particles.ppc0", ppc0);
    raise::ErrorIf(ppc0 <= 0.0, "ppc0 must be positive", HERE);
    set("particles.use_weights",
        toml::find_or(raw_data, "particles", "use_weights", false));
    set("particles.sort_interval",
        toml::find_or(raw_data, "particles", "sort_interval", defaults::sort_interval));

    /* [particles.species] -------------------------------------------------- */
    std::vector<ParticleSpecies> species;
    const auto species_tab = toml::find<toml::array>(raw_data, "particles", "species");
    set("particles.nspec", species_tab.size());

    int idx = 1;
    for (const auto& sp : species_tab) {
      const auto label  = toml::find_or<std::string>(sp,
                                                    "label",
                                                    "s" + std::to_string(idx));
      const auto mass   = toml::find<float>(sp, "mass");
      const auto charge = toml::find<float>(sp, "charge");
      raise::ErrorIf((charge != 0.0f) && (mass == 0.0f),
                     "mass of the charged species must be non-zero",
                     HERE);
      const auto is_massless   = (mass == 0.0f) && (charge == 0.0f);
      const auto def_pusher    = (is_massless ? defaults::ph_pusher
                                              : defaults::em_pusher);
      const auto maxnpart_real = toml::find<double>(sp, "maxnpart");
      const auto maxnpart      = static_cast<std::size_t>(maxnpart_real);
      const auto pusher = toml::find_or(sp, "pusher", std::string(def_pusher));
      const auto npayloads = toml::find_or(sp,
                                           "n_payloads",
                                           static_cast<unsigned short>(0));
      const auto cooling   = toml::find_or(sp, "cooling", std::string("None"));
      raise::ErrorIf((fmt::toLower(cooling) != "none") && is_massless,
                     "cooling is only applicable to massive particles",
                     HERE);
      raise::ErrorIf((fmt::toLower(pusher) == "photon") && !is_massless,
                     "photon pusher is only applicable to massless particles",
                     HERE);
      const auto pusher_enum  = PrtlPusher::pick(pusher.c_str());
      const auto cooling_enum = Cooling::pick(cooling.c_str());
      if ((pusher_enum == PrtlPusher::VAY_GCA) ||
          (pusher_enum == PrtlPusher::BORIS_GCA)) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "GCA pushers are only supported for SRPIC",
                       HERE);
        promiseToDefine("algorithms.gca.e_ovr_b_max");
        promiseToDefine("algorithms.gca.larmor_max");
      }
      if (cooling_enum == Cooling::SYNCHROTRON) {
        raise::ErrorIf(engine_enum != SimEngine::SRPIC,
                       "Synchrotron cooling is only supported for SRPIC",
                       HERE);
        promiseToDefine("algorithms.synchrotron.gamma_rad");
      }

      species.emplace_back(ParticleSpecies(idx,
                                           label,
                                           mass,
                                           charge,
                                           maxnpart,
                                           pusher_enum,
                                           cooling_enum,
                                           npayloads));
      idx += 1;
    }
    set("particles.species", species);

    /* [output] ------------------------------------------------------------- */
    const auto flds_out = toml::find_or(raw_data,
                                        "output",
                                        "fields",
                                        std::vector<std::string> {});
    const auto prtl_out = toml::find_or(raw_data,
                                        "output",
                                        "particles",
                                        std::vector<std::string> {});
    if (flds_out.size() == 0) {
      raise::Warning("No fields output specified", HERE);
    }
    if (prtl_out.size() == 0) {
      raise::Warning("No particle output specified", HERE);
    }
    set("output.fields", flds_out);
    set("output.particles", prtl_out);

    set("output.format",
        toml::find_or(raw_data, "output", "format", defaults::output::format));
    set("output.mom_smooth",
        toml::find_or(raw_data, "output", "mom_smooth", defaults::output::mom_smooth));
    set("output.flds_stride",
        toml::find_or(raw_data, "output", "flds_stride", defaults::output::flds_stride));
    set("output.prtl_stride",
        toml::find_or(raw_data, "output", "prtl_stride", defaults::output::prtl_stride));
    set("output.interval",
        toml::find_or(raw_data, "output", "interval", defaults::output::interval));
    set("output.interval_time",
        toml::find_or(raw_data, "output", "interval_time", -ONE));

    /* [output.debug] ------------------------------------------------------- */
    set("output.debug.as_is",
        toml::find_or(raw_data, "output", "debug", "as_is", false));
    set("output.debug.ghosts",
        toml::find_or(raw_data, "output", "debug", "ghosts", false));

    /* [diagnostics] -------------------------------------------------------- */
    set("diagnostics.interval",
        toml::find_or(raw_data, "diagnostics", "interval", defaults::diag::interval));
    set("diagnostics.log_level",
        toml::find_or(raw_data, "diagnostics", "log_level", defaults::diag::log_level));
    set("diagnostics.blocking_timers",
        toml::find_or(raw_data, "diagnostics", "blocking_timers", false));

    /* inferred variables --------------------------------------------------- */
    // extent
    if (extent.size() > dim) {
      extent.erase(extent.begin() + (std::size_t)(dim), extent.end());
    }
    raise::ErrorIf(extent[0].size() != 2, "invalid `grid.extent[0]`", HERE);
    if (coord_enum != Coord::Cart) {
      raise::ErrorIf(extent.size() > 1,
                     "invalid `grid.extent` for non-cartesian geometry",
                     HERE);
      extent.push_back({ ZERO, constant::PI });
      if (dim == Dim::_3D) {
        extent.push_back({ ZERO, TWO * constant::PI });
      }
    }
    raise::ErrorIf(extent.size() != dim, "invalid inferred `grid.extent`", HERE);
    set("grid.extent", extent);

    // fields/particle boundaries
    std::vector<std::vector<FldsBC>> flds_bc_enum;
    std::vector<std::vector<PrtlBC>> prtl_bc_enum;
    if (coord_enum == Coord::Cart) {
      raise::ErrorIf(flds_bc.size() != (std::size_t)dim,
                     "invalid `grid.boundaries.fields`",
                     HERE);
      raise::ErrorIf(prtl_bc.size() != (std::size_t)dim,
                     "invalid `grid.boundaries.particles`",
                     HERE);
      for (unsigned short d = 0; d < (unsigned short)dim; ++d) {
        flds_bc_enum.push_back({});
        prtl_bc_enum.push_back({});
        const auto fbc = flds_bc[d];
        const auto pbc = prtl_bc[d];
        raise::ErrorIf(fbc.size() < 1 || fbc.size() > 2,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        raise::ErrorIf(pbc.size() < 1 || pbc.size() > 2,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[0]).c_str());
        auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[0]).c_str());
        if (fbc.size() == 1) {
          raise::ErrorIf(fbc_enum != FldsBC::PERIODIC,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
          flds_bc_enum.back().push_back(FldsBC(FldsBC::PERIODIC));
        } else {
          raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          flds_bc_enum.back().push_back(fbc_enum);
          auto fbc_enum = FldsBC::pick(fmt::toLower(fbc[1]).c_str());
          raise::ErrorIf(fbc_enum == FldsBC::PERIODIC,
                         "invalid `grid.boundaries.fields`",
                         HERE);
          flds_bc_enum.back().push_back(fbc_enum);
        }
        if (pbc.size() == 1) {
          raise::ErrorIf(pbc_enum != PrtlBC::PERIODIC,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
          prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
        } else {
          raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          prtl_bc_enum.back().push_back(pbc_enum);
          auto pbc_enum = PrtlBC::pick(fmt::toLower(pbc[1]).c_str());
          raise::ErrorIf(pbc_enum == PrtlBC::PERIODIC,
                         "invalid `grid.boundaries.particles`",
                         HERE);
          prtl_bc_enum.back().push_back(pbc_enum);
        }
      }
    } else {
      raise::ErrorIf(flds_bc.size() > 1, "invalid `grid.boundaries.fields`", HERE);
      raise::ErrorIf(prtl_bc.size() > 1, "invalid `grid.boundaries.particles`", HERE);
      if (engine_enum == SimEngine::SRPIC) {
        raise::ErrorIf(flds_bc[0].size() != 2,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        flds_bc_enum.push_back(
          { FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()),
            FldsBC::pick(fmt::toLower(flds_bc[0][1]).c_str()) });
        flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
        if (dim == Dim::_3D) {
          flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        }
        raise::ErrorIf(prtl_bc[0].size() != 2,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        prtl_bc_enum.push_back(
          { PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()),
            PrtlBC::pick(fmt::toLower(prtl_bc[0][1]).c_str()) });
        prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
        if (dim == Dim::_3D) {
          prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
        }
      } else {
        raise::ErrorIf(flds_bc[0].size() != 1,
                       "invalid `grid.boundaries.fields`",
                       HERE);
        raise::ErrorIf(prtl_bc[0].size() != 1,
                       "invalid `grid.boundaries.particles`",
                       HERE);
        flds_bc_enum.push_back(
          { FldsBC::HORIZON, FldsBC::pick(fmt::toLower(flds_bc[0][0]).c_str()) });
        flds_bc_enum.push_back({ FldsBC::AXIS, FldsBC::AXIS });
        if (dim == Dim::_3D) {
          flds_bc_enum.push_back({ FldsBC::PERIODIC, FldsBC::PERIODIC });
        }
        prtl_bc_enum.push_back(
          { PrtlBC::HORIZON, PrtlBC::pick(fmt::toLower(prtl_bc[0][0]).c_str()) });
        prtl_bc_enum.push_back({ PrtlBC::AXIS, PrtlBC::AXIS });
        if (dim == Dim::_3D) {
          prtl_bc_enum.push_back({ PrtlBC::PERIODIC, PrtlBC::PERIODIC });
        }
      }
    }

    raise::ErrorIf(flds_bc_enum.size() != (std::size_t)dim,
                   "invalid inferred `grid.boundaries.fields`",
                   HERE);
    raise::ErrorIf(prtl_bc_enum.size() != (std::size_t)dim,
                   "invalid inferred `grid.boundaries.particles`",
                   HERE);
    for (unsigned short d = 0; d < (unsigned short)dim; ++d) {
      raise::ErrorIf(
        flds_bc_enum[d].size() != 2,
        fmt::format("invalid inferred `grid.boundaries.fields[%d]`", d),
        HERE);
      raise::ErrorIf(
        prtl_bc_enum[d].size() != 2,
        fmt::format("invalid inferred `grid.boundaries.particles[%d]`", d),
        HERE);
    }
    set("grid.boundaries.fields", flds_bc_enum);
    set("grid.boundaries.particles", prtl_bc_enum);

    if (isPromised("grid.boundaries.absorb_d")) {
      if (coord_enum == Coord::Cart) {
        auto min_extent = std::numeric_limits<real_t>::max();
        for (const auto& e : extent) {
          min_extent = std::min(min_extent, e[1] - e[0]);
        }
        set("grid.boundaries.absorb_d",
            toml::find_or(raw_data,
                          "grid",
                          "boundaries",
                          "absorb_d",
                          min_extent * defaults::bc::d_absorb_frac));
      } else {
        auto r_extent = extent[0][1] - extent[0][0];
        set("grid.boundaries.absorb_d",
            toml::find_or(raw_data,
                          "grid",
                          "boundaries",
                          "absorb_d",
                          r_extent * defaults::bc::d_absorb_frac));
      }
      set("grid.boundaries.absorb_coeff",
          toml::find_or(raw_data,
                        "grid",
                        "boundaries",
                        "absorb_coeff",
                        defaults::bc::absorb_coeff));
    }

    // gca
    if (isPromised("algorithms.gca.e_ovr_b_max")) {
      set("algorithms.gca.e_ovr_b_max",
          toml::find_or(raw_data,
                        "algorithms",
                        "gca",
                        "e_ovr_b_max",
                        defaults::gca::EovrB_max));
      set("algorithms.gca.larmor_max",
          toml::find_or(raw_data, "algorithms", "gca", "larmor_max", ZERO));
    }

    // cooling
    if (isPromised("algorithms.synchrotron.gamma_rad")) {
      set("algorithms.synchrotron.gamma_rad",
          toml::find_or(raw_data,
                        "algorithms",
                        "synchrotron",
                        "gamma_rad",
                        defaults::synchrotron::gamma_rad));
    }

    // metric, dx0, V0, n0, q0
    {
      std::vector<std::pair<real_t, real_t>> ext;
      for (const auto& e : extent) {
        ext.push_back({ e[0], e[1] });
      }
      std::map<std::string, real_t> params;
      if (coord_enum == Coord::Qsph) {
        params["r0"] = get<real_t>("grid.metric.qsph_r0");
        params["h"]  = get<real_t>("grid.metric.qsph_h");
      }
      if ((engine_enum == SimEngine::GRPIC) &&
          (metric_enum != Metric::Kerr_Schild_0)) {
        params["a"] = get<real_t>("grid.metric.ks_a");
      }
      std::pair<real_t, real_t> dx0_V0;
      if (metric_enum == Metric::Minkowski) {
        if (dim == Dim::_1D) {
          dx0_V0 = get_dx0_V0<Minkowski<Dim::_1D>>(res, ext, params);
        } else if (dim == Dim::_2D) {
          dx0_V0 = get_dx0_V0<Minkowski<Dim::_2D>>(res, ext, params);
        } else {
          dx0_V0 = get_dx0_V0<Minkowski<Dim::_3D>>(res, ext, params);
        }
      } else if (metric_enum == Metric::Spherical) {
        dx0_V0 = get_dx0_V0<Spherical<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::QSpherical) {
        dx0_V0 = get_dx0_V0<QSpherical<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::Kerr_Schild) {
        dx0_V0 = get_dx0_V0<KerrSchild<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::Kerr_Schild_0) {
        dx0_V0 = get_dx0_V0<KerrSchild0<Dim::_2D>>(res, ext, params);
      } else if (metric_enum == Metric::QKerr_Schild) {
        dx0_V0 = get_dx0_V0<QKerrSchild<Dim::_2D>>(res, ext, params);
      }
      auto [dx0, V0] = dx0_V0;
      set("scales.dx0", dx0);
      set("scales.V0", V0);
      set("scales.n0", ppc0 / V0);
      set("scales.q0", V0 / (ppc0 * SQR(skindepth0)));

      set("grid.metric.metric", metric_enum);
    }

    raise::ErrorIf(!promisesFulfilled(),
                   "Have not defined all the necessary variables",
                   HERE);
  }
} // namespace ntt
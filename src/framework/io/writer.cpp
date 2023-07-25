#include "writer.h"

#include "wrapper.h"

#include "sim_params.h"
#include "simulation.h"

#include "communications/metadomain.h"
#include "io/output.h"
#include "meshblock/meshblock.h"
#include "utils/utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <plog/Log.h>
#include <toml.hpp>

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <type_traits>

namespace ntt {

#ifdef OUTPUT_ENABLED
  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::Initialize(const SimulationParams& params,
                                const Metadomain<D>&    metadomain,
                                const Meshblock<D, S>&  mblock) {
    m_io = m_adios.DeclareIO("EntityOutput");
    m_io.SetEngine(params.outputFormat() != "disabled" ? params.outputFormat() : "HDF5");
    adios2::Dims shape, start, count;

    /* ---------------------------- Common attributes --------------------------- */
    auto         global_metric = metadomain.globalMetric();

    m_io.DefineVariable<int>("Step");
    m_io.DefineVariable<real_t>("Time");

    m_io.DefineAttribute("Metric", global_metric.label);
    m_io.DefineAttribute("Coordinates", params.coordinates());
    m_io.DefineAttribute("Engine", stringizeSimulationEngine(S));

    m_io.DefineAttribute("Dimension", (int)D);
    for (auto& block : params.inputdata().as_table()) {
      for (auto& attr : block.second.as_table()) {
        WriteTomlAttribute(block.first + "/" + attr.first, attr.second);
      }
    }

    /* ----------------------- Output grid as an attribute ---------------------- */
    // !TODO: this needs to change to output at every timestep
    WriteMeshGrid(metadomain);

    if constexpr (S == GRPICEngine) {
      m_io.DefineAttribute("Spin", global_metric.getParameter("spin"));
      m_io.DefineAttribute("Rhorizon", global_metric.getParameter("rhorizon"));
    }
    m_io.DefineAttribute("Timestep", mblock.timestep());

    /* -------------------- Determine field shapes & offsets -------------------- */
    auto local_domain = metadomain.localDomain();
    {
      const auto gh_zones = params.outputGhosts() ? N_GHOSTS : 0;
      m_io.DefineAttribute("NGhosts", gh_zones);
      for (short d = 0; d < (short)D; ++d) {
        shape.push_back(metadomain.globalNcells()[d]
                        + 2 * metadomain.globalNdomainsPerDim()[d] * gh_zones);
        start.push_back(local_domain->offsetNcells()[d]
                        + 2 * gh_zones * local_domain->offsetNdomains()[d]);
        count.push_back(local_domain->ncells()[d] + 2 * gh_zones);
      }
    }
    auto isLayoutRight
      = std::is_same<typename ndfield_t<D, 6>::array_layout, Kokkos::LayoutRight>::value;

    if (isLayoutRight) {
      m_io.DefineAttribute("LayoutRight", 1);
    } else {
      std::reverse(shape.begin(), shape.end());
      std::reverse(start.begin(), start.end());
      std::reverse(count.begin(), count.end());
      m_io.DefineAttribute("LayoutRight", 0);
    }

    /* ------------------ Determine field quantities to output ------------------ */
    for (auto& var : params.outputFields()) {
      m_fields.push_back(InterpretInputForFieldOutput(var));
      m_fields.back().initialize(S);
      m_fields.back().ghosts = params.outputGhosts();
    }
    for (auto& fld : m_fields) {
      for (std::size_t i { 0 }; i < fld.comp.size(); ++i) {
        m_io.DefineVariable<real_t>(fld.name(i), shape, start, count, adios2::ConstantDims);
      }
    }
    /* ---------------- Determine particle quantities to output ----------------- */
    for (std::size_t sp { 0 }; sp < mblock.particles.size(); ++sp) {
      m_io.DefineAttribute<std::string>("species-" + std::to_string(sp + 1),
                                        mblock.particles[sp].label());
    }
    for (auto& var : params.outputParticles()) {
      auto prtl_to_output = InterpretInputForParticleOutput(var);
      if (prtl_to_output.speciesID().size() == 0) {
        // if no species specified, pick all
        std::vector<int> species;
        for (std::size_t s { 0 }; s < params.species().size(); ++s) {
          species.push_back(s + 1);
        }
        prtl_to_output.setSpeciesID(species);
      }
      m_particles.push_back(prtl_to_output);
    }
    for (auto& prtl : m_particles) {
      for (auto& sp_index : prtl.speciesID()) {
        if (prtl.id() == PrtlID::X) {
          // !TODO: change this to a pre-defined argument (number of coords or smth)
#  ifndef MINKOWSKI_METRIC
          const auto dmax = (D == Dim2) ? 3 : (short)D;
#  else
          const auto dmax = (short)D;
#  endif
          for (auto d { 0 }; d < dmax; ++d) {
            m_io.DefineVariable<real_t>(
              "X" + std::to_string(d + 1) + "_" + std::to_string(sp_index),
              {},
              {},
              { adios2::UnknownDim });
          }
        } else if (prtl.id() == PrtlID::U) {
          for (auto d { 0 }; d < 3; ++d) {
            m_io.DefineVariable<real_t>(
              "U" + std::to_string(d + 1) + "_" + std::to_string(sp_index),
              {},
              {},
              { adios2::UnknownDim });
          }
        } else if (prtl.id() == PrtlID::W) {
          m_io.DefineVariable<real_t>(
            "W_" + std::to_string(sp_index), {}, {}, { adios2::UnknownDim });
        }
      }
    }
    m_writer = m_io.Open(params.title() + (params.outputFormat() == "HDF5" ? ".h5" : ".bp"),
                         adios2::Mode::Write);
    m_adios.EnterComputationBlock();
  }

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteMeshGrid(const Metadomain<D>& metadomain) {
    auto global_metric = metadomain.globalMetric();

    if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
      m_io.DefineAttribute("X1Min", global_metric.x1_min);
      m_io.DefineAttribute("X1Max", global_metric.x1_max);

      const auto Ni1 { metadomain.globalNcells()[0] };
      auto       x1 = new real_t[Ni1 + 1];
      for (std::size_t i { 0 }; i <= Ni1; ++i) {
        coord_t<D> xph { ZERO }, xi { ZERO };
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[0] = (real_t)(i);
        global_metric.x_Code2Phys(xi, xph);
        x1[i] = xph[0];
      }
      m_io.DefineAttribute("X1", x1, Ni1 + 1);
    }
    if constexpr (D == Dim2 || D == Dim3) {
      m_io.DefineAttribute("X2Min", global_metric.x2_min);
      m_io.DefineAttribute("X2Max", global_metric.x2_max);

      const auto Ni2 { metadomain.globalNcells()[1] };
      auto       x2 = new real_t[Ni2 + 1];
      for (std::size_t i { 0 }; i <= Ni2; ++i) {
        coord_t<D> xph { ZERO }, xi { ZERO };
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[1] = (real_t)(i);
        global_metric.x_Code2Phys(xi, xph);
        x2[i] = xph[1];
      }
      m_io.DefineAttribute("X2", x2, Ni2 + 1);
    }
    if constexpr (D == Dim3) {
      m_io.DefineAttribute("X3Min", global_metric.x3_min);
      m_io.DefineAttribute("X3Max", global_metric.x3_max);

      const auto Ni3 { metadomain.globalNcells()[2] };
      auto       x3 = new real_t[Ni3];
      for (std::size_t i { 0 }; i <= Ni3; ++i) {
        coord_t<D> xph { ZERO }, xi { ZERO };
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[2] = (real_t)(i);
        global_metric.x_Code2Phys(xi, xph);
        x3[i] = xph[2];
      }
      m_io.DefineAttribute("X3", x3, Ni3 + 1);
    }
  }

  namespace {
    template <typename T>
    auto unrollTomlVector(const toml::value& attr, bool simple) -> std::vector<T> {
      if (simple) {
        return toml::get<std::vector<T>>(attr);
      } else {
        auto val_vec     = std::vector<T> {};
        auto vec_of_vecs = toml::get<std::vector<std::vector<T>>>(attr);
        for (const auto& vec : vec_of_vecs) {
          val_vec.insert(val_vec.end(), vec.begin(), vec.end());
        }
        return val_vec;
      }
    }
  }    // namespace

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteTomlAttribute(const std::string& name, const toml::value& attr) {
    auto isSimpleType = [](const toml::value_t& type) {
      return (type == toml::value_t::integer || type == toml::value_t::floating
              || type == toml::value_t::string || type == toml::value_t::boolean);
    };
    switch (attr.type()) {
    case toml::value_t::integer:
      m_io.DefineAttribute(name, attr.as_integer());
      break;
    case toml::value_t::floating:
      m_io.DefineAttribute(name, attr.as_floating());
      break;
    case toml::value_t::string:
      m_io.DefineAttribute(name, (std::string)attr.as_string());
      break;
    case toml::value_t::boolean:
      m_io.DefineAttribute(name, (attr.as_boolean() ? 1 : 0));
      break;
    case toml::value_t::array: {
      auto is_simple = isSimpleType(attr.at(0).type());
      auto element   = is_simple ? attr.at(0) : attr.at(0).at(0);
      if (element.is_integer()) {
        auto attrs = unrollTomlVector<int>(attr, is_simple);
        m_io.DefineAttribute(name, attrs.data(), attrs.size());
      } else if (element.is_floating()) {
        auto attrs = unrollTomlVector<real_t>(attr, is_simple);
        m_io.DefineAttribute(name, attrs.data(), attrs.size());
      } else if (element.is_boolean()) {
        auto attrs = unrollTomlVector<int>(attr, is_simple);
        m_io.DefineAttribute(name, attrs.data(), attrs.size());
      } else if (element.is_string()) {
        auto attrs = unrollTomlVector<std::string>(attr, is_simple);
        m_io.DefineAttribute(name, attrs.data(), attrs.size());
      } else {
        NTTHostError("Unknown type of attribute: " + name);
      }
      break;
    }
    default:
      NTTHostError("Unknown type of attribute: " + name);
      break;
    }
  }

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteAll(const SimulationParams& params,
                              const Metadomain<D>&    metadomain,
                              Meshblock<D, S>&        mblock,
                              const real_t&           time,
                              const std::size_t&      tstep) {
    // check if output is enabled
    auto output_enabled = (params.outputFormat() != "disabled");
    // check if output is done by # of steps or by physical time
    auto output_by_step = (params.outputIntervalTime() <= 0.0);
    auto output_by_time = !output_by_step;
    // check if current timestep is an output step
    // based on # of steps or passed time since last output
    auto is_output_step = (tstep % params.outputInterval() == 0);
    auto is_output_time = (time - m_last_output_time >= params.outputIntervalTime())
                          || (m_last_output_time <= 0.0);
    // combine the logic
    auto do_output
      = (output_enabled
         && ((output_by_step && is_output_step) || (output_by_time && is_output_time)));

    if (do_output) {
      m_adios.ExitComputationBlock();
      WaitAndSynchronize();
      NTTLog();
      m_writer.BeginStep();
      int step = (int)tstep;

      m_writer.Put(m_io.InquireVariable<int>("Step"), &step);
      m_writer.Put(m_io.InquireVariable<real_t>("Time"), &time);

      WriteFields(params, mblock, time, tstep);
      WriteParticles(params, metadomain, mblock, time, tstep);

      m_writer.EndStep();
      m_last_output_time = time;
      m_adios.EnterComputationBlock();
    }
  }

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams& params,
                                 Meshblock<D, S>&        mblock,
                                 const real_t&,
                                 const std::size_t&) {
    // traverse all the fields and put them. ...
    // ... also make sure that the fields are ready for output, ...
    // ... i.e. they have been written into proper arrays
    for (auto& fld : m_fields) {
      fld.compute<D, S>(params, mblock);
      fld.put<D, S>(m_io, m_writer, mblock);
    }
  }

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteParticles(const SimulationParams& params,
                                    const Metadomain<D>&    metadomain,
                                    Meshblock<D, S>&        mblock,
                                    const real_t&,
                                    const std::size_t&) {
    // traverse all the particle quantities and put them.
    for (auto& prtl : m_particles) {
      prtl.put<D, S>(m_io, m_writer, params, metadomain, mblock);
    }
  }

#else
  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::Initialize(const SimulationParams&,
                                const Metadomain<D>&,
                                const Meshblock<D, S>&) {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteAll(const SimulationParams&,
                              const Metadomain<D>&,
                              Meshblock<D, S>&,
                              const real_t&,
                              const std::size_t&) {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams&,
                                 Meshblock<D, S>&,
                                 const real_t&,
                                 const std::size_t&) {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteParticles(const SimulationParams&,
                                    const Metadomain<D>&,
                                    Meshblock<D, S>&,
                                    const real_t&,
                                    const std::size_t&) {}

#endif

}    // namespace ntt

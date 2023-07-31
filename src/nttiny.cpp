#include "wrapper.h"

#include "io/cargs.h"
#include "io/output.h"

#include <nttiny/api.h>
#include <nttiny/tools.h>
#include <nttiny/vis.h>

#if defined(PIC_ENGINE)

#  include "pic.h"
template <ntt::Dimension D>
using SimEngine = ntt::PIC<D>;

#elif defined(GRPIC_ENGINE)

#  include "grpic.h"
template <ntt::Dimension D>
using SimEngine = ntt::GRPIC<D>;

#endif

#include <toml.hpp>

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template <ntt::SimulationEngine S>
class NTTSimulationVis : public nttiny::SimulationAPI<real_t, 2> {
protected:
  ntt::SIMULATION_CONTAINER<ntt::Dim2>& m_sim;
  int                                   sx1, sx2;
  const std::vector<ntt::OutputField>   m_fields_to_plot;
  const int                             m_fields_stride;

public:
  NTTSimulationVis(ntt::SIMULATION_CONTAINER<ntt::Dim2>& sim,
                   const std::vector<ntt::OutputField>&       fields_to_plot,
                   const int&                            fields_stride)
    : nttiny::SimulationAPI<real_t, 2> {sim.params()->title(),
                                        sim.meshblock.metric.label == "minkowski"
                                          ? nttiny::Coord::Cartesian
                                          : nttiny::Coord::Spherical,
                                        {(int)sim.meshblock.Ni1() / fields_stride,
                                         (int)sim.meshblock.Ni2() / fields_stride},
                                        N_GHOSTS},
      m_sim(sim),
      sx1 {(int)sim.meshblock.Ni1() / fields_stride},
      sx2 {(int)sim.meshblock.Ni2() / fields_stride},
      m_fields_to_plot{fields_to_plot},
      m_fields_stride{fields_stride} {
    m_time       = 0.0;
    m_timestep   = 0;
    auto& mblock = m_sim.meshblock;
    auto  params = *(m_sim.params());

    generateFields();
    generateGrid();
    generateParticles();

    m_sim.problem_generator.UserInitBuffers_nttiny(params, mblock, this->buffers);
    setData();
  }

  void setData() override {
    auto&      Grid           = this->m_global_grid;
    auto&      Fields         = this->fields;
    const auto nx1            = (std::size_t)(Grid.m_size[0] + Grid.m_ngh * 2);
    const auto nx2            = (std::size_t)(Grid.m_size[1] + Grid.m_ngh * 2);
    const auto ngh            = Grid.m_ngh;
    const auto nfields        = m_fields_to_plot.size();
    const auto fields_to_plot = m_fields_to_plot;
    const auto fields_stride  = m_fields_stride;

    auto&      mblock         = m_sim.meshblock;
    auto       params         = *(m_sim.params());

    // precompute necessary fields
    // !FIX: this has a potential of overwriting the buffer and backup fields
    for (auto& f : m_fields_to_plot) {
      f.template compute<ntt::Dim2, S>(params, mblock);
    }

    auto buffer_h = Kokkos::create_mirror_view(mblock.buff);
    Kokkos::deep_copy(buffer_h, mblock.buff);
    auto backup_h = Kokkos::create_mirror_view(mblock.bckp);
    Kokkos::deep_copy(backup_h, mblock.bckp);

    Kokkos::parallel_for(
      "setData-Fields",
      ntt::CreateRangePolicyOnHost<ntt::Dim2>({ (std::size_t)0, (std::size_t)0 },
                                              { (std::size_t)nx1, (std::size_t)nx2 }),
      [=](std::size_t i1, std::size_t j1) {
        int i, j;
        if (i1 < ngh) {
          i = i1;
        } else if (i1 >= nx1 - ngh) {
          i = (nx1 - 2 * ngh) * fields_stride + 2 * ngh + (int)i1 - nx1;
        } else {
          i = ((int)i1 - ngh) * fields_stride + ngh;
        }
        if (j1 < ngh) {
          j = j1;
        } else if (j1 >= nx2 - ngh) {
          j = (nx2 - 2 * ngh) * fields_stride + 2 * ngh + (int)j1 - nx2;
        } else {
          j = ((int)j1 - ngh) * fields_stride + ngh;
        }
        auto idx = Index((int)i1 - ngh, (int)j1 - ngh);

        for (std::size_t fi { 0 }; fi < nfields; ++fi) {
          auto f = fields_to_plot[fi];
          for (std::size_t c { 0 }; c < f.comp.size(); ++c) {
            auto   fld = f.name(c);

            real_t val { ZERO };
            if (f.is_field() || f.is_gr_aux_field() || f.is_vpotential()) {
              val = backup_h(i, j, f.address[c]);
            } else if (f.is_current() || f.is_moment()) {
              val = buffer_h(i, j, f.address[c]);
            }
            Fields.at(fld)[idx] = math::isfinite(val) ? val : -100000.0;
          }
        }
      });

    auto  s         = 0;
    auto& Particles = this->particles;
    for (const auto& [lbl, species] : Particles) {
      auto sim_species = m_sim.meshblock.particles[s];
      auto nprtl       = species.first;
      auto slice       = std::make_pair(0, nprtl);
      auto i1_h        = Kokkos::create_mirror_view(Kokkos::subview(sim_species.i1, slice));
      Kokkos::deep_copy(i1_h, Kokkos::subview(sim_species.i1, slice));
      auto i2_h = Kokkos::create_mirror_view(Kokkos::subview(sim_species.i2, slice));
      Kokkos::deep_copy(i2_h, Kokkos::subview(sim_species.i2, slice));
      auto dx1_h = Kokkos::create_mirror_view(Kokkos::subview(sim_species.dx1, slice));
      Kokkos::deep_copy(dx1_h, Kokkos::subview(sim_species.dx1, slice));
      auto dx2_h = Kokkos::create_mirror_view(Kokkos::subview(sim_species.dx2, slice));
      Kokkos::deep_copy(dx2_h, Kokkos::subview(sim_species.dx2, slice));

      for (auto p { 0 }; p < nprtl; ++p) {
        real_t                  x1 { (real_t)(i1_h(p)) + dx1_h(p) };
        real_t                  x2 { (real_t)(i2_h(p)) + dx2_h(p) };
        ntt::coord_t<ntt::Dim2> xy { ZERO, ZERO };
        m_sim.meshblock.metric.x_Code2Cart({ x1, x2 }, xy);
        species.second[0][p] = xy[0];
        species.second[1][p] = xy[1];
      }
      ++s;
    }
    m_sim.problem_generator.UserSetBuffers_nttiny(m_time, params, mblock, this->buffers);
  }

  void stepFwd() override {
    m_sim.StepForward(ntt::DiagFlags_Timers | ntt::DiagFlags_Species);
    m_timestep = m_sim.tstep();
    m_time     = m_sim.time();
  }

  void restart() override {
    m_time     = 0.0;
    m_timestep = 0;
    m_sim.ResetSimulation();
    m_sim.InitialStep();
    setData();
  }

  void stepBwd() override {}

  void generateFields() {
    auto&      Fields = this->fields;
    auto&      Grid   = this->m_global_grid;
    const auto nx1 { Grid.m_size[0] + Grid.m_ngh * 2 };
    const auto nx2 { Grid.m_size[1] + Grid.m_ngh * 2 };
    for (std::size_t i { 0 }; i < m_fields_to_plot.size(); ++i) {
      for (std::size_t c { 0 }; c < m_fields_to_plot[i].comp.size(); ++c) {
        Fields.insert({ m_fields_to_plot[i].name(c), new real_t[nx1 * nx2] });
      }
    }
  }

  void generateParticles() {
    auto  params    = *(m_sim.params());
    auto& Particles = this->particles;
    int   s         = 0;
    for (auto& species : m_sim.meshblock.particles) {
      auto nprtl { m_sim.meshblock.particles[s].npart() / params.outputPrtlStride() };
      nprtl = nprtl > 0 ? nprtl : 1;
      Particles.insert({
        species.label(), {nprtl, { new real_t[nprtl], new real_t[nprtl] }}
      });
      ++s;
    }
  }

  void generateGrid() {
    auto& Grid = this->m_global_grid;
    if (Grid.m_coord == nttiny::Coord::Spherical) {
      const auto sx1 { Grid.m_size[0] };
      const auto sx2 { Grid.m_size[1] };
      for (int i { 0 }; i <= sx1; ++i) {
        auto                    i_ { (real_t)(i * m_fields_stride) };
        auto                    j_ { (real_t)(sx2 * m_fields_stride) * HALF };
        ntt::coord_t<ntt::Dim2> rth_;
        m_sim.meshblock.metric.x_Code2Sph({ i_, j_ }, rth_);
        Grid.m_xi[0][i] = rth_[0];
      }
      for (int j { 0 }; j <= sx2; ++j) {
        auto                    i_ { (real_t)(sx1 * m_fields_stride) * HALF };
        auto                    j_ { (real_t)(j * m_fields_stride) };
        ntt::coord_t<ntt::Dim2> rth_;
        m_sim.meshblock.metric.x_Code2Sph({ i_, j_ }, rth_);
        Grid.m_xi[1][j] = rth_[1];
      }
      Grid.ExtendGridWithGhosts();
    } else {
      const auto s1 { m_sim.meshblock.metric.x1_max - m_sim.meshblock.metric.x1_min };
      const auto s2 { m_sim.meshblock.metric.x2_max - m_sim.meshblock.metric.x2_min };
      const auto sx1 { Grid.m_size[0] };
      const auto sx2 { Grid.m_size[1] };
      for (int i { 0 }; i <= sx1; ++i) {
        Grid.m_xi[0][i] = m_sim.meshblock.metric.x1_min + s1 * (real_t)(i) / (real_t)(sx1);
      }
      for (int j { 0 }; j <= sx2; ++j) {
        Grid.m_xi[1][j] = m_sim.meshblock.metric.x2_min + s2 * (real_t)(j) / (real_t)(sx2);
      }
    }
  }

  void customAnnotatePcolor2d(const nttiny::UISettings& ui_settings) override {
#ifdef GRPIC_ENGINE
    auto&  mblock   = m_sim.meshblock;
    auto   params   = *(m_sim.params());
    real_t r_absorb = params.metricParameters()[2];
    real_t rh       = mblock.metric.rhorizon();
    nttiny::tools::drawCircle(
      { 0.0f, 0.0f }, rh, { 0.0f, ntt::constant::PI }, 128, ui_settings.OutlineColor);
    nttiny::tools::drawCircle(
      { 0.0f, 0.0f }, r_absorb, { 0.0f, ntt::constant::PI }, 128, ui_settings.OutlineColor);
#else
    auto& Grid = this->m_global_grid;
    if (Grid.m_coord == nttiny::Coord::Spherical) {
      auto   params   = *(m_sim.params());
      real_t r_absorb = params.metricParameters()[2];
      nttiny::tools::drawCircle(
        { 0.0f, 0.0f }, r_absorb, { 0.0f, ntt::constant::PI }, 128, ui_settings.OutlineColor);
    }
#endif
  }
};

auto main(int argc, char* argv[]) -> int {
  ntt::GlobalInitialize(argc, argv);
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto inputfilename = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto inputdata     = toml::parse(static_cast<std::string>(inputfilename));
    auto sim_title     = ntt::readFromInput<std::string>(
      inputdata, "simulation", "title", ntt::defaults::title);

    auto scale_str          = cl_args.getArgument("-scale", "1.0");
    auto scale              = std::stof(std::string(scale_str));

    auto fields_to_plot_str = ntt::readFromInput<std::vector<std::string>>(
      inputdata, "output", "fields", std::vector<std::string>());
    auto fields_stride = ntt::readFromInput<int>(inputdata, "output", "fields_stride", 1);

    std::vector<ntt::OutputField> fields_to_plot;
    for (auto& fld : fields_to_plot_str) {
      fields_to_plot.push_back(ntt::InterpretInputForFieldOutput(fld));
      fields_to_plot.back().initialize(simulation_engine);
    }

    SimEngine<ntt::Dim2> sim(inputdata);

    sim.Verify();
    sim.ResetSimulation();
    sim.InitialStep();
    sim.PrintDetails();
    NTTSimulationVis<sim.engine>     visApi(sim, fields_to_plot, fields_stride);

    nttiny::Visualization<real_t, 2> vis { scale };
    vis.bindSimulation(&visApi);
    vis.loop();

    sim.Finalize();

  } catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    ntt::GlobalFinalize();

    return -1;
  }
  ntt::GlobalFinalize();

  return 0;
}

#ifdef PIC_ENGINE
template class NTTSimulationVis<ntt::PICEngine>;
#else
template class NTTSimulationVis<ntt::GRPICEngine>;
#endif

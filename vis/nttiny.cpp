#include "nttiny/vis.h"
#include "nttiny/api.h"

#include "global.h"
#include "cargs.h"
#include "input.h"

#if SIMTYPE == PIC_SIMTYPE
#  include "pic.h"
#  define SIMULATION_CONTAINER PIC
#elif SIMTYPE == GRPIC_SIMTYPE
#  include "grpic.h"
#  include "init_fields.hpp"
#  define SIMULATION_CONTAINER GRPIC
#endif

#include <toml/toml.hpp>
#include <plog/Log.h>
#include <plog/Init.h>
#include <plog/Appenders/ColorConsoleAppender.h>

#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

using plog_t = plog::ColorConsoleAppender<plog::NTTFormatter>;
void initLogger(plog_t* console_appender);

struct NTTSimulationVis : public nttiny::SimulationAPI<float> {
  int                                               nx1, nx2;
  ntt::SIMULATION_CONTAINER<ntt::Dimension::TWO_D>& m_sim;
  std::vector<nttiny::Data<float>>                  m_data;
  std::vector<std::unique_ptr<nttiny::Data<float>>> prtl_pointers;
  std::vector<std::string>                          m_fields_to_plot;

  NTTSimulationVis(ntt::SIMULATION_CONTAINER<ntt::Dimension::TWO_D>& sim,
                   const std::vector<std::string>&                   fields_to_plot)
/**
 * TODO: make this less ugly
 */
#if SIMTYPE == PIC_SIMTYPE
    : nttiny::SimulationAPI<float> {sim.mblock().metric.label},
#elif SIMTYPE == GRPIC_SIMTYPE
    : nttiny::SimulationAPI<float> {"qspherical"},
#endif
      nx1(sim.mblock().Ni() + 2 * ntt::N_GHOSTS),
      nx2(sim.mblock().Nj() + 2 * ntt::N_GHOSTS),
      m_sim(sim),
      m_fields_to_plot(fields_to_plot) {
    this->m_timestep = 0;
    this->m_time     = 0.0;
    generateFields();
    generateGrid();
    generateParticles();
    setData();
  }

  void setData() override {
#if SIMTYPE == GRPIC_SIMTYPE
    m_sim.computeVectorPotential();
    // compute the vector potential
#endif

    for (int j {0}; j < nx2; ++j) {
      for (int i {0}; i < nx1; ++i) {
        for (std::size_t f {0}; f < m_fields_to_plot.size(); ++f) {
#if SIMTYPE == PIC_SIMTYPE
          auto                                i_ {(real_t)(i - ntt::N_GHOSTS)};
          auto                                j_ {(real_t)(j - ntt::N_GHOSTS)};
          ntt::vec_t<ntt::Dimension::THREE_D> e_hat, b_hat;
          m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF},
                                            {m_sim.mblock().em(i, j, ntt::em::ex1),
                                             m_sim.mblock().em(i, j, ntt::em::ex2),
                                             m_sim.mblock().em(i, j, ntt::em::ex3)},
                                            e_hat);
          m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF},
                                            {m_sim.mblock().em(i, j, ntt::em::bx1),
                                             m_sim.mblock().em(i, j, ntt::em::bx2),
                                             m_sim.mblock().em(i, j, ntt::em::bx3)},
                                            b_hat);
          if (m_fields_to_plot[f] == "Er" || m_fields_to_plot[f] == "Ex") {
            m_data[f].set(i, j, e_hat[0]);
          } else if (m_fields_to_plot[f] == "Etheta" || m_fields_to_plot[f] == "Ey") {
            m_data[f].set(i, j, e_hat[1]);
          } else if (m_fields_to_plot[f] == "Ephi" || m_fields_to_plot[f] == "Ez") {
            m_data[f].set(i, j, e_hat[2]);
          } else if (m_fields_to_plot[f] == "Br" || m_fields_to_plot[f] == "Bx") {
            m_data[f].set(i, j, b_hat[0]);
          } else if (m_fields_to_plot[f] == "Btheta" || m_fields_to_plot[f] == "By") {
            m_data[f].set(i, j, b_hat[1]);
          } else if (m_fields_to_plot[f] == "Bphi" || m_fields_to_plot[f] == "Bz") {
            m_data[f].set(i, j, b_hat[2]);
          }
#elif SIMTYPE == GRPIC_SIMTYPE
          auto i_ {(real_t)(i - ntt::N_GHOSTS)};
          auto j_ {(real_t)(j - ntt::N_GHOSTS)};
          // interpolate and transform to spherical
          ntt::vec_t<ntt::Dimension::THREE_D> Dsph {0, 0, 0}, Bsph {0, 0, 0}, D0sph {0, 0, 0}, B0sph {0, 0, 0};
          if ((i < ntt::N_GHOSTS) || (i >= nx1 - ntt::N_GHOSTS) || (j < ntt::N_GHOSTS) || (j >= nx2 - ntt::N_GHOSTS)) {
            Dsph[0]  = m_sim.mblock().em(i, j, ntt::em::ex1);
            Dsph[1]  = m_sim.mblock().em(i, j, ntt::em::ex2);
            Dsph[2]  = m_sim.mblock().em(i, j, ntt::em::ex3);
            Bsph[0]  = m_sim.mblock().em(i, j, ntt::em::bx1);
            Bsph[1]  = m_sim.mblock().em(i, j, ntt::em::bx2);
            Bsph[2]  = m_sim.mblock().em(i, j, ntt::em::bx3);
            D0sph[0] = m_sim.mblock().em0(i, j, ntt::em::ex1);
            D0sph[1] = m_sim.mblock().em0(i, j, ntt::em::ex2);
            D0sph[2] = m_sim.mblock().em0(i, j, ntt::em::ex3);
            B0sph[0] = m_sim.mblock().em0(i, j, ntt::em::bx1);
            B0sph[1] = m_sim.mblock().em0(i, j, ntt::em::bx2);
            B0sph[2] = m_sim.mblock().em0(i, j, ntt::em::bx3);
          } else {
            if ((m_fields_to_plot[f] == "Dr") || (m_fields_to_plot[f] == "Dtheta") || (m_fields_to_plot[f] == "Dphi")) {
              real_t Dx1, Dx2, Dx3;
              // interpolate to cell center
              Dx1 = 0.5 * (m_sim.mblock().em(i, j, ntt::em::ex1) + m_sim.mblock().em(i, j + 1, ntt::em::ex1));
              Dx2 = 0.5 * (m_sim.mblock().em(i, j, ntt::em::ex2) + m_sim.mblock().em(i + 1, j, ntt::em::ex2));
              Dx3 = 0.25
                    * (m_sim.mblock().em(i, j, ntt::em::ex3) + m_sim.mblock().em(i + 1, j, ntt::em::ex3)
                       + m_sim.mblock().em(i, j + 1, ntt::em::ex3) + m_sim.mblock().em(i + 1, j + 1, ntt::em::ex3));
              m_sim.mblock().metric.v_Cntr2SphCntrv({i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, Dsph);
            }
            if ((m_fields_to_plot[f] == "Br") || (m_fields_to_plot[f] == "Btheta") || (m_fields_to_plot[f] == "Bphi")) {
              real_t Bx1, Bx2, Bx3;
              // interpolate to cell center
              Bx1 = 0.5 * (m_sim.mblock().em(i + 1, j, ntt::em::bx1) + m_sim.mblock().em(i, j, ntt::em::bx1));
              Bx2 = 0.5 * (m_sim.mblock().em(i, j + 1, ntt::em::bx2) + m_sim.mblock().em(i, j, ntt::em::bx2));
              Bx3 = m_sim.mblock().em(i, j, ntt::em::bx3);
              m_sim.mblock().metric.v_Cntr2SphCntrv({i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, Bsph);
            }
            if ((m_fields_to_plot[f] == "D0r") || (m_fields_to_plot[f] == "D0theta")
                || (m_fields_to_plot[f] == "D0phi")) {
              real_t Dx1, Dx2, Dx3;
              // interpolate to cell center
              Dx1 = 0.5 * (m_sim.mblock().em0(i, j, ntt::em::ex1) + m_sim.mblock().em0(i, j + 1, ntt::em::ex1));
              Dx2 = 0.5 * (m_sim.mblock().em0(i, j, ntt::em::ex2) + m_sim.mblock().em0(i + 1, j, ntt::em::ex2));
              Dx3 = 0.25
                    * (m_sim.mblock().em0(i, j, ntt::em::ex3) + m_sim.mblock().em0(i + 1, j, ntt::em::ex3)
                       + m_sim.mblock().em0(i, j + 1, ntt::em::ex3) + m_sim.mblock().em0(i + 1, j + 1, ntt::em::ex3));
              m_sim.mblock().metric.v_Cntr2SphCntrv({i_ + HALF, j_ + HALF}, {Dx1, Dx2, Dx3}, D0sph);
            }
            if ((m_fields_to_plot[f] == "B0r") || (m_fields_to_plot[f] == "B0theta")
                || (m_fields_to_plot[f] == "B0phi")) {
              real_t Bx1, Bx2, Bx3;
              // interpolate to cell center
              Bx1 = 0.5 * (m_sim.mblock().em0(i + 1, j, ntt::em::bx1) + m_sim.mblock().em0(i, j, ntt::em::bx1));
              Bx2 = 0.5 * (m_sim.mblock().em0(i, j + 1, ntt::em::bx2) + m_sim.mblock().em0(i, j, ntt::em::bx2));
              Bx3 = m_sim.mblock().em0(i, j, ntt::em::bx3);
              m_sim.mblock().metric.v_Cntr2SphCntrv({i_ + HALF, j_ + HALF}, {Bx1, Bx2, Bx3}, B0sph);
            }
          }

          if (m_fields_to_plot[f] == "Dr") {
            m_data[f].set(i, j, Dsph[0]);
          } else if (m_fields_to_plot[f] == "Dtheta") {
            m_data[f].set(i, j, Dsph[1]);
          } else if (m_fields_to_plot[f] == "Dphi") {
            m_data[f].set(i, j, Dsph[2]);
          } else if (m_fields_to_plot[f] == "Br") {
            m_data[f].set(i, j, Bsph[0]);
          } else if (m_fields_to_plot[f] == "Btheta") {
            m_data[f].set(i, j, Bsph[1]);
          } else if (m_fields_to_plot[f] == "Bphi") {
            m_data[f].set(i, j, Bsph[2]);
          } else if (m_fields_to_plot[f] == "Er") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::ex1));
          } else if (m_fields_to_plot[f] == "Etheta") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::ex2));
          } else if (m_fields_to_plot[f] == "Ephi") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::ex3));
          } else if (m_fields_to_plot[f] == "Hr") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::bx1));
          } else if (m_fields_to_plot[f] == "Htheta") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::bx2));
          } else if (m_fields_to_plot[f] == "Hphi") {
            m_data[f].set(i, j, m_sim.mblock().aux(i, j, ntt::em::bx3));
          } else if (m_fields_to_plot[f] == "D0r") {
            m_data[f].set(i, j, D0sph[0]);
          } else if (m_fields_to_plot[f] == "D0theta") {
            m_data[f].set(i, j, D0sph[1]);
          } else if (m_fields_to_plot[f] == "D0phi") {
            m_data[f].set(i, j, D0sph[2]);
          } else if (m_fields_to_plot[f] == "B0r") {
            m_data[f].set(i, j, B0sph[0]);
          } else if (m_fields_to_plot[f] == "B0theta") {
            m_data[f].set(i, j, B0sph[1]);
          } else if (m_fields_to_plot[f] == "B0phi") {
            m_data[f].set(i, j, B0sph[2]);
          } else if (m_fields_to_plot[f] == "Aphi") {
            m_data[f].set(i, j, m_sim.mblock().aphi(i, j, 0));
          }
#endif
        }
      }
    }
    // int i {0};
    // for (auto& species : m_sim.mblock().particles) {
    // for (int k {0}; k < this->prtl_pointers[i]->get_size(0); ++k) {
    // float                               x1 {(float)(species.i1(k)) + species.dx1(k)};
    // float                               x2 {(float)(species.i2(k)) + species.dx2(k)};
    // ntt::coord_t<ntt::Dimension::TWO_D> xy {ZERO, ZERO};
    // m_sim.mblock().metric.x_Code2Cart({x1, x2}, xy);
    // this->prtl_pointers[i]->set(k, 0, xy[0]);
    // this->prtl_pointers[i + 1]->set(k, 0, xy[1]);
    //}
    // i += 2;
    //}
  }
  void stepFwd() override {
    for (int i {0}; i < this->get_jumpover(); ++i) {
      m_sim.step_forward(m_time);
      ++m_timestep;
      m_time += m_sim.mblock().timestep();
    }
    setData();
  }
  void restart() override {
    m_sim.initializeSetup();
    m_sim.initial_step(ZERO);
    setData();
    m_time     = 0.0;
    m_timestep = 0;
  }
  void stepBwd() override {
    // m_sim.step_backward(m_time);
    // setData();
    // --m_timestep;
    // m_time -= m_sim.mblock().timestep();
  }

  void generateFields() {
    for (std::size_t i {0}; i < m_fields_to_plot.size(); ++i) {
      m_data.push_back(std::move(nttiny::Data<float> {nx1, nx2}));
    }
    for (std::size_t i {0}; i < m_fields_to_plot.size(); ++i) {
      this->fields.insert({{m_fields_to_plot[i], &(this->m_data[i])}});
    }
  }

  void generateParticles() {
    // int s {0}, i {0};
    // for (auto& species : m_sim.mblock().particles) {
    //   auto nprt {m_sim.mblock().particles[s].npart()};
    //   auto x_prtl {std::make_unique<nttiny::Data<float>>(nprt, 1)};
    //   auto y_prtl {std::make_unique<nttiny::Data<float>>(nprt, 1)};
    //   this->prtl_pointers.push_back(std::move(x_prtl));
    //   this->prtl_pointers.push_back(std::move(y_prtl));
    //   this->particles.insert({species.label(), {(this->prtl_pointers[i].get()), (this->prtl_pointers[i +
    //   1].get())}}); s++; i += 2;
    // }
  }

  void generateGrid() {
    auto& field_data_0 = m_data[0];
    if (this->coords == "qspherical") {
      m_x1x2_extent[0] = m_sim.mblock().metric.x1_min;
      m_x1x2_extent[1] = m_sim.mblock().metric.x1_max;
      for (int i {ntt::N_GHOSTS}; i <= nx1 - ntt::N_GHOSTS; ++i) {
        auto                                i_ {(real_t)(i - ntt::N_GHOSTS)};
        auto                                j_ {ZERO};
        ntt::coord_t<ntt::Dimension::TWO_D> rth_;
        m_sim.mblock().metric.x_Code2Sph({i_, j_}, rth_);
        field_data_0.grid_x1[i] = rth_[0];
      }
      for (int i {ntt::N_GHOSTS - 1}; i >= 0; --i) {
        field_data_0.grid_x1[i] = field_data_0.grid_x1[i + 1]
                                  - (field_data_0.grid_x1[ntt::N_GHOSTS + 1] - field_data_0.grid_x1[ntt::N_GHOSTS]);
      }
      for (int i {nx1 - ntt::N_GHOSTS + 1}; i <= nx1; ++i) {
        field_data_0.grid_x1[i]
          = field_data_0.grid_x1[i - 1]
            + (field_data_0.grid_x1[nx1 - ntt::N_GHOSTS] - field_data_0.grid_x1[nx1 - ntt::N_GHOSTS - 1]);
      }
      for (int j {0}; j <= nx2; ++j) {
        auto                                i_ {ZERO};
        auto                                j_ {(real_t)(j - ntt::N_GHOSTS)};
        ntt::coord_t<ntt::Dimension::TWO_D> rth_;
        m_sim.mblock().metric.x_Code2Sph({i_, j_}, rth_);
        field_data_0.grid_x2[j] = rth_[1];
      }
      ntt::coord_t<ntt::Dimension::TWO_D> rth1_, rth2_;
      m_sim.mblock().metric.x_Code2Sph({ZERO, (real_t)(-ntt::N_GHOSTS)}, rth1_);
      m_sim.mblock().metric.x_Code2Sph({ZERO, (real_t)(nx2 - ntt::N_GHOSTS)}, rth2_);
      m_x1x2_extent[2] = rth1_[1];
      m_x1x2_extent[3] = rth2_[1];
    } else {
      auto sx1 {m_sim.mblock().metric.x1_max - m_sim.mblock().metric.x1_min};
      auto dx1 {sx1 / m_sim.mblock().metric.nx1};
      auto sx2 {m_sim.mblock().metric.x2_max - m_sim.mblock().metric.x2_min};
      auto dx2 {sx2 / m_sim.mblock().metric.nx2};
      m_x1x2_extent[0] = m_sim.mblock().metric.x1_min - dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[1] = m_sim.mblock().metric.x1_max + dx1 * ntt::N_GHOSTS;
      m_x1x2_extent[2] = m_sim.mblock().metric.x2_min - dx2 * ntt::N_GHOSTS;
      m_x1x2_extent[3] = m_sim.mblock().metric.x2_max + dx2 * ntt::N_GHOSTS;
      for (int i {0}; i <= nx1; ++i) {
        field_data_0.grid_x1[i]
          = m_x1x2_extent[0] + (m_x1x2_extent[1] - m_x1x2_extent[0]) * (double)(i) / (double)(nx1);
      }
      for (int j {0}; j <= nx2; ++j) {
        field_data_0.grid_x2[j]
          = m_x1x2_extent[2] + (m_x1x2_extent[3] - m_x1x2_extent[2]) * (double)(j) / (double)(nx2);
      }
    }
    for (auto const& [fld, arr] : this->fields) {
      for (int i {0}; i <= nx1; ++i) {
        arr->grid_x1[i] = field_data_0.grid_x1[i];
      }
      for (int i {0}; i <= nx2; ++i) {
        arr->grid_x2[i] = field_data_0.grid_x2[i];
      }
    }
  }

  void customAnnotatePcolor2d() override {
#if SIMTYPE == GRPIC_SIMTYPE
    float a        = m_sim.sim_params().metric_parameters()[4];
    float r_absorb = m_sim.sim_params().metric_parameters()[2];
    float rh       = 1.0f + math::sqrt(1.0f - a * a);
    nttiny::drawCircle({0.0f, 0.0f}, rh, {0.0f, ntt::constant::PI});
    nttiny::drawCircle({0.0f, 0.0f}, r_absorb, {0.0f, ntt::constant::PI});
#elif SIMTYPE == PIC_SIMTYPE
#endif
  }
};

auto main(int argc, char* argv[]) -> int {
  plog_t console_appender;
  initLogger(&console_appender);

  Kokkos::initialize();
  try {
    ntt::CommandLineArguments cl_args;
    cl_args.readCommandLineArguments(argc, argv);
    auto                     inputfilename  = cl_args.getArgument("-input", ntt::defaults::input_filename);
    auto                     inputdata      = toml::parse(static_cast<std::string>(inputfilename));
    auto&                    vis_data       = toml::find(inputdata, "visualization");
    std::vector<std::string> fields_to_plot = toml::find<std::vector<std::string>>(vis_data, "fields");

    ntt::SIMULATION_CONTAINER<ntt::Dimension::TWO_D> sim(inputdata);
    sim.initialize();
    sim.initializeSetup();
    sim.verify();
    sim.printDetails();
    sim.initial_step(ZERO);
    NTTSimulationVis visApi(sim, fields_to_plot);

    nttiny::Visualization<float> vis;
    vis.setTPSLimit(140.0f);
    vis.bindSimulation(&visApi);
    vis.loop();
  }
  catch (std::exception& err) {
    std::cerr << err.what() << std::endl;
    Kokkos::finalize();

    return -1;
  }
  Kokkos::finalize();

  return 0;
}

void initLogger(plog_t* console_appender) {
  plog::Severity max_severity;
#ifdef DEBUG
  max_severity = plog::verbose;
#else
  max_severity = plog::info;
#endif
  plog::init(max_severity, console_appender);
}

// LEGACY CODE:

// // auto i_ {(real_t)(i - ntt::N_GHOSTS)};
// // auto j_ {(real_t)(j - ntt::N_GHOSTS)};
// real_t dx1_cnt, dx2_cnt, dx3_cnt;
// real_t bx1_cnt, bx2_cnt, bx3_cnt;
// real_t ex1_cnt, ex2_cnt, ex3_cnt;
// real_t hx1_cnt, hx2_cnt, hx3_cnt;

// // if ((i < ntt::N_GHOSTS) || (j < ntt::N_GHOSTS) || (i >= nx2 - ntt::N_GHOSTS) || (j >= nx1 - ntt::N_GHOSTS)) {
// //   ex1_cnt = m_sim.mblock().aux(i, j, ntt::em::ex1);
// //   ex2_cnt = m_sim.mblock().aux(i, j, ntt::em::ex2);
// //   ex3_cnt = m_sim.mblock().aux(i, j, ntt::em::ex3);
// //   bx1_cnt = m_sim.mblock().em0(i, j, ntt::em::bx1);
// //   bx2_cnt = m_sim.mblock().em0(i, j, ntt::em::bx2);
// //   bx3_cnt = m_sim.mblock().em0(i, j, ntt::em::bx3);
// // } else {
// //   ex1_cnt = 0.5 * (m_sim.mblock().aux(i, j, ntt::em::ex1) + m_sim.mblock().aux(i, j + 1, ntt::em::ex1));
// //   ex2_cnt = 0.5 * (m_sim.mblock().aux(i, j, ntt::em::ex2) + m_sim.mblock().aux(i + 1, j, ntt::em::ex2));
// //   ex3_cnt = 0.25
// //             * (m_sim.mblock().aux(i, j, ntt::em::ex3) + m_sim.mblock().aux(i + 1, j, ntt::em::ex3)
// //                + m_sim.mblock().aux(i, j + 1, ntt::em::ex3) + m_sim.mblock().aux(i + 1, j + 1,
// //                ntt::em::ex3));
// //   bx1_cnt = 0.5 * (m_sim.mblock().em0(i, j, ntt::em::bx1) + m_sim.mblock().em0(i + 1, j, ntt::em::bx1));
// //   bx2_cnt = 0.5 * (m_sim.mblock().em0(i, j, ntt::em::bx2) + m_sim.mblock().em0(i, j + 1, ntt::em::bx2));
// //   bx3_cnt = m_sim.mblock().em0(i, j, ntt::em::bx3);
// // }

// // ex1_cnt = m_sim.mblock().em0(i, j, ntt::em::ex1);
// // ex2_cnt = m_sim.mblock().em0(i, j, ntt::em::ex2);
// // ex3_cnt = m_sim.mblock().em0(i, j, ntt::em::ex3);
// // bx1_cnt = m_sim.mblock().em0(i, j, ntt::em::bx1);
// // bx2_cnt = m_sim.mblock().em0(i, j, ntt::em::bx2);
// // bx3_cnt = m_sim.mblock().em0(i, j, ntt::em::bx3);

// ex1_cnt = m_sim.mblock().aux(i, j, ntt::em::ex1);
// ex2_cnt = m_sim.mblock().aux(i, j, ntt::em::ex2);
// ex3_cnt = m_sim.mblock().aux(i, j, ntt::em::ex3);
// hx1_cnt = m_sim.mblock().aux(i, j, ntt::em::bx1);
// hx2_cnt = m_sim.mblock().aux(i, j, ntt::em::bx2);
// hx3_cnt = m_sim.mblock().aux(i, j, ntt::em::bx3);

// dx1_cnt = m_sim.mblock().em(i, j, ntt::em::ex1);
// dx2_cnt = m_sim.mblock().em(i, j, ntt::em::ex2);
// dx3_cnt = m_sim.mblock().em(i, j, ntt::em::ex3);
// bx1_cnt = m_sim.mblock().em(i, j, ntt::em::bx1);
// bx2_cnt = m_sim.mblock().em(i, j, ntt::em::bx2);
// bx3_cnt = m_sim.mblock().em(i, j, ntt::em::bx3);

// ntt::vec_t<ntt::Dimension::THREE_D> d_hat, b_hat, e_hat, h_hat;

// // #if (SIMTYPE == PIC_SIMTYPE)
// //         m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {ex1_cnt, ex2_cnt, ex3_cnt}, e_hat);
// //         m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {bx1_cnt, bx2_cnt, bx3_cnt}, b_hat);
// // #elif (SIMTYPE == GRPIC_SIMTYPE)
// //         e_hat[0] = ex1_cnt;
// //         e_hat[1] = ex2_cnt;
// //         e_hat[2] = ex3_cnt;
// //         b_hat[0] = bx1_cnt;
// //         b_hat[1] = bx2_cnt;
// //         b_hat[2] = bx3_cnt;
// // #endif

// // m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {ex1_cnt, ex2_cnt, ex3_cnt}, e_hat);
// // m_sim.mblock().metric.v_Cntrv2Hat({i_ + HALF, j_ + HALF}, {bx1_cnt, bx2_cnt, bx3_cnt}, b_hat);

// // e_hat[0] = SIGN(ex1_cnt) * math::pow(math::abs(ex1_cnt), 0.25);
// // e_hat[1] = SIGN(ex2_cnt) * math::pow(math::abs(ex2_cnt), 0.25);
// // e_hat[2] = SIGN(ex3_cnt) * math::pow(math::abs(ex3_cnt), 0.25);
// // b_hat[0] = SIGN(bx1_cnt) * math::pow(math::abs(bx1_cnt), 0.25);
// // b_hat[1] = SIGN(bx2_cnt) * math::pow(math::abs(bx2_cnt), 0.25);
// // b_hat[2] = SIGN(bx3_cnt) * math::pow(math::abs(bx3_cnt), 0.25);
// e_hat[0] = ex1_cnt;
// e_hat[1] = ex2_cnt;
// e_hat[2] = ex3_cnt;
// b_hat[0] = bx1_cnt;
// b_hat[1] = bx2_cnt;
// b_hat[2] = bx3_cnt;
// d_hat[0] = dx1_cnt;
// d_hat[1] = dx2_cnt;
// d_hat[2] = dx3_cnt;
// h_hat[0] = hx1_cnt;
// h_hat[1] = hx2_cnt;
// h_hat[2] = hx3_cnt;

// // convert from contravariant to hatted
// m_data[0].set(i, j, e_hat[0]);

// // m_ex2.set(i, j, e_hat[1]);
// // m_ex3.set(i, j, e_hat[2]);
// // m_bx1.set(i, j, b_hat[0]);
// // m_bx2.set(i, j, b_hat[1]);
// // m_bx3.set(i, j, b_hat[2]);
// // m_dx1.set(i, j, d_hat[0]);
// // m_dx2.set(i, j, d_hat[1]);
// // m_dx3.set(i, j, d_hat[2]);
// // m_hx1.set(i, j, h_hat[0]);
// // m_hx2.set(i, j, h_hat[1]);
// // m_hx3.set(i, j, h_hat[2]);

// // m_ex1.set(i, j, ex1_cnt);
// // m_ex2.set(i, j, ex2_cnt);
// // m_ex3.set(i, j, ex3_cnt);
// // m_bx1.set(i, j, bx1_cnt);
// // m_bx2.set(i, j, bx2_cnt);
// // m_bx3.set(i, j, bx3_cnt);
// }

#include "writer.h"

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <plog/Log.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

namespace ntt {

#ifdef OUTPUT_ENABLED
  template <Dimension D, SimulationEngine S>
  Writer<D, S>::Writer(const SimulationParams& params, const Meshblock<D, S>& mblock) {
    m_io = m_adios.DeclareIO("WriteKokkos");
    m_io.SetEngine("HDF5");
    adios2::Dims shape, start, count;
    for (short d = 0; d < (short)D; ++d) {
      shape.push_back(mblock.Ni(d) + 2 * N_GHOSTS);
      count.push_back(mblock.Ni(d) + 2 * N_GHOSTS);
      start.push_back(0);
    }
    std::reverse(shape.begin(), shape.end());
    std::reverse(count.begin(), count.end());

    m_vars_i.emplace("step", m_io.DefineVariable<int>("step"));
    m_vars_r.emplace("time", m_io.DefineVariable<real_t>("time"));

    m_io.DefineAttribute<std::string>("metric", mblock.metric.label);
    if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("x1_min", mblock.metric.x1_min);
      m_io.DefineAttribute<real_t>("x1_max", mblock.metric.x1_max);

      auto x1 = new real_t[mblock.Ni1() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni1(); ++i) {
        auto x_ = mblock.metric.x1_min
                  + (mblock.metric.x1_max - mblock.metric.x1_min) * i / mblock.Ni1();
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[0] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x1[i] = xph[0];
      }
      m_io.DefineAttribute<real_t>("x1", x1, mblock.Ni1() + 1);
    }
    if constexpr (D == Dim2 || D == Dim3) {
      m_io.DefineAttribute<real_t>("x2_min", mblock.metric.x2_min);
      m_io.DefineAttribute<real_t>("x2_max", mblock.metric.x2_max);

      auto x2 = new real_t[mblock.Ni2() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni2(); ++i) {
        auto x_ = mblock.metric.x2_min
                  + (mblock.metric.x2_max - mblock.metric.x2_min) * i / mblock.Ni2();
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[1] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x2[i] = xph[1];
      }
      m_io.DefineAttribute<real_t>("x2", x2, mblock.Ni2() + 1);
    }
    if constexpr (D == Dim3) {
      m_io.DefineAttribute<real_t>("x3_min", mblock.metric.x3_min);
      m_io.DefineAttribute<real_t>("x3_max", mblock.metric.x3_max);

      auto x3 = new real_t[mblock.Ni3() + 1];
      for (std::size_t i { 0 }; i <= mblock.Ni3(); ++i) {
        coord_t<D> xph { ZERO }, xi;
        for (short d { 0 }; d < (short)D; ++d) {
          xi[d] = ONE;
        }
        xi[2] = (real_t)(i);
#  ifdef MINKOWSKI_METRIC
        mblock.metric.x_Code2Cart(xi, xph);
#  else
        mblock.metric.x_Code2Sph(xi, xph);
#  endif
        x3[i] = xph[2];
      }
      m_io.DefineAttribute<real_t>("x3", x3, mblock.Ni3() + 1);
    }
    m_io.DefineAttribute<int>("n_ghosts", N_GHOSTS);
    m_io.DefineAttribute<int>("dimension", (int)D);

    m_io.DefineAttribute<real_t>("dt", mblock.timestep());

    for (auto& var : { "ex1", "ex2", "ex3", "bx1", "bx2", "bx3" }) {
      m_vars_r.emplace(var, m_io.DefineVariable<real_t>(var, {}, {}, count));
    }
    m_vars_r.emplace("density", m_io.DefineVariable<real_t>("density", {}, {}, count));
  }

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::~Writer() {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams& params,
                                 const Meshblock<D, S>&  mblock,
                                 const real_t&           time,
                                 const std::size_t&      tstep) {
    m_writer = m_io.Open(params.title() + ".flds.h5", m_mode);
    m_mode   = adios2::Mode::Append;

    m_writer.BeginStep();

    int step = (int)tstep;
    m_writer.Put<int>(m_vars_i["step"], &step);
    m_writer.Put<real_t>(m_vars_r["time"], &time);

    std::vector<std::string> field_names = { "ex1", "ex2", "ex3", "bx1", "bx2", "bx3" };
    std::vector<em>          fields = { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 };

    // auto slice_i1 = std::make_pair(mblock.i_min(0), mblock.i_max(0));
    // auto slice_i2 = std::make_pair(mblock.i_min(1), mblock.i_max(1));
    // auto slice_i3 = std::make_pair(mblock.i_min(2), mblock.i_max(2));
    auto                     slice_i1 = Kokkos::ALL();
    auto                     slice_i2 = Kokkos::ALL();
    auto                     slice_i3 = Kokkos::ALL();

    if constexpr (D == Dim1) {
      for (auto&& [var_st, var] : c9::zip(field_names, fields)) {
        m_writer.Put<real_t>(m_vars_r[var_st],
                             Kokkos::subview(mblock.bckp_h, slice_i1, (int)var));
      }
      m_writer.Put<real_t>(m_vars_r["density"],
                           Kokkos::subview(mblock.buff_h, slice_i1, (int)(fld::dens)));
    } else if constexpr (D == Dim2) {
      for (auto&& [var_st, var] : c9::zip(field_names, fields)) {
        m_writer.Put<real_t>(m_vars_r[var_st],
                             Kokkos::subview(mblock.bckp_h, slice_i1, slice_i2, (int)var));
      }
      m_writer.Put<real_t>(
        m_vars_r["density"],
        Kokkos::subview(mblock.buff_h, slice_i1, slice_i2, (int)(fld::dens)));
    } else if constexpr (D == Dim3) {
      for (auto&& [var_st, var] : c9::zip(field_names, fields)) {
        m_writer.Put<real_t>(
          m_vars_r[var_st],
          Kokkos::subview(mblock.bckp_h, slice_i1, slice_i2, slice_i3, (int)var));
      }
      m_writer.Put<real_t>(
        m_vars_r["density"],
        Kokkos::subview(mblock.buff_h, slice_i1, slice_i2, slice_i3, (int)(fld::dens)));
    }
    m_writer.EndStep();
    m_writer.Close();
    PLOGD << "Wrote fields to file.";
  }

#else

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::Writer(const SimulationParams&, const Meshblock<D, S>&) {}

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::~Writer() {}

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams&,
                                 const Meshblock<D, S>&,
                                 const real_t&,
                                 const std::size_t&) {}

#endif

}    // namespace ntt

template class ntt::Writer<ntt::Dim1, ntt::PICEngine>;
template class ntt::Writer<ntt::Dim2, ntt::PICEngine>;
template class ntt::Writer<ntt::Dim3, ntt::PICEngine>;
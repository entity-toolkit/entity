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

    for (auto& var : params.outputFields()) {
      m_vars_r.emplace(var, m_io.DefineVariable<real_t>(var, {}, {}, count));
    }
  }

  template <Dimension D, SimulationEngine S>
  Writer<D, S>::~Writer() {}

  template <Dimension D, int N>
  void PutField(adios2::Engine&                 writer,
                const adios2::Variable<real_t>& var,
                const ndfield_mirror_t<D, N>&   field,
                const int&                      comp) {
    auto slice_i1 = Kokkos::ALL();
    auto slice_i2 = Kokkos::ALL();
    auto slice_i3 = Kokkos::ALL();

    if constexpr (D == Dim1) {
      writer.Put<real_t>(var, Kokkos::subview(field, slice_i1, comp));
    } else if constexpr (D == Dim2) {
      writer.Put<real_t>(var, Kokkos::subview(field, slice_i1, slice_i2, comp));
    } else if constexpr (D == Dim3) {
      writer.Put<real_t>(var, Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp));
    }
  }

  template <Dimension D, SimulationEngine S>
  void Writer<D, S>::WriteFields(const SimulationParams& params,
                                 Meshblock<D, S>&        mblock,
                                 const real_t&           time,
                                 const std::size_t&      tstep) {
    m_writer = m_io.Open(params.title() + ".flds.h5", m_mode);
    m_mode   = adios2::Mode::Append;

    m_writer.BeginStep();

    int step = (int)tstep;
    m_writer.Put<int>(m_vars_i["step"], &step);
    m_writer.Put<real_t>(m_vars_r["time"], &time);

    mblock.InterpolateAndConvertFieldsToHat();
    mblock.SynchronizeHostDevice();

    // traverse all the fields and put them. ...
    // ... also make sure that the fields are ready for output, ...
    // ... i.e. they have been written into proper arrays
    for (auto& var_str : params.outputFields()) {
      auto var = m_vars_r[var_str];
      if (var_str == "Ex" || var_str == "Er") {
        NTTHostErrorIf(mblock.bckp_h_content[em::ex1] != Content::ex1_hat_int,
                       "Ex1 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::ex1);
      } else if (var_str == "Ey" || var_str == "Etheta") {
        NTTHostErrorIf(mblock.bckp_h_content[em::ex2] != Content::ex2_hat_int,
                       "Ex2 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::ex2);
      } else if (var_str == "Ez" || var_str == "Ephi") {
        NTTHostErrorIf(mblock.bckp_h_content[em::ex3] != Content::ex3_hat_int,
                       "Ex3 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::ex3);
      } else if (var_str == "Bx" || var_str == "Br") {
        NTTHostErrorIf(mblock.bckp_h_content[em::bx1] != Content::bx1_hat_int,
                       "Bx1 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::bx1);
      } else if (var_str == "By" || var_str == "Btheta") {
        NTTHostErrorIf(mblock.bckp_h_content[em::bx2] != Content::bx2_hat_int,
                       "Bx2 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::bx2);
      } else if (var_str == "Bz" || var_str == "Bphi") {
        NTTHostErrorIf(mblock.bckp_h_content[em::bx3] != Content::bx3_hat_int,
                       "Bx3 is not ready for output");
        PutField<D, 6>(m_writer, var, mblock.bckp_h, em::bx3);
      } else if (var_str == "Jx" || var_str == "Jr") {
        NTTHostErrorIf(mblock.cur_h_content[cur::jx1] != Content::jx1_hat_int,
                       "Jx1 is not ready for output");
        PutField<D, 3>(m_writer, var, mblock.cur_h, cur::jx1);
      } else if (var_str == "Jy" || var_str == "Jtheta") {
        NTTHostErrorIf(mblock.cur_h_content[cur::jx2] != Content::jx2_hat_int,
                       "Jx2 is not ready for output");
        PutField<D, 3>(m_writer, var, mblock.cur_h, cur::jx2);
      } else if (var_str == "Jz" || var_str == "Jphi") {
        NTTHostErrorIf(mblock.cur_h_content[cur::jx3] != Content::jx3_hat_int,
                       "Jx3 is not ready for output");
        PutField<D, 3>(m_writer, var, mblock.cur_h, cur::jx3);
      } else {
        const std::vector<fld> fld_comps
          = { fld::dens, fld::chdens, fld::enrgdens, fld::dens };
        const std::vector<std::string> fld_labels
          = { "mass_density", "charge_density", "energy_density", "number_density" };
        const std::vector<Content> fld_contents = { Content::mass_density,
                                                    Content::charge_density,
                                                    Content::energy_density,
                                                    Content::number_density };
        for (int f { 0 }; f < fld_comps.size(); ++f) {
          if (var_str == fld_labels[f]) {
            if (mblock.buff_h_content[fld_comps[f]] != fld_contents[f]) {
              mblock.ComputeMoments(params, fld_contents[f], fld_comps[f]);
              mblock.SynchronizeHostDevice(Synchronize_buff);
            }
            NTTHostErrorIf(mblock.buff_h_content[fld_comps[f]] != fld_contents[f],
                           var_str + " is not ready for output");
            PutField<D, 3>(m_writer, var, mblock.buff_h, fld_comps[f]);
            ImposeEmptyContent(mblock.buff_h_content[fld_comps[f]]);
            ImposeEmptyContent(mblock.buff_content[fld_comps[f]]);
            break;
          }
        }
      }
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
#include "output.h"

#include "wrapper.h"

#include "fields.h"
#include "meshblock.h"
#include "sim_params.h"
#include "utils.h"

#ifdef OUTPUT_ENABLED
#  include <adios2.h>
#  include <adios2/cxx11/KokkosView.h>
#endif

#include <iostream>
#include <string>
#include <vector>

namespace ntt {
  namespace {
    auto InterpretInputField_helper(const FieldID&          fid,
                                    const std::vector<int>& comps,
                                    const std::vector<int>& species) -> OutputField {
      OutputField of;
      of.setId(fid);
      std::string fldname { StringizeFieldID(fid) };
      for (auto c : comps) {
        of.comp.push_back(c);
#ifdef MINKOWSKI_METRIC
        fldname += (c == 0 ? "t" : (c == 1 ? "x" : (c == 2 ? "y" : "z")));
#else
        fldname += std::to_string(c);
#endif
      }
      if (species.size() > 0) {
        fldname += "_";
        for (auto s : species) {
          of.species.push_back(s);
          fldname += std::to_string(s);
          fldname += "_";
        }
        fldname.pop_back();
      }
      of.setName(fldname);
      return of;
    }

    auto InterpretInputField_getcomponents(const std::vector<std::string>& comps)
      -> std::vector<std::vector<int>> {
      NTTHostErrorIf(comps.size() > 2, "Invalid field name");
      std::vector<int> comps_int;
      for (auto& c : comps) {
        TestValidOption(c, { "t", "x", "y", "z", "0", "1", "2", "3", "i", "j" });
        if (c == "t") {
          comps_int.push_back(0);
        } else if (c == "x") {
          comps_int.push_back(1);
        } else if (c == "y") {
          comps_int.push_back(2);
        } else if (c == "z") {
          comps_int.push_back(3);
        } else if (c == "i") {
          comps_int.push_back(-1);
        } else if (c == "j") {
          comps_int.push_back(-2);
        } else {
          comps_int.push_back(std::stoi(c));
        }
      }
      std::vector<std::vector<int>> comps_ints;
      if (comps_int.size() == 1) {
        auto c = comps_int[0];
        if (c < 0) {
          for (int i { 0 }; i < 3; ++i) {
            comps_ints.push_back({ i + 1 });
          }
        } else {
          NTTHostErrorIf((c > 3) || (c == 0), "Invalid field name");
          comps_ints.push_back({ c });
        }
      } else {
        auto c1 = comps_int[0];
        auto c2 = comps_int[1];
        if (c1 < 0) {
          for (int i { 0 }; i < 3; ++i) {
            if (c2 < 0) {
              for (int j { i }; j < 3; ++j) {
                comps_ints.push_back({ i + 1, j + 1 });
              }
            } else {
              comps_ints.push_back({ i + 1, c2 });
            }
          }
        } else {
          if (c2 < 0) {
            for (int j { 0 }; j < 3; ++j) {
              comps_ints.push_back({ c1, j + 1 });
            }
          } else {
            comps_ints.push_back({ c1, c2 });
          }
        }
      }
      return comps_ints;
    }

    auto InterpretInputField_getspecies(const std::string& fld) -> std::vector<int> {
      std::vector<int> species;
      if (fld.find("_") < fld.size()) {
        auto species_str = SplitString(fld.substr(fld.find("_") + 1), "_");
        for (const auto& specie : species_str) {
          species.push_back(std::stoi(specie));
        }
      }
      return species;
    }
  }    // namespace

  void OutputField::show(std::ostream& os) const {
    os << "OutputField: " << m_name << " (" << StringizeFieldID(m_id) << ")\n";
    if (comp.size() == 2) {
      os << "  comp: " << comp[0] << " " << comp[1] << "\n";
    } else if (comp.size() == 1) {
      os << "  comp: " << comp[0] << "\n";
    }
    if (species.size() > 0) {
      os << "  species: ";
      for (const auto& s : species) {
        os << s << " ";
      }
    }
    os << "\n";
  }

  auto InterpretInputField(const std::string& fld) -> std::vector<OutputField> {
    std::vector<OutputField> ofs;
    if (fld.find("T") == 0) {
      const auto id      = FieldID::T;
      auto       species = InterpretInputField_getspecies(fld);
      auto comps = InterpretInputField_getcomponents({ fld.substr(1, 1), fld.substr(2, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, species));
      }
    } else if (fld.find("Rho") == 0) {
      const auto id      = FieldID::Rho;
      auto       species = InterpretInputField_getspecies(fld);
      ofs.emplace_back(InterpretInputField_helper(id, {}, species));
    } else if (fld.find("N") == 0) {
      const auto id      = FieldID::N;
      auto       species = InterpretInputField_getspecies(fld);
      ofs.emplace_back(InterpretInputField_helper(id, {}, species));
    } else if (fld.find("E") == 0) {
      const auto id    = FieldID::E;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else if (fld.find("B") == 0) {
      const auto id    = FieldID::B;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else if (fld.find("J") == 0) {
      const auto id    = FieldID::J;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      for (auto& comp : comps) {
        ofs.emplace_back(InterpretInputField_helper(id, comp, {}));
      }
    } else {
      NTTHostError("Invalid field name");
    }
    return ofs;
  }

#ifdef OUTPUT_ENABLED
  namespace {
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
  }    // namespace

  template <Dimension D, SimulationEngine S>
  void OutputField::put(adios2::Engine&                 writer,
                        const adios2::Variable<real_t>& var,
                        const SimulationParams&         params,
                        Meshblock<D, S>&                mblock) const {
    if ((m_id == FieldID::E) || (m_id == FieldID::B)) {
      mblock.InterpolateAndConvertFieldsToHat();
      mblock.SynchronizeHostDevice(Synchronize_bckp);
      ImposeEmptyContent(mblock.bckp_content);
      // EM fields (vector)
      std::vector<em>      comp_options;
      std::vector<Content> content_options;
      if (m_id == FieldID::E) {
        comp_options    = { em::ex1, em::ex2, em::ex3 };
        content_options = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int };
      } else if (m_id == FieldID::B) {
        comp_options    = { em::bx1, em::bx2, em::bx3 };
        content_options = { Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
      }
      auto comp_id = comp_options[comp[0] - 1];
      auto cont_id = content_options[comp[0] - 1];
      AssertContent({ mblock.bckp_h_content[comp_id] }, { cont_id });
      PutField<D, 6>(writer, var, mblock.bckp_h, (int)(comp_id));
    } else if (m_id == FieldID::J) {
      mblock.InterpolateAndConvertCurrentsToHat();
      mblock.SynchronizeHostDevice(Synchronize_cur);
      ImposeEmptyContent(mblock.cur_content);
      // Currents (vector)
      std::vector<cur>     comp_options = { cur::jx1, cur::jx2, cur::jx3 };
      std::vector<Content> content_options
        = { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int };
      auto comp_id = comp_options[comp[0] - 1];
      auto cont_id = content_options[comp[0] - 1];
      AssertContent({ mblock.cur_h_content[comp_id] }, { cont_id });
      PutField<D, 3>(writer, var, mblock.cur_h, (int)(comp_id));
    } else {
      mblock.ComputeMoments(params, m_id, comp, species, 0);
      mblock.SynchronizeHostDevice(Synchronize_buff);
      PutField<D, 3>(writer, var, mblock.buff_h, 0);
      ImposeEmptyContent(mblock.buff_h_content[0]);
    }
  }

#endif
}    // namespace ntt

#ifdef OUTPUT_ENABLED

template void ntt::OutputField::put<ntt::Dim1, ntt::PICEngine>(
  adios2::Engine&,
  const adios2::Variable<real_t>&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim1, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim2, ntt::PICEngine>(
  adios2::Engine&,
  const adios2::Variable<real_t>&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim2, ntt::PICEngine>&) const;
template void ntt::OutputField::put<ntt::Dim3, ntt::PICEngine>(
  adios2::Engine&,
  const adios2::Variable<real_t>&,
  const ntt::SimulationParams&,
  ntt::Meshblock<ntt::Dim3, ntt::PICEngine>&) const;

#endif
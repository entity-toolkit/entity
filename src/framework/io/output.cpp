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
    auto InterpretInputField_helper(const FieldID&                       fid,
                                    const std::vector<std::vector<int>>& comps,
                                    const std::vector<int>& species) -> OutputField {
      OutputField of;
      of.setId(fid);
      for (auto ci : comps) {
        std::vector<int> component;
        for (auto c : ci) {
          component.push_back(c);
        }
        of.comp.push_back(component);
      }
      for (auto s : species) {
        of.species.push_back(s);
      }
      of.setName(StringizeFieldID(fid));
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

  auto InterpretInputField(const std::string& fld) -> OutputField {
    FieldID                       id;
    std::vector<std::vector<int>> comps   = { {} };
    std::vector<int>              species = {};
    if (fld.find("T") == 0) {
      id = FieldID::T;
    } else if (fld.find("Rho") == 0) {
      id = FieldID::Rho;
    } else if (fld.find("Nppc") == 0) {
      id = FieldID::Nppc;
    } else if (fld.find("N") == 0) {
      id = FieldID::N;
    } else if (fld.find("E") == 0) {
      id = FieldID::E;
    } else if (fld.find("B") == 0) {
      id = FieldID::B;
    } else if (fld.find("D") == 0) {
      id = FieldID::D;
    } else if (fld.find("H") == 0) {
      id = FieldID::H;
    } else if (fld.find("J") == 0) {
      id = FieldID::J;
    } else {
      NTTHostError("Invalid field name");
    }
    auto is_moment
      = (id == FieldID::T || id == FieldID::Rho || id == FieldID::Nppc || id == FieldID::N);
    auto is_field = (id == FieldID::E || id == FieldID::B || id == FieldID::D
                     || id == FieldID::H || id == FieldID::J);
    if (is_moment) {
      species = InterpretInputField_getspecies(fld);
    } else if (is_field) {
      comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
    }
    if (id == FieldID::T) {
      comps = InterpretInputField_getcomponents({ fld.substr(1, 1), fld.substr(2, 1) });
    }
    return InterpretInputField_helper(id, comps, species);
  }

#ifdef OUTPUT_ENABLED
  namespace {
    template <Dimension D, int N>
    void PutField(adios2::IO&            io,
                  adios2::Engine&        writer,
                  const std::string&     varname,
                  const ndfield_t<D, N>& field,
                  const int&             comp) {
      auto slice_i1 = Kokkos::ALL;
      auto slice_i2 = Kokkos::ALL;
      auto slice_i3 = Kokkos::ALL;

      auto var      = io.InquireVariable<real_t>(varname);

      if constexpr (D == Dim1) {
        auto slice        = Kokkos::subview(field, slice_i1, comp);
        auto output_field = array_t<real_t*>("output_field", slice.extent(0));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      } else if constexpr (D == Dim2) {
        auto slice = Kokkos::subview(field, slice_i1, slice_i2, comp);
        auto output_field
          = array_t<real_t**>("output_field", slice.extent(0), slice.extent(1));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      } else if constexpr (D == Dim3) {
        auto slice        = Kokkos::subview(field, slice_i1, slice_i2, slice_i3, comp);
        auto output_field = array_t<real_t***>(
          "output_field", slice.extent(0), slice.extent(1), slice.extent(2));
        Kokkos::deep_copy(output_field, slice);
        auto output_field_host = Kokkos::create_mirror_view(output_field);
        Kokkos::deep_copy(output_field_host, output_field);
        writer.Put<real_t>(var, output_field_host);
      }
    }
  }    // namespace

  template <Dimension D, SimulationEngine S>
  void OutputField::put(adios2::IO&             io,
                        adios2::Engine&         writer,
                        const SimulationParams& params,
                        Meshblock<D, S>&        mblock) const {
    // PrepareOutputFlags flags;
    // if constexpr (S != GRPICEngine) {
    //   flags = PrepareOutput_Default;
    // } else {
    if constexpr (S == GRPICEngine) {
      if (m_id == FieldID::E || m_id == FieldID::H) {
        NTTHostError("Output of E and H (aux) fields is not supported yet");
      }
    }
    //   flags = PrepareOutput_InterpToCellCenter;
    // }
    if ((m_id == FieldID::E) || (m_id == FieldID::B) || (m_id == FieldID::D)
        || (m_id == FieldID::H)) {
      mblock.PrepareFieldsForOutput();
      ImposeEmptyContent(mblock.bckp_content);
      std::vector<em>      comp_options;
      std::vector<Content> content_options
        = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
            Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
      if (m_id == FieldID::E || m_id == FieldID::D) {
        comp_options = { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 };
      } else if (m_id == FieldID::B || m_id == FieldID::H) {
        comp_options = { em::bx1, em::bx2, em::bx3 };
      }
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 6>(io, writer, name(i), mblock.bckp, (int)(comp_id));
      }
    } else if (m_id == FieldID::J) {
      mblock.PrepareCurrentsForOutput();
      ImposeEmptyContent(mblock.cur_content);
      std::vector<cur>     comp_options = { cur::jx1, cur::jx2, cur::jx3 };
      std::vector<Content> content_options
        = { Content::jx1_hat_int, Content::jx2_hat_int, Content::jx3_hat_int };
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 3>(io, writer, name(i), mblock.cur, (int)(comp_id));
      }
    } else {
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        // no smoothing for FieldID::Nppc
        mblock.ComputeMoments(params,
                              m_id,
                              comp[i],
                              species,
                              i % 3,
                              m_id == FieldID::Nppc ? 0 : params.outputMomSmooth());
        PutField<D, 3>(io, writer, name(i), mblock.buff, i % 3);
      }
    }
  }

#endif
}    // namespace ntt
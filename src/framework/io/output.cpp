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
    if (fld.find("T") == 0) {
      const auto id      = FieldID::T;
      auto       species = InterpretInputField_getspecies(fld);
      auto comps = InterpretInputField_getcomponents({ fld.substr(1, 1), fld.substr(2, 1) });
      return InterpretInputField_helper(id, comps, species);
    } else if (fld.find("Rho") == 0) {
      const auto id      = FieldID::Rho;
      auto       species = InterpretInputField_getspecies(fld);
      return InterpretInputField_helper(id, { {} }, species);
    } else if (fld.find("Nppc") == 0) {
      const auto id      = FieldID::Nppc;
      auto       species = InterpretInputField_getspecies(fld);
      return InterpretInputField_helper(id, { {} }, species);
    } else if (fld.find("N") == 0) {
      const auto id      = FieldID::N;
      auto       species = InterpretInputField_getspecies(fld);
      return InterpretInputField_helper(id, { {} }, species);
    } else if (fld.find("E") == 0) {
      const auto id    = FieldID::E;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      return InterpretInputField_helper(id, comps, {});
    } else if (fld.find("B") == 0) {
      const auto id    = FieldID::B;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      return InterpretInputField_helper(id, comps, {});
    } else if (fld.find("J") == 0) {
      const auto id    = FieldID::J;
      auto       comps = InterpretInputField_getcomponents({ fld.substr(1, 1) });
      return InterpretInputField_helper(id, comps, {});
    }
    NTTHostError("Invalid field name");
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
    if ((m_id == FieldID::E) || (m_id == FieldID::B)) {
      mblock.InterpolateAndConvertFieldsToHat();
      ImposeEmptyContent(mblock.bckp_content);
      // EM fields (vector)
      std::vector<em>      comp_options;
      std::vector<Content> content_options
        = { Content::ex1_hat_int, Content::ex2_hat_int, Content::ex3_hat_int,
            Content::bx1_hat_int, Content::bx2_hat_int, Content::bx3_hat_int };
      if (m_id == FieldID::E) {
        comp_options = { em::ex1, em::ex2, em::ex3, em::bx1, em::bx2, em::bx3 };
      } else if (m_id == FieldID::B) {
        comp_options = { em::bx1, em::bx2, em::bx3 };
      }
      for (std::size_t i { 0 }; i < comp.size(); ++i) {
        auto comp_id = comp_options[comp[i][0] - 1];
        PutField<D, 6>(io, writer, name(i), mblock.bckp, (int)(comp_id));
      }
    } else if (m_id == FieldID::J) {
      mblock.InterpolateAndConvertCurrentsToHat();
      ImposeEmptyContent(mblock.cur_content);
      // Currents (vector)
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
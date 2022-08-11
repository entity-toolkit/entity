#include "global.h"
#include "output_csv.h"
#include "meshblock.h"

#include <rapidcsv.h>

#include <string>
#include <filesystem>
#include <cassert>

namespace ntt {
  namespace csv {
    void ensureFileExists(const std::string& filename) {
      std::filesystem::path file(filename);
      assert(std::filesystem::exists(file));
    }

    template <Dimension D, SimulationType S>
    void
    writeField(const std::string& filename, const Meshblock<D, S>& mblock, const em& field) {
      rapidcsv::Document doc(
        "", rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', false, false));
      if constexpr (D == Dimension::ONE_D) {
        auto N = mblock.Ni1();
        for (int i {0}; i < N; ++i) {
          doc.SetCell<real_t>(i, 0, mblock.em(i + ntt::N_GHOSTS, field));
        }
      } else if constexpr (D == Dimension::TWO_D) {
        auto N1 = mblock.Ni1();
        auto N2 = mblock.Ni2();
        for (int i {0}; i < N1; ++i) {
          for (int j {0}; j < N2; ++j) {
            doc.SetCell<real_t>(i, j, mblock.em(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS, field));
          }
        }
      } else {
        (void)(field);
        NTTError("Cannot write 3D field data as csv");
      }
      doc.Save(filename);
    }

    template <Dimension D, SimulationType S>
    void
    writeField(const std::string& filename, const Meshblock<D, S>& mblock, const cur& field) {
      rapidcsv::Document doc(
        "", rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', false, false));
      if constexpr (D == Dimension::ONE_D) {
        auto N = mblock.Ni1();
        for (int i {0}; i < N; ++i) {
          doc.SetCell<real_t>(i, 0, mblock.cur(i + ntt::N_GHOSTS, field));
        }
      } else if constexpr (D == Dimension::TWO_D) {
        auto N1 = mblock.Ni1();
        auto N2 = mblock.Ni2();
        for (int i {0}; i < N1; ++i) {
          for (int j {0}; j < N2; ++j) {
            doc.SetCell<real_t>(i, j, mblock.cur(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS, field));
          }
        }
      } else {
        (void)(field);
        NTTError("Cannot write 3D field data as csv");
      }
      doc.Save(filename);
    }

    template <Dimension D, SimulationType S>
    void writeParticle(std::string            filename,
                       const Meshblock<D, S>& mblock,
                       const std::size_t&     species_id,
                       const std::size_t&     prtl_id,
                       const OutputMode&      mode) {
      int         i_new = 0;
      std::string fname = "";

      if (mode == OutputMode::APPEND) {
        ensureFileExists(filename);
        fname = filename;
      }

      rapidcsv::Document doc(fname, rapidcsv::LabelParams(0, -1));

      if (mode == OutputMode::APPEND) {
        i_new = doc.GetRowCount();
      } else if (mode == OutputMode::WRITE) {
        doc.SetColumnName(0, "ux1");
        doc.SetColumnName(1, "ux2");
        doc.SetColumnName(2, "ux3");
        doc.SetColumnName(3, "w");
        doc.SetColumnName(4, "x1");
        if constexpr (D == Dimension::TWO_D) {
          doc.SetColumnName(5, "x2");
        } else if constexpr (D == Dimension::THREE_D) {
          doc.SetColumnName(5, "x2");
          doc.SetColumnName(6, "x3");
        }
      }

      doc.SetCell<real_t>(0, i_new, mblock.particles[species_id].ux1(prtl_id));
      doc.SetCell<real_t>(1, i_new, mblock.particles[species_id].ux2(prtl_id));
      doc.SetCell<real_t>(2, i_new, mblock.particles[species_id].ux3(prtl_id));
      doc.SetCell<real_t>(3, i_new, mblock.particles[species_id].weight(prtl_id));
      auto x = (real_t)(mblock.particles[species_id].i1(prtl_id))
               + mblock.particles[species_id].dx1(prtl_id);
      doc.SetCell<real_t>(4, i_new, x);
      if constexpr (D == Dimension::TWO_D) {
        x = (real_t)(mblock.particles[species_id].i2(prtl_id))
            + mblock.particles[species_id].dx2(prtl_id);
        doc.SetCell<real_t>(5, i_new, x);
      } else if constexpr (D == Dimension::THREE_D) {
        x = (real_t)(mblock.particles[species_id].i2(prtl_id))
            + mblock.particles[species_id].dx2(prtl_id);
        doc.SetCell<real_t>(5, i_new, x);

        x = (real_t)(mblock.particles[species_id].i3(prtl_id))
            + mblock.particles[species_id].dx3(prtl_id);
        doc.SetCell<real_t>(6, i_new, x);
      }
      doc.Save(filename);
    }
  } // namespace csv
} // namespace ntt

#if SIMTYPE == PIC_SIMTYPE

using Meshblock1D = ntt::Meshblock<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>;
using Meshblock2D = ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>;
using Meshblock3D = ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>;

template void ntt::csv::writeField<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock1D&, const em&);
template void ntt::csv::writeField<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock2D&, const em&);
template void ntt::csv::writeField<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock3D&, const em&);

template void ntt::csv::writeField<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock1D&, const cur&);
template void ntt::csv::writeField<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock2D&, const cur&);
template void ntt::csv::writeField<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>(
  const std::string&, const Meshblock3D&, const cur&);

template void ntt::csv::writeParticle<ntt::Dimension::ONE_D, ntt::SimulationType::PIC>(
  std::string, const Meshblock1D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dimension::TWO_D, ntt::SimulationType::PIC>(
  std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dimension::THREE_D, ntt::SimulationType::PIC>(
  std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const OutputMode&);

#elif SIMTYPE == GRPIC_SIMTYPE

using Meshblock2D = ntt::Meshblock<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>;
using Meshblock3D = ntt::Meshblock<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>;

template void ntt::csv::writeField<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>(
  const std::string&, const Meshblock2D&, const em&);
template void ntt::csv::writeField<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>(
  const std::string&, const Meshblock3D&, const em&);

template void ntt::csv::writeParticle<ntt::Dimension::TWO_D, ntt::SimulationType::GRPIC>(
  std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const OutputMode&);
template void ntt::csv::writeParticle<ntt::Dimension::THREE_D, ntt::SimulationType::GRPIC>(
  std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const OutputMode&);

#endif
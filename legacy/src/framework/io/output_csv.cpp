#include "wrapper.h"
#include "output_csv.h"
#include "simulation.h"
#include "meshblock/meshblock.h"
#include "particle_macros.h"

#include <plog/Log.h>

#include <filesystem>
#include <string>
#include <cassert>
#include <fstream>

// namespace fs = std::filesystem;

namespace ntt {
  namespace csv {
    // void ensureFileExists(const std::string& filename) {
    //   fs::path file(filename);
    //   if (!fs::exists(file)) { throw std::runtime_error("File does not exist: " + filename);
    //   }
    // }

    // template <Dimension D, SimulationType S>
    // void writeField(const std::string&, const Meshblock<D, S>&, const em&) {
    //   // writeField(const std::string& filename, const Meshblock<D, S>& mblock, const em&
    //   // field) { rapidcsv::Document doc(
    //   //   "", rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', false, false));
    //   // if constexpr (D == Dim1) {
    //   //   auto N = mblock.Ni1();
    //   //   for (int i {0}; i < N; ++i) {
    //   //     doc.SetCell<real_t>(i, 0, mblock.em(i + ntt::N_GHOSTS, field));
    //   //   }
    //   // } else if constexpr (D == Dim2) {
    //   //   auto N1 = mblock.Ni1();
    //   //   auto N2 = mblock.Ni2();
    //   //   for (int i {0}; i < N1; ++i) {
    //   //     for (int j {0}; j < N2; ++j) {
    //   //       doc.SetCell<real_t>(i, j, mblock.em(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS,
    //   //       field));
    //   //     }
    //   //   }
    //   // } else {
    //   //   (void)(field);
    //   //   NTTHostError("Cannot write 3D field data as csv");
    //   // }
    //   // doc.Save(filename);
    // }

    // template <Dimension D, SimulationType S>
    // void writeField(const std::string&, const Meshblock<D, S>&, const cur&) {
    //   // writeField(const std::string& filename, const Meshblock<D, S>& mblock, const cur&
    //   // field) {
    //   //   rapidcsv::Document doc(
    //   //     "", rapidcsv::LabelParams(-1, -1), rapidcsv::SeparatorParams(',', false,
    //   false));
    //   //   if constexpr (D == Dim1) {
    //   //     auto N = mblock.Ni1();
    //   //     for (int i {0}; i < N; ++i) {
    //   //       doc.SetCell<real_t>(i, 0, mblock.cur(i + ntt::N_GHOSTS, field));
    //   //     }
    //   //   } else if constexpr (D == Dim2) {
    //   //     auto N1 = mblock.Ni1();
    //   //     auto N2 = mblock.Ni2();
    //   //     for (int i {0}; i < N1; ++i) {
    //   //       for (int j {0}; j < N2; ++j) {
    //   //         doc.SetCell<real_t>(i, j, mblock.cur(i + ntt::N_GHOSTS, j + ntt::N_GHOSTS,
    //   //         field));
    //   //       }
    //   //     }
    //   //   } else {
    //   //     (void)(field);
    //   //     NTTHostError("Cannot write 3D field data as csv");
    //   //   }
    //   //   doc.Save(filename);
    // }

    // template <Dimension D, SimulationType S>
    // void writeParticle(std::string            filename,
    //                    const Meshblock<D, S>& mblock,
    //                    const std::size_t&     species_id,
    //                    const std::size_t&     prtl_id,
    //                    const OutputMode&      mode) {
    //   std::ofstream outfile;

    //   if (mode == OutputMode::APPEND) {
    //     try {
    //       ensureFileExists(filename);
    //       outfile.open(filename, std::ios_base::app);
    //     }
    //     catch (const std::exception& e) {
    //       PLOGI << e.what();
    //       PLOGI << "Creating new file instead.";
    //       outfile.open(filename);
    //       outfile << "ux1,ux2,ux3,w,x1,x2,x3" << std::endl;
    //     }
    //   }
    //   if (!outfile.is_open()) {
    //     throw std::runtime_error("Could not open or create file: " + filename);
    //   }

    //   outfile << mblock.particles[species_id].ux1(prtl_id) << ",";
    //   outfile << mblock.particles[species_id].ux2(prtl_id) << ",";
    //   outfile << mblock.particles[species_id].ux3(prtl_id) << ",";
    //   outfile << mblock.particles[species_id].weight(prtl_id);
    //   if constexpr (D == Dim1 || D == Dim2 || D == Dim3) {
    //     auto x1 = get_prtl_x1(mblock.particles[species_id], prtl_id);
    //     outfile << "," << x1;
    //   }
    //   if constexpr (D == Dim2 || D == Dim3) {
    //     auto x2 = get_prtl_x2(mblock.particles[species_id], prtl_id);
    //     outfile << "," << x2;
    //   } else if constexpr (D == Dim3) {
    //     auto x3 = get_prtl_x3(mblock.particles[species_id], prtl_id);
    //     outfile << "," << x3;
    //   }
    //   outfile << "\n";

    //   outfile.close();
    // }
  } // namespace csv
} // namespace ntt

#ifdef PIC_SIMTYPE

// using Meshblock1D = ntt::Meshblock<ntt::Dim1, ntt::TypePIC>;
// using Meshblock2D = ntt::Meshblock<ntt::Dim2, ntt::TypePIC>;
// using Meshblock3D = ntt::Meshblock<ntt::Dim3, ntt::TypePIC>;

// template void ntt::csv::writeField<ntt::Dim1, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock1D&,
//                                                             const em&);
// template void ntt::csv::writeField<ntt::Dim2, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock2D&,
//                                                             const em&);
// template void ntt::csv::writeField<ntt::Dim3, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock3D&,
//                                                             const em&);

// template void ntt::csv::writeField<ntt::Dim1, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock1D&,
//                                                             const cur&);
// template void ntt::csv::writeField<ntt::Dim2, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock2D&,
//                                                             const cur&);
// template void ntt::csv::writeField<ntt::Dim3, ntt::TypePIC>(const std::string&,
//                                                             const Meshblock3D&,
//                                                             const cur&);

// template void ntt::csv::writeParticle<ntt::Dim1, ntt::TypePIC>(
//   std::string, const Meshblock1D&, const std::size_t&, const std::size_t&, const
//   OutputMode&);
// template void ntt::csv::writeParticle<ntt::Dim2, ntt::TypePIC>(
//   std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const
//   OutputMode&);
// template void ntt::csv::writeParticle<ntt::Dim3, ntt::TypePIC>(
//   std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const
//   OutputMode&);

// #elif defined(GRPIC_SIMTYPE)

// using Meshblock2D = ntt::Meshblock<ntt::Dim2, ntt::SimulationType::GRPIC>;
// using Meshblock3D = ntt::Meshblock<ntt::Dim3, ntt::SimulationType::GRPIC>;

// template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::GRPIC>(const
// std::string&,
//                                                                           const
//                                                                           Meshblock2D&,
//                                                                           const em&);
// template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::GRPIC>(const
// std::string&,
//                                                                           const
//                                                                           Meshblock3D&,
//                                                                           const em&);

// template void ntt::csv::writeField<ntt::Dim2, ntt::SimulationType::GRPIC>(const
// std::string&,
//                                                                           const
//                                                                           Meshblock2D&,
//                                                                           const cur&);
// template void ntt::csv::writeField<ntt::Dim3, ntt::SimulationType::GRPIC>(const
// std::string&,
//                                                                           const
//                                                                           Meshblock3D&,
//                                                                           const cur&);

// template void ntt::csv::writeParticle<ntt::Dim2, ntt::SimulationType::GRPIC>(
//   std::string, const Meshblock2D&, const std::size_t&, const std::size_t&, const
//   OutputMode&);
// template void ntt::csv::writeParticle<ntt::Dim3, ntt::SimulationType::GRPIC>(
//   std::string, const Meshblock3D&, const std::size_t&, const std::size_t&, const
//   OutputMode&);

#endif
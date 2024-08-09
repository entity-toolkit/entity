#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/colors.h"
#include "utils/formatting.h"
#include "utils/progressbar.h"
#include "utils/timer.h"

#include "metrics/kerr_schild.h"
#include "metrics/kerr_schild_0.h"
#include "metrics/minkowski.h"
#include "metrics/qkerr_schild.h"
#include "metrics/qspherical.h"
#include "metrics/spherical.h"

#include "engines/engine.hpp"

#include <iomanip>
#include <iostream>

namespace ntt {
  namespace {} // namespace

  template <SimEngine::type S, class M>
  void print_particles(const Metadomain<S, M>&,
                       unsigned short,
                       DiagFlags,
                       std::ostream& = std::cout);

  template <SimEngine::type S, class M>
  void Engine<S, M>::print_step_report(timer::Timers&         timers,
                                       pbar::DurationHistory& time_history,
                                       bool                   print_output,
                                       bool print_sorting) const {
    DiagFlags  diag_flags  = Diag::Default;
    TimerFlags timer_flags = Timer::Default;
    if (not m_params.get<bool>("diagnostics.colored_stdout")) {
      diag_flags  ^= Diag::Colorful;
      timer_flags ^= Timer::Colorful;
    }
    if (m_params.get<std::size_t>("particles.nspec") == 0) {
      diag_flags ^= Diag::Species;
    }
    if (print_output) {
      timer_flags |= Timer::PrintOutput;
    }
    if (print_sorting) {
      timer_flags |= Timer::PrintSorting;
    }
    CallOnce(
      [diag_flags](auto& time, auto& step, auto& max_steps, auto& dt) {
        const auto c_bgreen = color::get_color("bgreen",
                                               diag_flags & Diag::Colorful);
        const auto c_bblack = color::get_color("bblack",
                                               diag_flags & Diag::Colorful);
        const auto c_reset = color::get_color("reset", diag_flags & Diag::Colorful);
        std::cout << fmt::format("Step:%s %-8d%s %s[of %d]%s\n",
                                 c_bgreen.c_str(),
                                 step,
                                 c_reset.c_str(),
                                 c_bblack.c_str(),
                                 max_steps,
                                 c_reset.c_str());
        std::cout << fmt::format("Time:%s %-8.4f%s %s[Δt = %.4f]%s\n",
                                 c_bgreen.c_str(),
                                 (double)time,
                                 c_reset.c_str(),
                                 c_bblack.c_str(),
                                 (double)dt,
                                 c_reset.c_str())
                  << std::endl;
      },
      time,
      step,
      max_steps,
      dt);
    if (diag_flags & Diag::Timers) {
      timers.printAll(timer_flags, std::cout);
    }
    CallOnce([]() {
      std::cout << std::endl;
    });
    if (diag_flags & Diag::Species) {
      CallOnce([diag_flags]() {
        std::cout << color::get_color("bblack", diag_flags & Diag::Colorful);
#if defined(MPI_ENABLED)
        std::cout << "Particle count:" << std::setw(22) << std::right << "[TOT]"
                  << std::setw(20) << std::right << "[MIN (%)]" << std::setw(20)
                  << std::right << "[MAX (%)]";
#else
        std::cout << "Particle count:" << std::setw(25) << std::right
                  << "[TOT (%)]";
#endif
        std::cout << color::get_color("reset", diag_flags & Diag::Colorful)
                  << std::endl;
      });
      for (std::size_t sp { 0 }; sp < m_metadomain.species_params().size(); ++sp) {
        print_particles(m_metadomain, sp, diag_flags, std::cout);
      }
      CallOnce([]() {
        std::cout << std::endl;
      });
    }
    if (diag_flags & Diag::Progress) {
      pbar::ProgressBar(time_history, step, max_steps, diag_flags, std::cout);
    }
    CallOnce([]() {
      std::cout << std::setw(80) << std::setfill('.') << "" << std::endl
                << std::endl;
    });
  }

  template <SimEngine::type S, class M>
  void print_particles(const Metadomain<S, M>& md,
                       unsigned short          sp,
                       DiagFlags               flags,
                       std::ostream&           os) {

    static_assert(M::is_metric, "template arg for Engine class has to be a metric");
    std::size_t npart { 0 };
    std::size_t maxnpart { 0 };
    std::string species_label;
    int         species_index;
    // sum npart & maxnpart over all subdomains on the current rank
    md.runOnLocalDomainsConst(
      [&npart, &maxnpart, &species_label, &species_index, sp](auto& dom) {
        npart         += dom.species[sp].npart();
        maxnpart      += dom.species[sp].maxnpart();
        species_label  = dom.species[sp].label();
        species_index  = dom.species[sp].index();
      });
#if defined(MPI_ENABLED)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<std::size_t> mpi_npart(size, 0);
    std::vector<std::size_t> mpi_maxnpart(size, 0);
    MPI_Gather(&npart,
               1,
               mpi::get_type<std::size_t>(),
               mpi_npart.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    MPI_Gather(&maxnpart,
               1,
               mpi::get_type<std::size_t>(),
               mpi_maxnpart.data(),
               1,
               mpi::get_type<std::size_t>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank != MPI_ROOT_RANK) {
      return;
    }
    auto tot_npart = std::accumulate(mpi_npart.begin(), mpi_npart.end(), 0);
    std::size_t npart_max = *std::max_element(mpi_npart.begin(), mpi_npart.end());
    std::size_t npart_min = *std::min_element(mpi_npart.begin(), mpi_npart.end());
    std::vector<double> mpi_load(size, 0.0);
    for (auto r { 0 }; r < size; ++r) {
      mpi_load[r] = 100.0 * (double)(mpi_npart[r]) / (double)(mpi_maxnpart[r]);
    }
    double load_max      = *std::max_element(mpi_load.begin(), mpi_load.end());
    double load_min      = *std::min_element(mpi_load.begin(), mpi_load.end());
    auto   npart_min_str = npart_min > 9999
                             ? fmt::format("%.2Le", (long double)npart_min)
                             : std::to_string(npart_min);
    auto   tot_npart_str = tot_npart > 9999
                             ? fmt::format("%.2Le", (long double)tot_npart)
                             : std::to_string(tot_npart);
    auto   npart_max_str = npart_max > 9999
                             ? fmt::format("%.2Le", (long double)npart_max)
                             : std::to_string(npart_max);
    os << "  species " << fmt::format("%2d", species_index) << " ("
       << species_label << ")";

    const auto c_bblack  = color::get_color("bblack", flags & Diag::Colorful);
    const auto c_red     = color::get_color("red", flags & Diag::Colorful);
    const auto c_yellow  = color::get_color("yellow", flags & Diag::Colorful);
    const auto c_green   = color::get_color("green", flags & Diag::Colorful);
    const auto c_reset   = color::get_color("reset", flags & Diag::Colorful);
    auto       c_loadmin = (load_min > 80) ? c_red
                                           : ((load_min > 50) ? c_yellow : c_green);
    auto       c_loadmax = (load_max > 80) ? c_red
                                           : ((load_max > 50) ? c_yellow : c_green);
    const auto raw1 = fmt::format("%s (%4.1f%%)", npart_min_str.c_str(), load_min);
    const auto raw2 = fmt::format("%s (%4.1f%%)", npart_max_str.c_str(), load_max);
    os << c_bblack
       << fmt::pad(tot_npart_str, 20, '.', false).substr(0, 20 - tot_npart_str.size())
       << c_reset << tot_npart_str;
    os << fmt::pad(raw1, 20, ' ', false).substr(0, 20 - raw1.size())
       << fmt::format("%s (%s%4.1f%%%s)",
                      npart_min_str.c_str(),
                      c_loadmin.c_str(),
                      load_min,
                      c_reset.c_str());
    os << fmt::pad(raw2, 20, ' ', false).substr(0, 20 - raw2.size())
       << fmt::format("%s (%s%4.1f%%%s)",
                      npart_max_str.c_str(),
                      c_loadmax.c_str(),
                      load_max,
                      c_reset.c_str());
#else // not MPI_ENABLED
    auto load      = 100.0 * (double)(npart) / (double)(maxnpart);
    auto npart_str = npart > 9999 ? fmt::format("%.2Le", (long double)npart)
                                  : std::to_string(npart);
    const auto c_bblack = color::get_color("bblack", flags & Diag::Colorful);
    const auto c_red    = color::get_color("red", flags & Diag::Colorful);
    const auto c_yellow = color::get_color("yellow", flags & Diag::Colorful);
    const auto c_green  = color::get_color("green", flags & Diag::Colorful);
    const auto c_reset  = color::get_color("reset", flags & Diag::Colorful);
    const auto c_load   = (load > 80)
                            ? c_red.c_str()
                            : ((load > 50) ? c_yellow.c_str() : c_green.c_str());
    os << "  species " << species_index << " (" << species_label << ")";
    const auto raw = fmt::format("%s (%4.1f%%)", npart_str.c_str(), load);
    os << c_bblack << fmt::pad(raw, 24, '.').substr(0, 24 - raw.size()) << c_reset;
    os << fmt::format("%s (%s%4.1f%%%s)",
                      npart_str.c_str(),
                      c_load,
                      load,
                      c_reset.c_str());
#endif
    os << std::endl;
  }

  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_1D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::Minkowski<Dim::_3D>>;
  template class Engine<SimEngine::SRPIC, metric::Spherical<Dim::_2D>>;
  template class Engine<SimEngine::SRPIC, metric::QSpherical<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::KerrSchild0<Dim::_2D>>;
  template class Engine<SimEngine::GRPIC, metric::QKerrSchild<Dim::_2D>>;
} // namespace ntt

//   template <Dimension D, SimulationEngine S>
//   auto Simulation<D, S>::PrintDiagnostics(const std::size_t&   step,
//                                           const real_t&        time,
//                                           const timer::Timers& timers,
//                                           std::vector<long double>& tstep_durations,
//                                           const DiagFlags diag_flags,
//                                           std::ostream&   os) -> void {
//     if (tstep_durations.size() > m_params.diagMaxnForPbar()) {
//       tstep_durations.erase(tstep_durations.begin());
//     }
//     tstep_durations.push_back(timers.get("Total"));
//     if (step % m_params.diagInterval() == 0) {
//       auto&      mblock = this->meshblock;
//       const auto title {
//         fmt::format("Time = %f : step = %d : Δt = %f", time, step, mblock.timestep())
//       };
//       PrintOnce(
//         [](std::ostream& os, std::string title) {
//           os << title << std::endl;
//         },
//         os,
//         title);
//       if (diag_flags & DiagFlags_Timers) {
//         timers.printAll("", timer::TimerFlags_Default, os);
//       }
//       if (diag_flags & DiagFlags_Species) {
//         auto header = fmt::format("%s %27s", "[SPECIES]", "[TOT]");
// #if defined(MPI_ENABLED)
//         header += fmt::format("%17s %s", "[MIN (%) :", "MAX (%)]");
// #endif
//         PrintOnce(
//           [](std::ostream& os, std::string header) {
//             os << header << std::endl;
//           },
//           os,
//           header);
//         for (const auto& species : meshblock.particles) {
//           species.PrintParticleCounts(os);
//         }
//       }
//       if (diag_flags & DiagFlags_Progress) {
//         PrintOnce(
//           [](std::ostream& os) {
//             os << std::setw(65) << std::setfill('-') << "" << std::endl;
//           },
//           os);
//         ProgressBar(tstep_durations, time, m_params.totalRuntime(), os);
//       }
//       PrintOnce(
//         [](std::ostream& os) {
//           os << std::setw(65) << std::setfill('=') << "" << std::endl;
//         },
//         os);
//     }
//   }

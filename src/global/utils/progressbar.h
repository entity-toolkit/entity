/**
 * @file utils/progressbar.h
 * @brief Progress bar for logging the simulation progress
 * @implements
 *   - pbar::ProgressBar
 * @namespaces:
 *   - pbar::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - remove outlying values from the average (excl. output)
 */

#ifndef GLOBAL_UTILS_PROGRESSBAR_H
#define GLOBAL_UTILS_PROGRESSBAR_H

#include <iostream>
#include <numeric>
#include <vector>

#include <string_view>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif // MPI_ENABLED

namespace pbar {
  namespace params {
    // averaging window (last 10 % of steps)
    inline constexpr int       average_last_pct { 10 };
    inline constexpr int       width { 52 };
    constexpr std::string_view fill { "■" };
    constexpr std::string_view empty { " " };
    constexpr std::string_view start { "[" };
    constexpr std::string_view end { "]" };
  } // namespace params

  void ProgressBar(const std::vector<long double>& durations,
                   const real_t&                   time,
                   const real_t&                   runtime,
                   std::ostream&                   os = std::cout) {
    // durations are in us (microseconds)
    const auto window = IMIN(
      IMAX((int)(durations.size() * params::average_last_pct / 100), 10),
      (int)durations.size());
    auto avg_duration = std::accumulate(durations.end() - window,
                                        durations.end(),
                                        0.0) /
                        static_cast<long double>(window);

#if defined(MPI_ENABLED)
    int rank, size, root_rank { 0 };
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<long double> mpi_durations(size, 0.0);
    MPI_Gather(&avg_duration,
               1,
               mpi_get_type<long double>(),
               mpi_durations.data(),
               1,
               mpi_get_type<long double>(),
               root_rank,
               MPI_COMM_WORLD);
    if (rank != root_rank) {
      return;
    }
    avg_duration = *std::max_element(mpi_durations.begin(), mpi_durations.end());
#endif
    auto avg_reduced = avg_duration;
    auto avg_units   = "us";
    if (avg_reduced > 1e6) {
      avg_reduced /= 1e6;
      avg_units    = "s";
    } else if (avg_reduced > 1e3) {
      avg_reduced /= 1e3;
      avg_units    = "ms";
    } else if (avg_reduced < 1e-2) {
      avg_reduced *= 1e3;
      avg_units    = "ns";
    }

    const auto remain_nsteps = static_cast<int>(
      (runtime - time) * durations.size() / time);
    auto remain_time  = static_cast<long double>(remain_nsteps * avg_duration);
    auto remain_units = "us";
    if (remain_time <= 0.0) {
      remain_time = 0.0;
    }
    if ((remain_time > 0.0) && (1e3 <= remain_time) && (remain_time < 1e6)) {
      remain_time  /= 1e3;
      remain_units  = "ms";
    } else if ((1e6 <= remain_time) && (remain_time < 6e7)) {
      remain_time  /= 1e6;
      remain_units  = "s";
    } else if ((6e7 <= remain_time) && (remain_time < 3.6e9)) {
      remain_time  /= 6e7;
      remain_units  = "min";
    } else if (3.6e9 <= remain_time) {
      remain_time  /= 3.6e9;
      remain_units  = "hr";
    }

    const int    nfilled = IMIN(time / runtime * params::width, params::width);
    const int    nempty  = params::width - nfilled;
    const real_t pct     = time / runtime * static_cast<real_t>(100.0);
    os << "Average timestep: " << avg_reduced << " " << avg_units << " (last "
       << window << " steps)" << std::endl;
    if (durations.size() > 1) {
      os << "Remaining time: " << remain_time << " " << remain_units << std::endl;
    }
    os << params::start;
    for (auto i { 0 }; i < nfilled; ++i) {
      os << params::fill;
    }
    for (auto i { 0 }; i < nempty; ++i) {
      os << params::empty;
    }
    os << params::end << " " << std::fixed << std::setprecision(2) << pct << "%\n";
  }

} // namespace pbar

#endif // GLOBAL_UTILS_PROGRESSBAR_H
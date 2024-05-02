/**
 * @file utils/progressbar.h
 * @brief Progress bar for logging the simulation progress
 * @implements
 *   - pbar::ProgressBar -> void
 * @namespaces:
 *   - pbar::
 * @macros:
 *   - MPI_ENABLED
 * !TODO:
 *   - remove outlying values from the average (excl. output)
 */

#ifndef GLOBAL_UTILS_PROGRESSBAR_H
#define GLOBAL_UTILS_PROGRESSBAR_H

#include "utils/colors.h"
#include "utils/error.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif // MPI_ENABLED

namespace pbar {
  namespace params {
    inline constexpr int       width { 70 };
    constexpr std::string_view fill { "■" };
    constexpr std::string_view empty { " " };
    constexpr std::string_view start { "[" };
    constexpr std::string_view end { "]" };
  } // namespace params

  class DurationHistory {
    std::size_t                                              capacity;
    std::vector<long double>                                 durations;
    const std::chrono::time_point<std::chrono::system_clock> start;
    std::chrono::time_point<std::chrono::system_clock>       prev_start;

  public:
    DurationHistory(std::size_t cap)
      : capacity { cap }
      , start { std::chrono::system_clock::now() }
      , prev_start { start } {}

    ~DurationHistory() = default;

    void tick() {
      const auto now = std::chrono::system_clock::now();
      if (durations.size() >= capacity) {
        durations.erase(durations.begin());
      }
      durations.push_back(
        std::chrono::duration_cast<std::chrono::microseconds>(now - prev_start).count());
      prev_start = now;
    }

    auto average() const -> long double {
      if (durations.size() > 0) {
        return std::accumulate(durations.begin(), durations.end(), 0.0) /
               static_cast<long double>(durations.size());
      } else {
        return 0.0;
      }
    }

    auto elapsed() const -> long double {
      return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::system_clock::now() - start)
        .count();
    }
  };

  inline auto normalize_duration_fmt(long double t, const std::string& u)
    -> std::pair<long double, std::string> {
    const std::vector<std::pair<std::string, long double>> units {
      {"µs",   1e0},
      { "ms",   1e3},
      {  "s",   1e6},
      {"min",   6e7},
      { "hr", 3.6e9}
    };
    auto it    = std::find_if(units.begin(), units.end(), [&u](const auto& pr) {
      return pr.first == u;
    });
    int  u_idx = (it != units.end()) ? std::distance(units.begin(), it) : -1;
    raise::ErrorIf(u_idx < 0, "Invalid unit", HERE);
    int shift = 0;
    if (t < 1e-2) {
      shift = -1;
    } else if (1e3 <= t && t < 1e6) {
      shift = 1;
    } else if (1e6 <= t && t < 6e7) {
      shift += 2;
    } else if (6e7 <= t && t < 3.6e9) {
      shift += 3;
    } else if (3.6e9 <= t) {
      shift += 4;
    }
    auto newu_idx = std::min(std::max(0, u_idx + shift),
                             static_cast<int>(units.size()));
    return { t * (units[u_idx].second / units[newu_idx].second),
             units[newu_idx].first };
  }

  inline void ProgressBar(const DurationHistory& history,
                          std::size_t            step,
                          std::size_t            max_steps,
                          DiagFlags&             flags,
                          std::ostream&          os = std::cout) {
    auto avg_duration = history.average();

#if defined(MPI_ENABLED)
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<long double> mpi_avg_durations(size, 0.0);
    MPI_Gather(&avg_duration,
               1,
               mpi::get_type<long double>(),
               mpi_avg_durations.data(),
               1,
               mpi::get_type<long double>(),
               MPI_ROOT_RANK,
               MPI_COMM_WORLD);
    if (rank != MPI_ROOT_RANK) {
      return;
    }
    avg_duration = *std::max_element(mpi_avg_durations.begin(),
                                     mpi_avg_durations.end());
#endif
    auto [avg_reduced, avg_units] = normalize_duration_fmt(avg_duration, "µs");

    const auto remain_nsteps         = max_steps - step;
    auto [remain_time, remain_units] = normalize_duration_fmt(
      static_cast<long double>(remain_nsteps) * avg_duration,
      "µs");
    auto [elapsed_time,
          elapsed_units] = normalize_duration_fmt(history.elapsed(), "µs");

    const auto pct = static_cast<long double>(step) /
                     static_cast<long double>(max_steps);
    const int nfilled = std::min(static_cast<int>(pct * params::width),
                                 params::width);
    const int nempty  = params::width - nfilled;
    const auto c_bmagenta = color::get_color("bmagenta", flags & Diag::Colorful);
    const auto c_reset = color::get_color("reset", flags & Diag::Colorful);
    os << "Average timestep: " << c_bmagenta << avg_reduced << " " << avg_units
       << c_reset << std::endl;
    os << "Remaining time: " << c_bmagenta << remain_time << " " << remain_units
       << c_reset << std::endl;
    os << "Elapsed time: " << c_bmagenta << elapsed_time << " " << elapsed_units
       << c_reset << std::endl;
    os << params::start;
    for (auto i { 0 }; i < nfilled; ++i) {
      os << params::fill;
    }
    for (auto i { 0 }; i < nempty; ++i) {
      os << params::empty;
    }
    os << params::end << " " << std::fixed << std::setprecision(2)
       << std::setfill(' ') << std::setw(6) << std::right << pct * 100.0 << "%\n";
  }

} // namespace pbar

#endif // GLOBAL_UTILS_PROGRESSBAR_H

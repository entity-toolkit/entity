/**
 * @file utils/timer.h
 * @brief Basic timekeeping functionality with fancy printing
 * @implements
 *   - timer::Timers
 *   - enum timer::TimerFlags
 * @namespces:
 *   - timer::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_UTILS_TIMER_H
#define GLOBAL_UTILS_TIMER_H

#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/colors.h"
#include "utils/comparators.h"
#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/numeric.h"

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace timer {
  using timestamp = std::chrono::time_point<std::chrono::system_clock>;

  inline void convertTime(long double& value, std::string& units) {
    if (value > 1e6) {
      value /= 1e6;
      units  = " s";
    } else if (value > 1e3) {
      value /= 1e3;
      units  = "ms";
    } else if (value < 1e-2) {
      value *= 1e3;
      units  = "ns";
    }
  }

  class Timers {
    std::map<std::string, std::pair<timestamp, long double>> m_timers;
    std::vector<std::string>                                 m_names;
    const bool                                               m_blocking;
    const std::function<void(void)>                          m_synchronize;

  public:
    Timers(std::initializer_list<std::string> names,
           const std::function<void(void)>&   synchronize = nullptr,
           const bool&                        blocking    = false)
      : m_blocking { blocking }
      , m_synchronize { synchronize } {
      raise::ErrorIf((synchronize == nullptr) && blocking,
                     "Synchronize function not provided",
                     HERE);
      for (const auto& name : names) {
        m_timers.insert({
          name,
          {std::chrono::system_clock::now(), 0.0}
        });
        m_names.push_back(name);
      }
    }

    ~Timers() = default;

    void start(const std::string& name) {
      m_timers[name].first = std::chrono::system_clock::now();
    }

    void stop(const std::string& name) {
      if (m_blocking) {
        m_synchronize();
#if defined(MPI_ENABLED)
        MPI_Barrier(MPI_COMM_WORLD);
#endif
      }
      auto end     = std::chrono::system_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
                       end - m_timers[name].first)
                       .count();
      m_timers[name].second += elapsed;
    }

    void reset(const std::string& name) {
      m_timers[name].second = 0.0;
    }

    void resetAll() {
      for (auto& [name, _] : m_timers) {
        reset(name);
      }
    }

    [[nodiscard]]
    auto get(const std::string& name) const -> long double {
      if (name == "Total") {
        long double total = 0.0;
        for (auto& timer : m_timers) {
          total += timer.second.second;
        }
        return total;
      } else {
        raise::ErrorIf(m_timers.find(name) == m_timers.end(), "Timer not found", HERE);
        return m_timers.at(name).second;
      }
    }

    void printAll(const TimerFlags flags = Timer::Default,
                  std::ostream&    os    = std::cout) const {
      std::string header = fmt::format("%s %27s", "[SUBSTEP]", "[DURATION]");

      const auto c_bblack = color::get_color("bblack", flags & Timer::Colorful);
      const auto c_reset  = color::get_color("reset", flags & Timer::Colorful);
      const auto c_byellow = color::get_color("byellow", flags & Timer::Colorful);
      const auto c_blue = color::get_color("blue", flags & Timer::Colorful);

      if (flags & Timer::PrintRelative) {
        header += "  [% TOT]";
      }
#if defined(MPI_ENABLED)
      header += "   [MIN : MAX]";
#endif
      header = c_bblack + header + c_reset;
      CallOnce(
        [](std::ostream& os, std::string header) {
          os << header << std::endl;
        },
        os,
        header);
#if defined(MPI_ENABLED)
      int rank, size;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      std::map<std::string, std::vector<long double>> mpi_timers {};
      // accumulate timers from MPI blocks
      for (auto& [name, timer] : m_timers) {
        mpi_timers[name] = std::vector<long double>(size, 0.0);
        MPI_Gather(&timer.second,
                   1,
                   mpi::get_type<long double>(),
                   mpi_timers[name].data(),
                   1,
                   mpi::get_type<long double>(),
                   MPI_ROOT_RANK,
                   MPI_COMM_WORLD);
      }
      if (rank != MPI_ROOT_RANK) {
        return;
      }
      long double total = 0.0;
      for (auto& [name, timer] : m_timers) {
        auto        timers = mpi_timers[name];
        long double tot    = std::accumulate(timers.begin(), timers.end(), 0.0);
        if (name != "Output") {
          total += tot;
        }
      }
      for (auto& [name, timers] : mpi_timers) {
        // compute min, max, mean
        long double min_time = *std::min_element(timers.begin(), timers.end());
        long double max_time = *std::max_element(timers.begin(), timers.end());
        long double mean_time = std::accumulate(timers.begin(), timers.end(), 0.0) /
                                size;
        std::string mean_units = "µs";
        const auto  min_pct    = mean_time > ZERO
                                   ? (int)(((mean_time - min_time) / mean_time) * 100.0)
                                   : 0;
        const auto  max_pct    = mean_time > ZERO
                                   ? (int)(((max_time - mean_time) / mean_time) * 100.0)
                                   : 0;
        const auto  tot_pct    = (cmp::AlmostZero_host(total)
                                    ? 0
                                    : (mean_time * size / total) * 100.0);
        if (flags & Timer::AutoConvert) {
          convertTime(mean_time, mean_units);
        }
        if (flags & Timer::PrintIndents) {
          os << "  ";
        }
        os << ((name != "Sorting" or flags & Timer::PrintSorting) ? c_reset
                                                                  : c_bblack)
           << name << c_reset << c_bblack
           << fmt::pad(name, 20, '.', true).substr(name.size(), 20);
        os << std::setw(17) << std::right << std::setfill('.')
           << fmt::format("%s%.2Lf",
                          (name != "Sorting" or flags & Timer::PrintSorting)
                            ? c_byellow.c_str()
                            : c_bblack.c_str(),
                          mean_time);
        if (flags & Timer::PrintUnits) {
          os << " " << mean_units << " ";
        }
        if (flags & Timer::PrintRelative) {
          os << "  " << std::setw(5) << std::right << std::setfill(' ')
             << std::fixed << std::setprecision(2) << tot_pct << "%";
        }
        os << fmt::format("%+7s : %-7s",
                          fmt::format("-%d%%", min_pct).c_str(),
                          fmt::format("+%d%%", max_pct).c_str());
        os << c_reset << std::endl;
      }
      total /= size;
#else  // not MPI_ENABLED
      long double total = 0.0;
      for (auto& [name, timer] : m_timers) {
        if (name != "Output") {
          total += timer.second;
        }
      }
      for (auto& [name, timer] : m_timers) {
        if (name == "Output") {
          continue;
        }
        std::string units = "µs";
        auto        value = timer.second;
        if (flags & Timer::AutoConvert) {
          convertTime(value, units);
        }
        if (flags & Timer::PrintIndents) {
          os << "  ";
        }
        os << ((name != "Sorting" or flags & Timer::PrintSorting) ? c_reset
                                                                  : c_bblack)
           << name << c_bblack
           << fmt::pad(name, 20, '.', true).substr(name.size(), 20);
        os << std::setw(17) << std::right << std::setfill('.')
           << fmt::format("%s%.2Lf",
                          (name != "Sorting" or flags & Timer::PrintSorting)
                            ? c_byellow.c_str()
                            : c_bblack.c_str(),
                          value);
        if (flags & Timer::PrintUnits) {
          os << " " << units;
        }
        if (flags & Timer::PrintRelative) {
          os << "  " << std::setw(7) << std::right << std::setfill(' ')
             << std::fixed << std::setprecision(2)
             << (cmp::AlmostZero_host(total) ? 0 : (timer.second / total) * 100.0);
        }
        os << c_reset << std::endl;
      }
#endif // MPI_ENABLED
      if (flags & Timer::PrintTotal) {
        std::string units = "µs";
        auto        value = total;
        if (flags & Timer::AutoConvert) {
          convertTime(value, units);
        }
        os << c_bblack << std::setw(22) << std::left << std::setfill(' ')
           << "Total" << c_reset;
        os << c_blue << std::setw(12) << std::right << std::setfill(' ') << value;
        if (flags & Timer::PrintUnits) {
          os << " " << units;
        }
        os << c_reset << std::endl;
      }
      {
        std::string units = "µs";
        auto        value = get("Output");
        if (flags & Timer::AutoConvert) {
          convertTime(value, units);
        }
        os << ((flags & Timer::PrintOutput) ? c_reset : c_bblack) << "Output"
           << c_bblack << fmt::pad("Output", 22, '.', true).substr(6, 22);
        os << std::setw(17) << std::right << std::setfill('.')
           << fmt::format("%s%.2Lf",
                          (flags & Timer::PrintOutput) ? c_byellow.c_str()
                                                       : c_bblack.c_str(),
                          value);
        if (flags & Timer::PrintUnits) {
          os << " " << units;
        }
        os << c_reset << std::endl;
      }
    }
  };
} // namespace timer

#endif // GLOBAL_UTILS_TIMER_H

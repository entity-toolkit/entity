/**
 * @file utils/timer.h
 * @brief Basic timekeeping functionality with fancy printing
 * @implements
 *   - timer::Timers
 *   - enum timer::TimerFlags
 * @depends:
 *   - utils/error.h
 *   - utils/mpi_aliases.h
 * @namespces:
 *   - timer::
 * @macros:
 *   - MPI_ENABLED
 */

#ifndef GLOBAL_UTILS_TIMER_H
#define GLOBAL_UTILS_TIMER_H

#include "utils/errors.h"
#include "utils/mpi_aliases.h"

#include <chrono>
#include <functional>
#include <iomanip>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <unordered_map>

namespace timer {
  using timestamp = std::chrono::time_point<std::chrono::system_clock>;

  enum TimerFlags_ {
    TimerFlags_None          = 0,
    TimerFlags_PrintRelative = 1 << 0,
    TimerFlags_PrintUnits    = 1 << 1,
    TimerFlags_PrintIndents  = 1 << 2,
    TimerFlags_PrintTotal    = 1 << 3,
    TimerFlags_PrintTitle    = 1 << 4,
    TimerFlags_AutoConvert   = 1 << 5,
    TimerFlags_All = TimerFlags_PrintRelative | TimerFlags_PrintUnits |
                     TimerFlags_PrintIndents | TimerFlags_PrintTotal |
                     TimerFlags_PrintTitle | TimerFlags_AutoConvert,
    TimerFlags_Default = TimerFlags_All,
    // TimerFlags_... = 1 << 5,
    // TimerFlags_... = 1 << 6,
    // TimerFlags_... = 1 << 7,
  };

  void convertTime(long double& value, std::string& units) {
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

  typedef int TimerFlags;

  class Timers {
    std::unordered_map<std::string, std::pair<timer::timestamp, long double>> m_timers;
    std::vector<std::string>        m_names;
    const bool                      m_blocking;
    const std::function<void(void)> m_syncronize;

  public:
    //
    Timers(std::initializer_list<std::string> names,
           const std::function<void(void)>&   synchronize = nullptr,
           const bool&                        blocking    = false) :
      m_syncronize { synchronize },
      m_blocking { blocking } {
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
        synchronize();
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

    [[nodiscard]]
    auto get(const std::string& name) const -> long double {
      if (name == "Total") {
        long double total = 0.0;
        for (auto& timer : m_timers) {
          total += timer.second.second;
        }
        return total;
      } else {
        return m_timers.at(name).second;
      }
    }

    void printAll(const std::string& title = "",
                  const TimerFlags   flags = TimerFlags_Default,
                  std::ostream&      os    = std::cout) const {
      if ((flags & TimerFlags_PrintTitle) && !title.empty()) {
        CallOnce(
          [](std::ostream& os, std::string title) {
            os << title << std::endl;
          },
          os,
          title);
      }
      std::string header = fmt::format("%s %27s", "[SUBSTEP]", "[DURATION]");
      if (flags & TimerFlags_PrintRelative) {
        header += "  [% TOT]";
      }
#if defined(MPI_ENABLED)
      header += "   [MIN : MAX]";
#endif
      CallOnce(
        [](std::ostream& os, std::string header) {
          os << header << std::endl;
        },
        os,
        header);
#if defined(MPI_ENABLED)
      int rank, size, root_rank { 0 };
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &size);
      std::map<std::string, std::vector<long double>> mpi_timers {};
      // accumulate timers from MPI blocks
      for (auto& timer : m_timers) {
        mpi_timers[timer.first] = std::vector<long double>(size, 0.0);
        MPI_Gather(&timer.second.second,
                   1,
                   mpi_get_type<long double>(),
                   mpi_timers[timer.first].data(),
                   1,
                   mpi_get_type<long double>(),
                   root_rank,
                   MPI_COMM_WORLD);
      }
      if (rank != root_rank) {
        return;
      }
      long double total = 0.0;
      for (auto& timer : m_timers) {
        auto label   = timer.first;
        auto timers  = mpi_timers[label];
        total       += std::accumulate(timers.begin(), timers.end(), 0.0);
      }
      for (std::size_t t { 0 }; t < m_names.size(); ++t) {
        auto        label    = m_names[t];
        auto        timers   = mpi_timers[label];
        // compute min, max, mean
        long double min_time = *std::min_element(timers.begin(), timers.end());
        long double max_time = *std::max_element(timers.begin(), timers.end());
        long double mean_time = std::accumulate(timers.begin(), timers.end(), 0.0) /
                                size;
        std::string mean_units = "us";
        const auto  min_pct    = mean_time > ZERO
                                   ? (int)(((mean_time - min_time) / mean_time) * 100.0)
                                   : 0;
        const auto  max_pct    = mean_time > ZERO
                                   ? (int)(((max_time - mean_time) / mean_time) * 100.0)
                                   : 0;
        const auto  tot_pct    = ((mean_time * size) / total * 100.0);
        if (flags & TimerFlags_AutoConvert) {
          convertTime(mean_time, mean_units);
        }
        if (flags & TimerFlags_PrintIndents) {
          os << "  ";
        }
        os << std::setw(20) << std::left << std::setfill('.') << label;
        os << std::setw(12) << std::right << std::setfill('.')
           << fmt::format("%.2Lf", mean_time);
        if (flags & TimerFlags_PrintUnits) {
          os << " " << mean_units << " ";
        }
        if (flags & TimerFlags_PrintRelative) {
          os << " | " << std::setw(5) << std::right << std::setfill(' ')
             << std::fixed << std::setprecision(2) << tot_pct << "%";
        }
        os << fmt::format("%+7s : %-7s",
                          fmt::format("-%d%%", min_pct).c_str(),
                          fmt::format("+%d%%", max_pct).c_str());
        os << std::endl;
      }
      total /= size;
#else  // not MPI_ENABLED
      long double total = 0.0;
      for (auto& timer : m_timers) {
        total += timer.second.second;
      }
      for (std::size_t t { 0 }; t < m_names.size(); ++t) {
        auto&       timer = m_timers.at(m_names[t]);
        std::string units = "us";
        auto        value = timer.second;
        if (flags & TimerFlags_AutoConvert) {
          convertTime(value, units);
        }
        if (flags & TimerFlags_PrintIndents) {
          os << "  ";
        }
        os << std::setw(20) << std::left << std::setfill('.') << m_names[t];
        os << std::setw(12) << std::right << std::setfill('.') << value;
        if (flags & TimerFlags_PrintUnits) {
          os << " " << units;
        }
        if (flags & TimerFlags_PrintRelative) {
          os << " | " << std::setw(5) << std::right << std::setfill(' ')
             << std::fixed << std::setprecision(2)
             << (timer.second / total) * 100.0 << "%";
        }
        os << std::endl;
      }
#endif // MPI_ENABLED
      if (flags & TimerFlags_PrintTotal) {
        std::string units = "us";
        auto        value = total;
        if (flags & TimerFlags_AutoConvert) {
          convertTime(value, units);
        }
        os << std::setw(22) << std::left << std::setfill(' ') << "Total";
        os << std::setw(12) << std::right << std::setfill(' ') << value;
        if (flags & TimerFlags_PrintUnits) {
          os << " " << units;
        }
        os << std::endl;
      }
    }
  };
} // namespace timer

#endif // GLOBAL_UTILS_TIMER_H
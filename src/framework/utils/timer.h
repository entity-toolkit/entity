#ifndef FRAMEWORK_UTILS_TIMER_H
#define FRAMEWORK_UTILS_TIMER_H

#include <iostream>
#include <string>
#include <chrono>

using timestamp = std::chrono::time_point<std::chrono::system_clock>;

namespace ntt {
  namespace timer {
    enum TimerFlags_ {
      TimerFlags_None          = 0,
      TimerFlags_PrintRelative = 1 << 0,
      TimerFlags_PrintUnits    = 1 << 1,
      TimerFlags_PrintIndents  = 1 << 2,
      TimerFlags_PrintTotal    = 1 << 3,
      TimerFlags_PrintTitle    = 1 << 4,
      TimerFlags_AutoConvert   = 1 << 5,
      TimerFlags_All           = TimerFlags_PrintRelative | TimerFlags_PrintUnits
                       | TimerFlags_PrintIndents | TimerFlags_PrintTotal
                       | TimerFlags_PrintTitle | TimerFlags_AutoConvert,
      TimerFlags_Default = TimerFlags_All,
      // TimerFlags_... = 1 << 5,
      // TimerFlags_... = 1 << 6,
      // TimerFlags_... = 1 << 7,
    };
    typedef int TimerFlags;

    class Timers {
    public:
      Timers(std::initializer_list<std::string> names) {
        for (auto& name : names) {
          m_timers.insert({name, {std::chrono::system_clock::now(), 0.0}});
        }
      }
      ~Timers() = default;
      void start(const std::string& name) {
        m_timers[name].first = std::chrono::system_clock::now();
      }
      void stop(const std::string& name) {
        auto end = std::chrono::system_clock::now();
        auto elapsed
          = std::chrono::duration_cast<std::chrono::microseconds>(end - m_timers[name].first)
              .count();
        m_timers[name].second += elapsed;
      }
      void reset(const std::string& name) { m_timers[name].second = 0.0; }

      void printAll(const std::string& title = "",
                    const TimerFlags   flags = TimerFlags_Default,
                    std::ostream&      os    = std::cout) const {
        os << std::setw(46) << std::setfill('-') << "" << std::endl;
        if ((flags & TimerFlags_PrintTitle) && !title.empty()) { os << title << std::endl; }
        long double total = 0.0;
        for (auto& timer : m_timers) {
          total += timer.second.second;
        }
        for (auto& timer : m_timers) {
          std::string units = "us";
          auto        value = timer.second.second;
          if (flags & TimerFlags_AutoConvert) {
            if (value > 1e6) {
              value /= 1e6;
              units = "s";
            } else if (value > 1e3) {
              value /= 1e3;
              units = "ms";
            } else if (value < 1e-2) {
              value *= 1e3;
              units = "ns";
            }
          }
          if (flags & TimerFlags_PrintIndents) { os << "  "; }
          os << std::setw(20) << std::left << std::setfill('.') << timer.first;
          os << std::setw(12) << std::right << std::setfill('.') << value;
          if (flags & TimerFlags_PrintUnits) { os << " " << units; }
          if (flags & TimerFlags_PrintRelative) {
            os << " | " << std::setw(5) << std::right << std::setfill(' ') << std::fixed
               << std::setprecision(2) << (timer.second.second / total) * 100.0 << "%";
          }
          os << std::endl;
        }
        if (flags & TimerFlags_PrintTotal) {
          std::string units = "us";
          auto        value = total;
          if (flags & TimerFlags_AutoConvert) {
            if (value > 1e6) {
              value /= 1e6;
              units = "s";
            } else if (value > 1e3) {
              value /= 1e3;
              units = "ms";
            } else if (value < 1e-2) {
              value *= 1e3;
              units = "ns";
            }
          }
          os << std::setw(22) << std::left << std::setfill(' ') << "Total";
          os << std::setw(12) << std::right << std::setfill(' ') << value;
          if (flags & TimerFlags_PrintUnits) { os << " " << units; }
          os << std::endl;
        }
      }

    private:
      std::map<std::string, std::pair<timestamp, long double>> m_timers;
    };
  } // namespace timer
} // namespace ntt

#endif // FRAMEWORK_UTILS_TIMER_H
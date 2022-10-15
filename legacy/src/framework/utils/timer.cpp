#include "wrapper.h"
#include "timer.h"

#include <string>
#include <cassert>
#include <utility>
#include <vector>
#include <iostream>
#include <iomanip>

namespace ntt {

  auto TimeUnit::getMultiplier() const -> double { return multiplier; }
  auto operator<<(std::ostream& os, const TimeUnit& v) -> std::ostream& {
    return os << v.unitname;
  }

  Time::Time(const long double& v, const TimeUnit& u) {
    value = static_cast<long double>(v);
    unit  = &u;
  }
  // auto Time::getValue() const -> long double { return value; }
  // void Time::convert(const TimeUnit& to) {
  //   if (&to != unit) {
  //     value = value * unit->getMultiplier() / to.getMultiplier();
  //     unit  = &to;
  //   }
  // }
  auto Time::represent(const TimeUnit& to) const -> Time {
    if (&to != unit) {
      return Time(value * unit->getMultiplier() / to.getMultiplier(), to);
    } else {
      return Time(value, to);
    }
  }
  auto operator<<(std::ostream& os, const Time& t) -> std::ostream& {
    return os << t.value << " " << *(t.unit);
  }
  // auto Time::operator-() const -> Time { return Time(-(this->value), *(this->unit)); }
  // auto       operator+(const Time& t1, const Time& t2) -> Time {
  //   if (t1.unit == t2.unit) {
  //           return Time(static_cast<long double>(t1.value + t2.value), *(t1.unit));
  //   } else {
  //           const TimeUnit* main_unit;
  //           if (t1.unit->getMultiplier() < t2.unit->getMultiplier())
  //       main_unit = t1.unit;
  //     else
  //       main_unit = t2.unit;
  //     return Time(static_cast<long double>(t1.represent(*main_unit).value
  //                                          + t2.represent(*main_unit).value),
  //                 *(main_unit));
  //   }
  // }
  // auto operator-(const Time& t1, const Time& t2) -> Time { return (t1 + (-t2)); }
  // auto operator*(double x, const Time& t) -> Time { return Time(t.value * x, *(t.unit)); }
  // auto operator*(const Time& t, double x) -> Time { return Time(t.value * x, *(t.unit)); }

  namespace { // anonymous namespace
    void timeNow(TimeContainer& time) { time = std::chrono::system_clock::now(); }
    void timeElapsed(TimeContainer& time_start, Time& time_elapsed) {
      long double dt;
      dt = std::chrono::duration<long double>(std::chrono::system_clock::now() - time_start)
             .count();
      time_elapsed = Time(dt, second);
    }
  } // namespace

  void Timer::start() {
    init = true;
    on   = true;
    timeNow(t_start);
  }
  void Timer::check() {
    assert(init && "# Error: timer is not initialized.");
    assert(on && "# Error: timer is not running.");
    timeElapsed(t_start, t_elapsed);
  }
  void Timer::stop() {
    check();
    on = false;
  }
  auto Timer::getElapsedIn(const TimeUnit& u) const -> long double {
    if (init) {
      return t_elapsed.represent(u).value;
    } else {
      return 0.0;
    }
  }
  auto Timer::getName() const -> std::string { return name; }
  void Timer::printElapsed(std::ostream& os, const TimeUnit& u) const {
    auto repr = t_elapsed.represent(u);
    os << std::setw(25) << std::left << "timer `" + name + "`"
       << ": " << repr;
    if (on) { os << " (and running)"; }
  }
  void Timer::printElapsed(const TimeUnit& u) const { printElapsed(std::cout, u); }

  // TimerCollection::TimerCollection(const std::vector<std::string>& timers) {
  //   for (auto& t : timers) {
  //     m_timers.push_back(Timer(t));
  //     // m_timers.emplace_back(Timer(t));
  //   }
  // }

  void TimerCollection::start(const int& i) { m_timers[i - 1].start(); }

  void TimerCollection::stop(const int& i) { m_timers[i - 1].stop(); }

  void TimerCollection::printAll(std::ostream& os, const TimeUnit& u) const {
    os << "==============================" << std::endl;
    for (auto& t : m_timers) {
      t.printElapsed(os, u);
      os << std::endl;
    }
    os << "------------------------------" << std::endl;
  }

  void TimerCollection::printAll(const TimeUnit& u) const { printAll(std::cout, u); }

} // namespace ntt

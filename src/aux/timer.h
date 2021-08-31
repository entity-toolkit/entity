#ifndef AUX_TIMER_H
#define AUX_TIMER_H

#include <string>
#ifndef _OPENMP
#  include <chrono>
#else
#  include <omp.h>
#endif

#include <string>
#include <cassert>
#include <utility>
#include <iostream>

namespace ntt::timer {
#ifndef _OPENMP
inline constexpr char BACKEND[] = "Chrono";
#else
inline constexpr char BACKEND[] = "OpenMP";
#endif
// Type to be used for s/ms/us/ms
class TimeUnit {
private:
  double multiplier;
  std::string unitname;

public:
  TimeUnit() = default;
  TimeUnit(double mult, std::string unit)
      : multiplier(static_cast<double>(mult)), unitname(std::move(unit)) {}
  ~TimeUnit() = default;
  [[nodiscard]] auto getMultiplier() const -> double;
  friend auto operator<<(std::ostream &os, TimeUnit const &v) -> std::ostream &;
};
auto TimeUnit::getMultiplier() const -> double { return multiplier; }
auto operator<<(std::ostream &os, TimeUnit const &v) -> std::ostream & {
  return os << v.unitname;
}

// declaration of s/ms/us/ms
inline const TimeUnit second(1, "s");
inline const TimeUnit millisecond(1e-3, "ms");
inline const TimeUnit microsecond(1e-6, "us");
inline const TimeUnit nanosecond(1e-9, "ns");

// Type to keep track of timestamp
class Time {
private:
  long double value;
  const TimeUnit *unit;

public:
  Time() = default;
  Time(long double v, TimeUnit const &u);
  ~Time() = default;
  [[nodiscard]] auto getValue() const -> long double;
  void convert(const TimeUnit to);
  [[nodiscard]] auto represent(const TimeUnit to) const -> Time;
  // Time operator=(const Time & rhs);
  auto operator-() const -> Time;

  friend auto operator+(Time const &, Time const &) -> Time;
  friend auto operator-(Time const &, Time const &) -> Time;
  friend auto operator*(double x, Time const &t) -> Time;
  friend auto operator*(Time const &, double x) -> Time;
  friend auto operator<<(std::ostream &os, Time const &t) -> std::ostream &;
  friend class Timer;
};
Time::Time(long double v, TimeUnit const &u) {
  value = static_cast<long double>(v);
  unit = &u;
}
auto Time::getValue() const -> long double { return value; }
void Time::convert(const TimeUnit to) {
  if (&to != unit) {
    value = value * unit->getMultiplier() / to.getMultiplier();
    unit = &to;
  }
}
auto Time::represent(const TimeUnit to) const -> Time {
  if (&to != unit) {
    return Time(value * unit->getMultiplier() / to.getMultiplier(), to);
  } else {
    return Time(value, to);
  }
}
auto operator<<(std::ostream &os, Time const &t) -> std::ostream & {
  return os << t.value << " " << *(t.unit);
}
// Time Time::operator=(const Time & rhs) {
// if(this == &rhs)
// return *this;
// else {
// value = rhs.value;
// unit = rhs.unit;
// return *this;
// }
// }
auto Time::operator-() const -> Time {
  return Time(-(this->value), *(this->unit));
}
auto operator+(Time const &t1, Time const &t2) -> Time {
  if (t1.unit == t2.unit) {
    return Time(static_cast<long double>(t1.value + t2.value), *(t1.unit));
  } else {
    const TimeUnit *main_unit;
    if (t1.unit->getMultiplier() < t2.unit->getMultiplier())
      main_unit = t1.unit;
    else
      main_unit = t2.unit;
    return Time(static_cast<long double>(t1.represent(*main_unit).value +
                                         t2.represent(*main_unit).value),
                *(main_unit));
  }
}
auto operator-(Time const &t1, Time const &t2) -> Time { return (t1 + (-t2)); }
auto operator*(double x, Time const &t) -> Time {
  return Time(t.value * x, *(t.unit));
}
auto operator*(Time const &t, double x) -> Time {
  return Time(t.value * x, *(t.unit));
}

#ifndef _OPENMP
using TimeContainer = std::chrono::time_point<std::chrono::system_clock>;
#else
using TimeContainer = Time;
#endif

namespace { // anonymous namespace
// Timing implementation for different libraries ...
// ... various implementation are brought to a standard here
void timeNow(TimeContainer &time) {
#ifndef _OPENMP
  // use `chrono`
  time = std::chrono::system_clock::now();
#else
  // use OpenMP `wtime` function
  time = Time(omp_get_wtime(), second);
#endif
}
void timeElapsed(TimeContainer &time_start, Time &time_elapsed) {
#ifndef _OPENMP
  // use `chrono`
  long double dt;
  dt = std::chrono::duration<long double>(std::chrono::system_clock::now() -
                                          time_start)
           .count();
  time_elapsed = Time(dt, second);
#else
  // use OpenMP `wtime` function
  time_elapsed = Time(omp_get_wtime(), second) - time_start;
#endif
}
} // namespace

class Timer {
private:
  bool init = false;
  bool on = false;
  TimeContainer t_start;
  Time t_elapsed;
  std::string name;

public:
  Timer() : name("NULL") {}
  Timer(std::string name) : name(std::move(name)) {}
  ~Timer() = default;
  void start();
  void check();
  void stop();
  [[nodiscard]] auto getElapsedIn(TimeUnit const &u) const -> long double;
  [[nodiscard]] auto getName() const -> std::string;
  void printElapsed(TimeUnit const &u) const;
  void printElapsed() const;
};
void Timer::start() {
  init = true;
  on = true;
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
auto Timer::getElapsedIn(TimeUnit const &u) const -> long double {
  assert(init && "# Error: timer is not initialized.");
  return t_elapsed.represent(u).value;
}
auto Timer::getName() const -> std::string { return name; }
void Timer::printElapsed(TimeUnit const &u) const {
  assert(init && "# Error: timer is not initialized.");
  std::cout << "timer `" << name << "` : " << t_elapsed.represent(u);
  if (on)
    std::cout << " (and running)";
  std::cout << "\n";
}
void Timer::printElapsed() const { Timer::printElapsed(second); }

} // namespace ntt::timer

#endif // TIMER_H

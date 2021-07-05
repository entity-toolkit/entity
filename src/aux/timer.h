#ifndef AUX_TIMER_H
#define AUX_TIMER_H

#include <string>
#ifndef _OPENMP
#  include <chrono>
#endif

#include <utility>

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

#ifndef _OPENMP
using TimeContainer = std::chrono::time_point<std::chrono::system_clock>;
#else
using TimeContainer = Time;
#endif

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
} // namespace ntt::timer

#endif // TIMER_H

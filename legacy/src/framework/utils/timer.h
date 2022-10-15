#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include <string>
#include <chrono>
#include <iostream>
#include <vector>

namespace ntt {
  // Type to be used for s/ms/us/ms
  class TimeUnit {
  private:
    double      multiplier;
    std::string unitname;

  public:
    TimeUnit() = default;
    TimeUnit(const double& mult, const std::string& unit)
      : multiplier {mult}, unitname {unit} {}
    ~TimeUnit() = default;
    [[nodiscard]] auto getMultiplier() const -> double;
    friend auto        operator<<(std::ostream& os, TimeUnit const& v) -> std::ostream&;
  };

  // declaration of s/ms/us/ms
  inline const TimeUnit second(1, "s");
  inline const TimeUnit millisecond(1e-3, "ms");
  inline const TimeUnit microsecond(1e-6, "us");
  inline const TimeUnit nanosecond(1e-9, "ns");

  // Type to keep track of timestamp
  class Time {
  private:
    long double     value {0.0};
    const TimeUnit* unit;

  public:
    Time() = default;
    Time(const long double& v, const TimeUnit& u = second);
    ~Time() = default;
    // void               convert(const TimeUnit& to);
    // [[nodiscard]] auto getValue() const -> long double;
    [[nodiscard]] auto represent(const TimeUnit& to) const -> Time;
    // Time operator=(const Time & rhs);
    auto operator-() const -> Time;

    // friend auto operator+(Time const&, Time const&) -> Time;
    // friend auto operator-(Time const&, Time const&) -> Time;
    // friend auto operator*(double x, Time const& t) -> Time;
    // friend auto operator*(Time const&, double x) -> Time;
    friend auto operator<<(std::ostream& os, Time const& t) -> std::ostream&;
    friend class Timer;
  };

  using TimeContainer = std::chrono::time_point<std::chrono::system_clock>;

  class Timer {
  private:
    std::string   name;
    bool          init {false};
    bool          on {false};
    TimeContainer t_start;
    Time          t_elapsed;

  public:
    Timer() : name("NULL"), t_elapsed {0.0} {}
    Timer(const std::string& name) : name(std::move(name)), t_elapsed {0.0} {}
    ~Timer() = default;
    void               start();
    void               check();
    void               stop();
    [[nodiscard]] auto getElapsedIn(const TimeUnit& u) const -> long double;
    [[nodiscard]] auto getName() const -> std::string;
    void               printElapsed(const TimeUnit& u = second) const;
    void printElapsed(std::ostream& os = std::cout, const TimeUnit& u = second) const;
  };

  class TimerCollection {
  private:
    // TODO: maybe map?
    std::vector<Timer> m_timers;

  public:
    // TimerCollection(const std::vector<std::string>& timers);
    TimerCollection(const std::vector<Timer>& timers) : m_timers(timers) {}
    ~TimerCollection() = default;
    void start(const int& i);
    void stop(const int& i);
    void printAll(std::ostream& os = std::cout, const TimeUnit& u = second) const;
    void printAll(const TimeUnit& u = second) const;
  };

} // namespace ntt

#endif // TIMER_H

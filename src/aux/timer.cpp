#include "timer.h"

#ifdef _OPENMP
# include <omp.h>
#endif

#include <string>
#include <cassert>
#include <iostream>

namespace ntt {
  namespace timer {
    namespace { // anonymous namespace
      // Timing implementation for different libraries ...
      // ... various implementation are brought to a standard here
      void timeNow(TimeContainer& time) {
        #ifndef _OPENMP
          // use `chrono`
          time = std::chrono::system_clock::now();
        #else
          // use OpenMP `wtime` function
          time = Time(omp_get_wtime(), second);
        #endif
      }
      void timeElapsed(TimeContainer& time_start, Time& time_elapsed) {
        #ifndef _OPENMP
          // use `chrono`
          long double dt;
          dt = std::chrono::duration<long double>(std::chrono::system_clock::now() - time_start).count();
          time_elapsed = Time(dt, second);
        #else
          // use OpenMP `wtime` function
          time_elapsed = Time(omp_get_wtime(), second) - time_start;
        #endif
      }
    }

  // `TimeUnit` class -->
    double TimeUnit::getMultiplier() const {
      return multiplier;
    }
    std::ostream& operator<<(std::ostream& os, TimeUnit const& v) {
      return os << v.unitname;
    }
  // <-- `TimeUnit` class

  // `Time` class -->
    Time::Time(long double v, TimeUnit const& u)  {
      value = static_cast<long double>(v);
      unit = &u;
    }
    long double Time::getValue() const {
      return value;
    }
    void Time::convert(const TimeUnit to) {
      if (&to != unit) {
        value = value * unit->getMultiplier() / to.getMultiplier();
        unit = &to;
      }
    }
    Time Time::represent(const TimeUnit to) const {
      if (&to != unit) {
        return Time(value * unit->getMultiplier() / to.getMultiplier(), to);
      } else {
        return Time(value, to);
      }
    }
    std::ostream& operator<<(std::ostream& os, Time const& t) {
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
    Time Time::operator-() const {
      return Time(-(this->value), *(this->unit));
    }
    Time operator+(Time const &t1, Time const &t2) {
      if (t1.unit == t2.unit) {
        return Time(static_cast<long double>(t1.value + t2.value), *(t1.unit));
      } else {
        const TimeUnit * main_unit;
        if (t1.unit->getMultiplier() < t2.unit->getMultiplier())
          main_unit = t1.unit;
        else
          main_unit = t2.unit;
        t1.represent(*main_unit);
        return Time(static_cast<long double>(t1.represent(*main_unit).value + t2.represent(*main_unit).value), *(main_unit));
      }
    }
    Time operator-(Time const &t1, Time const &t2) {
      return (t1 + (-t2));
    }
    Time operator*(double x, Time const &t) {
      return Time(t.value * x, *(t.unit));
    }
    Time operator*(Time const &t, double x) {
      return Time(t.value * x, *(t.unit));
    }
  // <-- `Time` class

  // `Timer` class -->
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
    long double Timer::getElapsedIn(TimeUnit const &u) const {
      assert(init && "# Error: timer is not initialized.");
      return t_elapsed.represent(u).value;
    }
    std::string Timer::getName() const {
      return name;
    }
    void Timer::printElapsed(TimeUnit const &u) const {
      assert(init && "# Error: timer is not initialized.");
      std::cout << "timer `"  << name << "` : " << t_elapsed.represent(u);
      if (on)
        std::cout << " (and running)";
      std::cout << "\n";
    }
    void Timer::printElapsed() const {
      Timer::printElapsed(second);
    }
  // <-- `Timer` class
  }
}

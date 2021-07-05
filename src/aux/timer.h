#ifndef AUX_TIMER_H
#define AUX_TIMER_H

#include <string>
#ifndef _OPENMP
# include <chrono>
#endif

namespace ntt {
  namespace timer {
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
      TimeUnit(TimeUnit const &u) : multiplier(u.multiplier), unitname(u.unitname) {};
      TimeUnit(double mult, const std::string& unit) : multiplier(static_cast<double>(mult)), unitname(unit) {};
      ~TimeUnit() = default;
      double getMultiplier() const;
      friend std::ostream& operator<<(std::ostream& os, TimeUnit const& v);
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
      const TimeUnit * unit;
    public:
      Time() {};
      Time(long double v, TimeUnit const& u);
      ~Time() = default;
      long double getValue() const;
      void convert(const TimeUnit to);
      Time represent(const TimeUnit to) const;
      // Time operator=(const Time & rhs);
      Time operator-() const;

      friend Time operator+(Time const &, Time const &);
      friend Time operator-(Time const &, Time const &);
      friend Time operator*(double x, Time const &t);
      friend Time operator*(Time const &, double x);
      friend std::ostream& operator<<(std::ostream& os, Time const& t);
      friend class Timer;
    };

    #ifndef _OPENMP
      typedef std::chrono::time_point<std::chrono::system_clock> TimeContainer;
    #else
      typedef Time TimeContainer;
    #endif

    class Timer {
    private:
      bool init = false;
      bool on = false;
      TimeContainer t_start;
      Time t_elapsed;
      std::string name;
    public:
      Timer() : name("NULL") {};
      Timer(const std::string& name) : name(name) {};
      ~Timer() = default;
      void start();
      void check();
      void stop();
      long double getElapsedIn(TimeUnit const &u) const;
      std::string getName() const;
      void printElapsed(TimeUnit const &u) const;
      void printElapsed() const;
    };
  }
}

#endif // TIMER_H

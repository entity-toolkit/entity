/**
 * @file output/stats.h
 * @brief Class defining the metadata necessary to prepare the stats for output
 * @implements
 *   - out::OutputStats
 *   - out::Writer
 * @cpp:
 *   - stats.cpp
 * @namespaces:
 *   - stats::
 */

#ifndef OUTPUT_STATS_H
#define OUTPUT_STATS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/formatting.h"
#include "utils/tools.h"

#if defined(MPI_ENABLED)
  #include "arch/mpi_aliases.h"

  #include <mpi.h>
#endif

#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

using namespace ntt;

namespace stats {

  class OutputStats {
    const std::string m_name;
    StatsID           m_id { StatsID::INVALID };

  public:
    std::vector<std::vector<unsigned short>> comp {};
    std::vector<spidx_t>                     species {};

    OutputStats(const std::string&, bool);

    ~OutputStats() = default;

    [[nodiscard]]
    auto is_moment() const -> bool {
      return (id() == StatsID::T || id() == StatsID::Rho || id() == StatsID::Npart ||
              id() == StatsID::N || id() == StatsID::Charge);
    }

    [[nodiscard]]
    auto is_vector() const -> bool {
      return id() == StatsID::ExB || id() == StatsID::E2 || id() == StatsID::B2;
    }

    [[nodiscard]]
    auto is_custom() const -> bool {
      return id() == StatsID::Custom;
    }

    [[nodiscard]]
    inline auto name() const -> std::string {
      if (id() == StatsID::Custom) {
        return m_name;
      }
      // generate the name
      std::string tmp = std::string(id().to_string());
      if (tmp == "exb") {
        tmp = "ExB";
      } else if (tmp == "j.e") {
        tmp = "J.E";
      } else {
        // capitalize the first letter
        tmp[0] = std::toupper(tmp[0]);
      }
      if (id() == StatsID::T) {
        tmp += m_name.substr(1, 2);
      } else if (is_vector()) {
        if (id() == StatsID::E2 || id() == StatsID::B2) {
          tmp = fmt::format("%ci^2", tmp[0]);
        } else {
          tmp += "i";
        }
      }
      if (species.size() > 0) {
        tmp += "_";
        for (auto& s : species) {
          tmp += std::to_string(s);
          tmp += "_";
        }
        tmp.pop_back();
      }
      return tmp;
    }

    [[nodiscard]]
    inline auto name(std::size_t ci) const -> std::string {
      raise::ErrorIf(
        comp.size() == 0,
        "OutputField::name(ci) called but no components were available",
        HERE);
      raise::ErrorIf(
        ci >= comp.size(),
        "OutputField::name(ci) called with an invalid component index",
        HERE);
      raise::ErrorIf(
        comp[ci].size() == 0,
        "OutputField::name(ci) called but no components were available",
        HERE);
      // generate the name
      auto tmp = std::string(id().to_string());
      // capitalize the first letter
      if (tmp == "exb") {
        tmp = "ExB";
      } else {
        // capitalize the first letter
        tmp[0] = std::toupper(tmp[0]);
      }
      if (tmp == "E^2" or tmp == "B^2") {
        tmp = fmt::format("%c%d^2", tmp[0], comp[ci][0]);
      } else {
        for (auto& c : comp[ci]) {
          tmp += std::to_string(c);
        }
        if (species.size() > 0) {
          tmp += "_";
          for (auto& s : species) {
            tmp += std::to_string(s);
            tmp += "_";
          }
          tmp.pop_back();
        }
      }
      return tmp;
    }

    [[nodiscard]]
    auto id() const -> StatsID {
      return m_id;
    }
  };

  class Writer {
    std::string              m_fname;
    std::vector<OutputStats> m_stat_writers;
    tools::Tracker           m_tracker;

  public:
    Writer() {}

    ~Writer() = default;

    Writer(Writer&&) = default;

    void init(timestep_t, simtime_t);
    void defineStatsFilename(const std::string&);
    void defineStatsOutputs(const std::vector<std::string>&, bool);

    void writeHeader();

    [[nodiscard]]
    auto shouldWrite(timestep_t, simtime_t) -> bool;

    template <typename T>
    inline void write(const T& value) const {
      auto tot_value { static_cast<T>(0) };
#if defined(MPI_ENABLED)
      MPI_Reduce(&value,
                 &tot_value,
                 1,
                 mpi::get_type<T>(),
                 MPI_SUM,
                 MPI_ROOT_RANK,
                 MPI_COMM_WORLD);
#else
      tot_value = value;
#endif
      CallOnce(
        [](auto&& fname, auto&& value) {
          std::fstream StatsOut(fname, std::fstream::out | std::fstream::app);
          StatsOut << std::setw(14) << value << ",";
          StatsOut.close();
        },
        m_fname,
        tot_value);
    }

    void endWriting();

    [[nodiscard]]
    auto statsWriters() const -> const std::vector<OutputStats>& {
      return m_stat_writers;
    }
  };

} // namespace stats

#endif // OUTPUT_STATS_H

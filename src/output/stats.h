/**
 * @file output/stats.h
 * @brief Class defining the metadata necessary to prepare the stats for output
 * @implements
 *   - out::OutputStats
 * @cpp:
 *   - stats.cpp
 * @namespaces:
 *   - out::
 */

#ifndef OUTPUT_STATS_H
#define OUTPUT_STATS_H

#include "enums.h"
#include "global.h"

#include "utils/error.h"
#include "utils/tools.h"

#include <fstream>
#include <iostream>
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

    OutputStats(const std::string&);

    ~OutputStats() = default;

    [[nodiscard]]
    auto is_moment() const -> bool {
      return (id() == StatsID::T || id() == StatsID::Rho || id() == StatsID::Npart ||
              id() == StatsID::N || id() == StatsID::Charge);
    }

    [[nodiscard]]
    auto is_vector() const -> bool {
      return id() == StatsID::ExB;
    }

    [[nodiscard]]
    inline auto name() const -> std::string {
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
        tmp += "i";
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
    void defineStatsOutputs(const std::vector<std::string>&);

    void writeHeader();

    [[nodiscard]]
    auto shouldWrite(timestep_t, simtime_t) -> bool;

    template <typename T>
    inline void write(const T& value) const {
#if defined(MPI_ENABLED)
        // @TODO: reduce
#endif
      CallOnce(
        [](auto& fname, auto& value) {
          std::fstream StatsOut(fname, std::fstream::out | std::fstream::app);
          StatsOut << value << ",";
          StatsOut.close();
        },
        m_fname,
        value);
    }

    void endWriting();

    [[nodiscard]]
    auto statsWriters() const -> const std::vector<OutputStats>& {
      return m_stat_writers;
    }
  };

} // namespace stats

#endif // OUTPUT_STATS_H

#include "output/stats.h"

#include "enums.h"
#include "global.h"

#include "arch/mpi_aliases.h"
#include "utils/error.h"
#include "utils/formatting.h"

#include "output/utils/interpret_prompt.h"

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace ntt;
using namespace out;

namespace stats {

  OutputStats::OutputStats(const std::string& name) : m_name { name } {
    // determine the stats ID
    const auto pos = name.find("_");
    auto name_raw  = (pos == std::string::npos) ? name : name.substr(0, pos);
    if ((name_raw[0] != 'E') and (name_raw[0] != 'B') and (name_raw[0] != 'J')) {
      name_raw = name_raw.substr(0, name_raw.find_first_of("0123ijxyzt"));
    }
    if (StatsID::contains(fmt::toLower(name_raw).c_str())) {
      m_id = StatsID::pick(fmt::toLower(name_raw).c_str());
    } else {
      raise::Error("Unrecognized stats ID " + fmt::toLower(name_raw), HERE);
    }
    // determine the species and components to output
    if (is_moment()) {
      species = InterpretSpecies(name);
    } else {
      species = {};
    }
    if (is_vector()) {
      // always write all the ExB and V components
      comp = { { 1 }, { 2 }, { 3 } };
    } else if (id() == StatsID::T) {
      // energy-momentum tensor
      comp = InterpretComponents({ name.substr(1, 1), name.substr(2, 1) });
    } else {
      // scalar (e.g., Rho, E^2, etc.)
      comp = {};
    }
  }

  void Writer::init(timestep_t interval, simtime_t interval_time) {
    m_tracker = tools::Tracker("stats", interval, interval_time);
  }

  auto Writer::shouldWrite(timestep_t step, simtime_t time) -> bool {
    return m_tracker.shouldWrite(step, time);
  }

  void Writer::defineStatsFilename(const std::string& filename) {
    m_fname = filename;
  }

  void Writer::defineStatsOutputs(const std::vector<std::string>& stats_to_write) {
    for (const auto& stat : stats_to_write) {
      m_stat_writers.emplace_back(stat);
    }
  }

  void Writer::writeHeader() {
    CallOnce(
      [](auto& fname, auto& stat_writers) {
        std::fstream StatsOut(fname, std::fstream::out | std::fstream::app);
        StatsOut << "step,time,";
        for (const auto& stat : stat_writers) {
          if (stat.is_vector()) {
            for (auto i { 0u }; i < stat.comp.size(); ++i) {
              StatsOut << stat.name(i) << ",";
            }
          } else {
            StatsOut << stat.name() << ",";
          }
        }
        StatsOut << std::endl;
        StatsOut.close();
      },
      m_fname,
      m_stat_writers);
  }

  template <typename T>
  void Writer::write<T>(const T& value) {
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

  void Writer::endWriting() {
    CallOnce(
      [](auto& fname) {
        std::fstream StatsOut(fname, std::fstream::out | std::fstream::app);
        StatsOut << std::endl;
        StatsOut.close();
      },
      m_fname);
  }

  template void Writer::write<char>(const char&);
  template void Writer::write<std::string>(const std::string&);
  template void Writer::write<float>(const float&);
  template void Writer::write<double>(const double&);
  template void Writer::write<int>(const int&);
  template void Writer::write<unsigned int>(const unsigned int&);
  template void Writer::write<short>(const short&);
  template void Writer::write<unsigned short>(const unsigned short&);
  template void Writer::write<long int>(const long int&);
  template void Writer::write<unsigned long int>(const unsigned long int&);
  template void Writer::write<long long int>(const long long int&);
  template void Writer::write<unsigned long long int>(
    const unsigned long long int&);

} // namespace stats

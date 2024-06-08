/**
 * @file output/writer.h
 * @brief Writer class which takes care of data output
 */

#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/param_container.h"

#include "output/fields.h"
#include "output/particles.h"
#include "output/spectra.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <string>
#include <vector>

namespace out {

  class Tracker {
    const std::string m_type;
    const std::size_t m_interval;
    const long double m_interval_time;
    const bool        m_use_time;

    long double m_last_output_time { -1.0 };

  public:
    Tracker(const std::string& type, std::size_t interval, long double interval_time)
      : m_type { type }
      , m_interval { interval }
      , m_interval_time { interval_time }
      , m_use_time { interval_time > 0.0 } {}

    ~Tracker() = default;

    auto shouldWrite(std::size_t step, long double time) -> bool {
      if (m_use_time) {
        if (time - m_last_output_time >= m_interval_time) {
          m_last_output_time = time;
          return true;
        } else {
          return false;
        }
      } else {
        return step % m_interval == 0;
      }
    }
  };

  class Writer {
#if !defined(MPI_ENABLED)
    adios2::ADIOS m_adios;
#else // MPI_ENABLED
    adios2::ADIOS m_adios { MPI_COMM_WORLD };
#endif
    adios2::IO     m_io;
    adios2::Engine m_writer;
    adios2::Mode   m_mode { adios2::Mode::Write };

    // global shape of the fields array to output
    adios2::Dims      m_flds_g_shape;
    // local corner of the fields array to output
    adios2::Dims      m_flds_l_corner;
    // local shape of the fields array to output
    adios2::Dims      m_flds_l_shape;
    bool              m_flds_ghosts;
    const std::string m_engine;

    std::map<std::string, Tracker> m_trackers;

    std::vector<OutputField>   m_flds_writers;
    std::vector<OutputSpecies> m_prtl_writers;
    std::vector<OutputSpectra> m_spectra_writers;

  public:
    Writer() : m_engine { "disabled" } {}

    Writer(const std::string& engine);
    ~Writer() = default;

    Writer(Writer&&) = default;

    void addTracker(const std::string&, std::size_t, long double);
    auto shouldWrite(const std::string&, std::size_t, long double) -> bool;

    void writeAttrs(const prm::Parameters& params);

    void defineMeshLayout(const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          bool incl_ghosts,
                          Coord);

    void defineFieldOutputs(const SimEngine&, const std::vector<std::string>&);
    void defineParticleOutputs(Dimension, const std::vector<unsigned short>&);
    void defineSpectraOutputs(const std::vector<unsigned short>&);

    void writeMesh(unsigned short, const array_t<real_t*>&, const array_t<real_t*>&);

    template <Dimension D, int N>
    void writeField(const std::vector<std::string>&,
                    const ndfield_t<D, N>&,
                    const std::vector<std::size_t>&);

    void writeParticleQuantity(const array_t<real_t*>&, const std::string&);
    void writeSpectrum(const array_t<real_t*>&, const std::string&);
    void writeSpectrumBins(const array_t<real_t*>&, const std::string&);

    void beginWriting(const std::string&, std::size_t, long double);
    void endWriting();

    /* getters -------------------------------------------------------------- */
    auto fieldWriters() const -> const std::vector<OutputField>& {
      return m_flds_writers;
    }

    auto speciesWriters() const -> const std::vector<OutputSpecies>& {
      return m_prtl_writers;
    }

    auto spectraWriters() const -> const std::vector<OutputSpectra>& {
      return m_spectra_writers;
    }
  };

} // namespace out

#endif // OUTPUT_WRITER_H

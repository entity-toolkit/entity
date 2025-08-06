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
#include "utils/tools.h"

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

  class Writer {
    adios2::ADIOS* p_adios { nullptr };

    adios2::IO     m_io;
    adios2::Engine m_writer;
    adios2::Mode   m_mode { adios2::Mode::Write };

    bool m_separate_files;

    // global shape of the fields array to output
    std::vector<ncells_t> m_flds_g_shape;
    // local corner of the fields array to output
    std::vector<ncells_t> m_flds_l_corner;
    // local shape of the fields array to output
    std::vector<ncells_t> m_flds_l_shape;

    // downsampling factors for each dimension
    std::vector<unsigned int> m_dwn;
    // starting cell in each dimension (not including ghosts)
    std::vector<ncells_t>     m_flds_l_first;

    // same but downsampled
    adios2::Dims m_flds_g_shape_dwn;
    adios2::Dims m_flds_l_corner_dwn;
    adios2::Dims m_flds_l_shape_dwn;

    bool        m_flds_ghosts;
    std::string m_engine;
    path_t      m_root;

    std::map<std::string, tools::Tracker> m_trackers;

    std::vector<OutputField>   m_flds_writers;
    std::vector<OutputSpecies> m_prtl_writers;
    std::vector<OutputSpectra> m_spectra_writers;

    WriteModeTags m_active_mode { WriteMode::None };

  public:
    Writer() {}

    ~Writer() = default;

    Writer(Writer&&) = default;

    void init(adios2::ADIOS*, const std::string&, const std::string&, bool);

    void setMode(adios2::Mode);

    void addTracker(const std::string&, timestep_t, simtime_t);
    auto shouldWrite(const std::string&, timestep_t, simtime_t) -> bool;

    void writeAttrs(const prm::Parameters&);

    void defineMeshLayout(const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          const std::vector<std::size_t>&,
                          const std::pair<unsigned int, unsigned int>&,
                          const std::vector<unsigned int>&,
                          bool,
                          Coord);

    void defineFieldOutputs(const SimEngine&, const std::vector<std::string>&);
    void defineParticleOutputs(Dimension, const std::vector<spidx_t>&);
    void defineSpectraOutputs(const std::vector<spidx_t>&);

    void writeMesh(unsigned short,
                   const array_t<real_t*>&,
                   const array_t<real_t*>&,
                   const std::vector<std::size_t>&);

    template <Dimension D, int N>
    void writeField(const std::vector<std::string>&,
                    const ndfield_t<D, N>&,
                    const std::vector<std::size_t>&);

    void writeParticleQuantity(const array_t<real_t*>&,
                               npart_t,
                               npart_t,
                               const std::string&);
    void writeSpectrum(const array_t<real_t*>&, const std::string&);
    void writeSpectrumBins(const array_t<real_t*>&, const std::string&);

    void beginWriting(WriteModeTags, timestep_t, simtime_t);
    void endWriting(WriteModeTags);

    /* getters -------------------------------------------------------------- */
    auto root() const -> const path_t& {
      return m_root;
    }

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

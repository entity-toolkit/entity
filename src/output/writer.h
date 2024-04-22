/**
 * @file output/writer.h
 * @brief Writer class which takes care of data output
 * @depends:
 *   - enums.h
 *   - global.h
 *   - arch/kokkos_aliases.h
 *   - utils/param_container.h
 *   - output/fields.h
 *   - output/particles.h
 */

#ifndef OUTPUT_WRITER_H
#define OUTPUT_WRITER_H

#include "enums.h"
#include "global.h"

#include "arch/kokkos_aliases.h"
#include "utils/param_container.h"

#include "output/fields.h"
#include "output/particles.h"

#include <adios2.h>
#include <adios2/cxx11/KokkosView.h>

#if defined(MPI_ENABLED)
  #include <mpi.h>
#endif

#include <string>
#include <vector>

namespace out {

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

    std::vector<OutputField> m_flds_writers;
    // std::vector<OutputParticles> m_prtl_writers;

  public:
    Writer() : m_engine { "disabled" } {}

    Writer(const std::string& engine);
    ~Writer() = default;

    Writer(Writer&&) = default;

    void writeAttrs(const prm::Parameters& params);

    void defineFieldLayout(const std::vector<std::size_t>&,
                           const std::vector<std::size_t>&,
                           const std::vector<std::size_t>&,
                           bool incl_ghosts);

    void defineFieldOutputs(const SimEngine& S, const std::vector<std::string>&);

    template <Dimension D, int N>
    void writeField(const std::vector<std::string>&,
                    const ndfield_t<D, N>&,
                    const std::vector<std::size_t>&);

    void beginWriting(const std::string&, std::size_t, real_t);
    void endWriting();

    /* getters -------------------------------------------------------------- */
    auto fieldWriters() const -> const std::vector<OutputField>& {
      return m_flds_writers;
    }
  };

} // namespace out

#endif // OUTPUT_WRITER_H
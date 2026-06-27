/**
 * @file output/ascent_writer.h
 * @brief Wrapper around the Ascent in situ visualization library
 * @implements
 *   - out::AscentWriter
 * @cpp:
 *   - ascent_writer.cpp
 * @namespaces:
 *   - out::
 * @macros:
 *   - MPI_ENABLED
 *   - OUTPUT_ENABLED
 *   - ASCENT_ENABLED
 * @note
 * AscentWriter mirrors the layout of out::Writer but instead of dumping data
 * to ADIOS2 it publishes a Conduit-Blueprint mesh to Ascent and triggers a
 * user-supplied actions file to render images on the fly.
 */

#ifndef OUTPUT_ASCENT_WRITER_H
#define OUTPUT_ASCENT_WRITER_H

#if defined(ASCENT_ENABLED)

  #include "enums.h"
  #include "global.h"

  #include "arch/kokkos_aliases.h"
  #include "utils/tools.h"

  #include <ascent.hpp>
  #include <conduit.hpp>

  #if defined(MPI_ENABLED)
    #include <mpi.h>
  #endif

  #include <string>
  #include <vector>

namespace out {

  class AscentWriter {
    ascent::Ascent m_ascent;
    conduit::Node  m_mesh;
    conduit::Node  m_options;
    // Pristine pre-loaded actions tree (int64 leaves already demoted to
    // int32 once at init time). Used as the source for `m_actions_work`
    // when a non-zero `v_drift` requires a per-render rewrite.
    conduit::Node  m_actions;
    // Working copy with `v_drift * t` applied to camera positions /
    // look-at points. Allocated lazily; only populated when v_drift != 0.
    conduit::Node  m_actions_work;
    bool           m_initialized { false };
    bool           m_mesh_defined { false };
    bool           m_pending_render { false };
    bool           m_have_actions { false };
    bool           m_vector_aliases { true };
    // Drift velocity along the x-axis (code units). At render time the
    // first component of each `camera/position` and `camera/look_at`
    // entry in the actions tree is shifted by `m_v_drift * time`. Zero
    // means the actions file is executed unmodified.
    real_t         m_v_drift { 0.0 };
    // Rotational velocity (radians per code time unit) around each
    // camera's `look_at` point, using `up` as the rotation axis. The
    // rotation `m_v_rot * time` is applied to the camera position
    // before the v_drift translation, so the camera orbits the drifted
    // focus point. Zero leaves the camera unrotated.
    real_t         m_v_rot { 0.0 };
    // Blueprint mesh structure does not change between renders (only state
    // values + field values do). Verify once, skip on subsequent renders.
    bool           m_verified { false };

    Dimension                m_dim { Dim::_3D };
    // m_l_shape stores the *downsampled* local cell shape (what gets
    // published). m_l_first_cell + m_downsample define how to map each
    // downsampled cell back to an original (full-resolution) cell index.
    std::vector<std::size_t> m_l_shape;
    std::vector<std::size_t> m_l_corner;
    std::vector<std::size_t> m_l_first_cell;
    std::vector<std::size_t> m_downsample;
    std::string              m_root;
    std::string              m_actions_file;
    std::vector<std::string> m_fields;

    // Persistent staging buffers reused across all publishField() calls.
    // Sized once in defineMesh(); avoids per-render device + host allocations
    // and a redundant LayoutLeft → C-order host reorder.
    array_t<double*>        m_field_buf_d;
    array_mirror_t<double*> m_field_buf_h;
    std::size_t             m_buf_nelem { 0 };

    tools::Tracker m_tracker;

  public:
    AscentWriter() = default;
    ~AscentWriter();

    AscentWriter(const AscentWriter&)            = delete;
    AscentWriter& operator=(const AscentWriter&) = delete;

    /**
     * @brief Initialize the underlying ascent::Ascent instance.
     * @param title Simulation name (used for the output directory).
     * @param actions_file Path to a yaml/json file with Ascent actions.
     * @param fields List of field names (e.g. "B3") that will be published.
     * @param interval Step interval between renders (used when interval_time<=0).
     * @param interval_time Sim-time interval between renders.
     * @param vector_aliases If true, scalar components named B1/B2/B3 (etc.)
     *        are *also* published as the matching component of an MCArray
     *        vector field "B". Set to false to halve the data volume when
     *        the actions file only references the scalar form.
     * @param v_drift Drift velocity along the x-axis (code units). At
     *        render time `v_drift * time` is added to the x-component of
     *        every `camera/position` and `camera/look_at` in the actions
     *        tree, letting the camera follow a moving window. Zero
     *        disables the rewrite and keeps the actions file untouched.
     * @param v_rot Rotational velocity (radians per code time unit)
     *        applied to each camera position about its `look_at` point,
     *        using `up` as the rotation axis. Combined freely with
     *        `v_drift`. Zero leaves the camera position unrotated.
     */
    void init(const std::string&              title,
              const std::string&              actions_file,
              const std::vector<std::string>& fields,
              timestep_t                      interval,
              simtime_t                       interval_time,
              bool                            vector_aliases,
              real_t                          v_drift,
              real_t                          v_rot);

    /**
     * @brief Whether the writer should fire on the current cycle.
     */
    auto shouldRender(timestep_t step, simtime_t time) -> bool;

    /**
     * @brief Define the local rectilinear-mesh layout.
     * @param dim Dimensionality of the mesh (1/2/3).
     * @param l_corner Local lower-left corner in global cell index space.
     * @param l_shape Local number of *downsampled* active cells in each direction.
     * @param l_first_cell Per-axis offset (in original-grid cells) of the first
     *        published cell within the local domain. Empty defaults to zeros.
     * @param downsample Per-axis stride applied when sampling the original
     *        full-resolution field. Empty defaults to ones (no downsampling).
     */
    void defineMesh(Dimension                       dim,
                    const std::vector<std::size_t>& l_corner,
                    const std::vector<std::size_t>& l_shape,
                    const std::vector<std::size_t>& l_first_cell = {},
                    const std::vector<std::size_t>& downsample   = {});

    /**
     * @brief Set the cell-edge coordinate arrays for one dimension.
     */
    void setMeshCoords(unsigned short dim, const array_t<real_t*>& xe);

    /**
     * @brief Push a single field component into the mesh blueprint.
     * @param name Field name (used by the actions file).
     * @param fld Backing storage for the simulation fields.
     * @param comp Component index to extract.
     */
    template <Dimension D, int N>
    void publishField(const std::string&     name,
                      const ndfield_t<D, N>& fld,
                      std::size_t            comp);

    /**
     * @brief Trigger the Ascent pipeline for the current step.
     * @return true if the pipeline actually executed (data was pending),
     *         false if there was nothing to render.
     */
    auto render(timestep_t step, simtime_t time) -> bool;

    /**
     * @brief Whether there is data published since the last render.
     */
    [[nodiscard]]
    auto hasPending() const -> bool {
      return m_pending_render;
    }

    /**
     * @brief Whether the writer was successfully initialized.
     */
    [[nodiscard]]
    auto initialized() const -> bool {
      return m_initialized;
    }

    /**
     * @brief Names of fields registered for in situ rendering.
     */
    [[nodiscard]]
    auto fields() const -> const std::vector<std::string>& {
      return m_fields;
    }
  };

} // namespace out

#endif // ASCENT_ENABLED

#endif // OUTPUT_ASCENT_WRITER_H

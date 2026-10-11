/**
 * @file framework/domain/comm/halo_exchange.hpp
 * @brief Batched halo exchange: every field and every neighbor direction of
 *        one CommunicateFields / SynchronizeFields call is packed by one
 *        kernel, sent as one message per direction with all messages in
 *        flight at once, and unpacked by one kernel (fill) or by one kernel per
 *        direction in direction order (additive).
 * @implements
 *   - comm::HaloSegment
 *   - comm::HaloField
 *   - comm::HaloDirection
 *   - comm::MakeHaloField<> -> comm::HaloField
 *   - comm::HaloExchanger<>
 * @namespaces:
 *   - comm::
 * @macros:
 *   - DEVICE_ENABLED
 *   - GPU_AWARE_MPI
 * @note Used only when MPI_ENABLED is set.
 * @note Message and buffer layout: the message of one direction is the
 *       concatenation of the blocks of all fields of the call, in the field
 *       order given by the caller; a block lists its (i1, i2, i3, comp)
 *       elements with i1 fastest. Sender and receiver build the same layout
 *       from the same slice extents, so no metadata travels with the data.
 * @note Pack buffers, receive buffers, segment tables and the SynchronizeFields
 *       accumulators persist across calls (grow-only); a cached segment table
 *       is re-uploaded only when the slices or the field allocations change.
 */

#ifndef FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP
#define FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP

#include "global.h"

#include "arch/kokkos_aliases.h"
#include "arch/mpi_aliases.h"
#include "utils/error.h"

#include <Kokkos_Core.hpp>
#include <mpi.h>

#include <climits>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace comm {
  using namespace ntt;

  /**
   * One rectangular block [lo, lo + n) x [c0, c0 + nc) of a field, moved
   * between a field and a flat message buffer, or between two blocks of
   * fields with identical strides. All members are 64-bit so the struct has
   * no padding and two tables compare with memcmp.
   */
  struct HaloSegment {
    enum Kind : int64_t {
      Pack,      // buffer   <- src block
      Unpack,    // dst block  = buffer
      UnpackAdd, // dst block += buffer
      Copy,      // dst block  = src block
      CopyAdd    // dst block += src block
    };

    real_t* src { nullptr };
    real_t* dst { nullptr };
    int64_t st[4] { 0, 0, 0, 0 }; // element strides of (i1, i2, i3, comp)
    int64_t src_lo[3] { 0, 0, 0 };
    int64_t dst_lo[3] { 0, 0, 0 };
    int64_t n[3] { 1, 1, 1 }; // block extent per axis (1 for absent axes)
    int64_t c0 { 0 };
    int64_t nc { 0 };
    int64_t off { 0 }; // element offset of the block in its flat buffer
    int64_t kind { Pack };

    [[nodiscard]]
    auto size() const -> int64_t {
      return n[0] * n[1] * n[2] * nc;
    }
  };

  /**
   * A field taking part in an exchange: values are read from `src` and
   * written (fill) or accumulated (additive) into `dst`; src == dst for a
   * plain ghost fill. Both must share the strides `st`.
   */
  struct HaloField {
    real_t*      src { nullptr };
    real_t*      dst { nullptr };
    int64_t      st[4] { 0, 0, 0, 0 };
    cell_range_t comps { 0, 0 };
  };

  template <Dimension D, unsigned short N>
  auto MakeHaloField(const ndfield_t<D, N>& src,
                     const ndfield_t<D, N>& dst,
                     const cell_range_t&    comps) -> HaloField {
    HaloField f;
    f.src = src.data();
    f.dst = dst.data();
    for (auto r { 0u }; r <= static_cast<unsigned>(D); ++r) {
      raise::ErrorIf(src.stride(r) != dst.stride(r),
                     "HaloExchange: source and destination layouts differ",
                     HERE);
      raise::ErrorIf(src.extent(r) != dst.extent(r),
                     "HaloExchange: source and destination extents differ",
                     HERE);
    }
    for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
      f.st[d] = static_cast<int64_t>(src.stride(d));
    }
    f.st[3] = static_cast<int64_t>(src.stride(static_cast<unsigned>(D)));
    f.comps = comps;
    return f;
  }

  /**
   * One neighbor direction of a call. `local` marks the periodic self-wrap
   * (the domain is its own neighbor): the block is copied within the domain
   * and no message is sent. `tag` is the index of the direction in
   * dir::Directions<D>::all, which the sender and the receiver of a message
   * share.
   */
  struct HaloDirection {
    int                       tag { 0 };
    bool                      local { false };
    int                       send_rank { -1 };
    int                       recv_rank { -1 };
    std::vector<cell_range_t> send_slice;
    std::vector<cell_range_t> recv_slice;
  };

  /**
   * Device functor applying the segments [sb, se) of a table to the flat
   * element range [start(sb), start(se)) (thread l handles element
   * start(sb) + l).
   */
  struct HaloSegments_kernel {
    Kokkos::View<const HaloSegment*> seg;
    Kokkos::View<const int64_t*>     start;
    Kokkos::View<real_t*>            sbuf;
    Kokkos::View<const real_t*>      rbuf;
    std::size_t                      sb, se;
    int64_t                          e0;

    Inline void operator()(const int64_t l) const {
      const int64_t e  = e0 + l;
      // largest s in [sb, se) with start(s) <= e
      std::size_t   lo = sb;
      std::size_t   hi = se;
      while (hi - lo > 1u) {
        const std::size_t mid = (lo + hi) / 2u;
        if (start(mid) <= e) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      const HaloSegment& g   = seg(lo);
      const int64_t      loc = e - start(lo);
      int64_t            r   = loc;
      const int64_t      i1  = r % g.n[0];
      r                     /= g.n[0];
      const int64_t i2       = r % g.n[1];
      r                     /= g.n[1];
      const int64_t i3       = r % g.n[2];
      r                     /= g.n[2];
      const int64_t c        = g.c0 + r;
      const int64_t a_src    = (g.src_lo[0] + i1) * g.st[0] +
                            (g.src_lo[1] + i2) * g.st[1] +
                            (g.src_lo[2] + i3) * g.st[2] + c * g.st[3];
      const int64_t a_dst = (g.dst_lo[0] + i1) * g.st[0] +
                            (g.dst_lo[1] + i2) * g.st[1] +
                            (g.dst_lo[2] + i3) * g.st[2] + c * g.st[3];
      if (g.kind == HaloSegment::Pack) {
        sbuf(g.off + loc) = g.src[a_src];
      } else if (g.kind == HaloSegment::Unpack) {
        g.dst[a_dst] = rbuf(g.off + loc);
      } else if (g.kind == HaloSegment::UnpackAdd) {
        g.dst[a_dst] += rbuf(g.off + loc);
      } else if (g.kind == HaloSegment::Copy) {
        g.dst[a_dst] = g.src[a_src];
      } else {
        g.dst[a_dst] += g.src[a_src];
      }
    }
  };

  template <Dimension D>
  class HaloExchanger {
    using seg_range_t = std::pair<std::size_t, std::size_t>;

    struct Message {
      int     rank;
      int     tag;
      int64_t off;
      int64_t count;

      auto operator==(const Message& o) const -> bool {
        return (rank == o.rank) and (tag == o.tag) and (off == o.off) and
               (count == o.count);
      }
    };

    struct Plan {
      std::vector<HaloSegment> h_seg;
      std::vector<int64_t>     h_start; // prefix element counts, size nseg + 1
      seg_range_t              out { 0u, 0u };   // pack (+ fill-mode local copies)
      seg_range_t              in { 0u, 0u };    // fill-mode unpack
      std::vector<seg_range_t> per_dir;          // additive unpack, direction order
      std::vector<Message>     sends, recvs;
      int64_t                  send_total { 0 }, recv_total { 0 };

      Kokkos::View<HaloSegment*> d_seg;
      Kokkos::View<int64_t*>     d_start;

      auto same_as(const Plan& o) const -> bool {
        return (h_seg.size() == o.h_seg.size()) and
               (std::memcmp(h_seg.data(),
                            o.h_seg.data(),
                            h_seg.size() * sizeof(HaloSegment)) == 0) and
               (h_start == o.h_start) and (out == o.out) and (in == o.in) and
               (per_dir == o.per_dir) and (sends == o.sends) and
               (recvs == o.recvs);
      }
    };

    // message tags: base + direction index (unique per direction)
    static constexpr int TAG_BASE = 1024;
    // cached tables per call key; a key keeps up to MAX_VARIANTS tables (field
    // allocations swapped between calls, e.g. a ping-pong filter)
    static constexpr std::size_t MAX_VARIANTS = 4u;

    std::map<int, std::vector<Plan>> m_plans;
    std::map<int, std::size_t>       m_next_slot;

    Kokkos::View<real_t*> m_send, m_recv;
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
    Kokkos::View<real_t*, Kokkos::HostSpace> m_send_h, m_recv_h;
#endif
    std::vector<MPI_Request> m_reqs;

    std::map<int, ndfield_t<D, 6>> m_acc6;
    std::map<int, ndfield_t<D, 3>> m_acc3;

    static void set_block(int64_t* lo, int64_t* n, const std::vector<cell_range_t>& slice) {
      for (auto d { 0u }; d < static_cast<unsigned>(D); ++d) {
        lo[d] = static_cast<int64_t>(slice[d].first);
        n[d]  = static_cast<int64_t>(slice[d].second - slice[d].first);
      }
    }

    static auto segment(const HaloField&                 f,
                        HaloSegment::Kind                kind,
                        const std::vector<cell_range_t>* src_slice,
                        const std::vector<cell_range_t>* dst_slice,
                        int64_t                          off) -> HaloSegment {
      HaloSegment g;
      std::memset(&g, 0, sizeof(HaloSegment));
      g.n[0] = g.n[1] = g.n[2] = 1;
      g.src                    = f.src;
      g.dst                    = f.dst;
      for (auto r { 0u }; r < 4u; ++r) {
        g.st[r] = f.st[r];
      }
      if (src_slice != nullptr) {
        set_block(g.src_lo, g.n, *src_slice);
      }
      if (dst_slice != nullptr) {
        int64_t n_dst[3] { 1, 1, 1 };
        set_block(g.dst_lo, n_dst, *dst_slice);
        if (src_slice != nullptr) {
          for (auto d { 0u }; d < 3u; ++d) {
            raise::ErrorIf(n_dst[d] != g.n[d],
                           "HaloExchange: send and receive blocks differ",
                           HERE);
          }
        } else {
          for (auto d { 0u }; d < 3u; ++d) {
            g.n[d] = n_dst[d];
          }
        }
      }
      g.c0   = static_cast<int64_t>(f.comps.first);
      g.nc   = static_cast<int64_t>(f.comps.second - f.comps.first);
      g.off  = off;
      g.kind = kind;
      return g;
    }

    static auto build(const std::vector<HaloDirection>& dirs,
                      const std::vector<HaloField>&     flds,
                      bool                              additive) -> Plan {
      Plan p;
      // pack segments, one message per sending direction
      for (const auto& dr : dirs) {
        if (dr.local or dr.send_rank < 0) {
          continue;
        }
        const int64_t msg_off = p.send_total;
        for (const auto& f : flds) {
          p.h_seg.push_back(
            segment(f, HaloSegment::Pack, &dr.send_slice, nullptr, p.send_total));
          p.send_total += p.h_seg.back().size();
        }
        p.sends.push_back({ dr.send_rank, dr.tag, msg_off, p.send_total - msg_off });
      }
      // fill mode: the local wraps read active cells and write ghosts, so they
      // share the pack kernel
      if (not additive) {
        for (const auto& dr : dirs) {
          if (not dr.local) {
            continue;
          }
          for (const auto& f : flds) {
            p.h_seg.push_back(
              segment(f, HaloSegment::Copy, &dr.send_slice, &dr.recv_slice, 0));
          }
        }
      }
      p.out = { 0u, p.h_seg.size() };
      if (not additive) {
        // fill mode: receive blocks of different directions are disjoint, so
        // one kernel unpacks all of them
        for (const auto& dr : dirs) {
          if (dr.local or dr.recv_rank < 0) {
            continue;
          }
          const int64_t msg_off = p.recv_total;
          for (const auto& f : flds) {
            p.h_seg.push_back(
              segment(f, HaloSegment::Unpack, nullptr, &dr.recv_slice, p.recv_total));
            p.recv_total += p.h_seg.back().size();
          }
          p.recvs.push_back({ dr.recv_rank, dr.tag, msg_off, p.recv_total - msg_off });
        }
        p.in = { p.out.second, p.h_seg.size() };
      } else {
        // additive mode: receive blocks overlap, so contributions are summed
        // one direction at a time, in direction order
        for (const auto& dr : dirs) {
          const auto first = p.h_seg.size();
          if (dr.local) {
            for (const auto& f : flds) {
              p.h_seg.push_back(
                segment(f, HaloSegment::CopyAdd, &dr.send_slice, &dr.recv_slice, 0));
            }
          } else if (dr.recv_rank >= 0) {
            const int64_t msg_off = p.recv_total;
            for (const auto& f : flds) {
              p.h_seg.push_back(segment(f,
                                        HaloSegment::UnpackAdd,
                                        nullptr,
                                        &dr.recv_slice,
                                        p.recv_total));
              p.recv_total += p.h_seg.back().size();
            }
            p.recvs.push_back(
              { dr.recv_rank, dr.tag, msg_off, p.recv_total - msg_off });
          }
          if (p.h_seg.size() > first) {
            p.per_dir.push_back({ first, p.h_seg.size() });
          }
        }
      }
      p.h_start.resize(p.h_seg.size() + 1u, 0);
      for (auto s { 0u }; s < p.h_seg.size(); ++s) {
        p.h_start[s + 1] = p.h_start[s] + p.h_seg[s].size();
      }
      return p;
    }

    void upload(Plan& p) {
      const auto nseg = p.h_seg.size();
      p.d_seg   = Kokkos::View<HaloSegment*> { "halo_segments", nseg };
      p.d_start = Kokkos::View<int64_t*> { "halo_segment_starts", nseg + 1u };
      auto h_seg   = Kokkos::create_mirror_view(p.d_seg);
      auto h_start = Kokkos::create_mirror_view(p.d_start);
      for (auto s { 0u }; s < nseg; ++s) {
        h_seg(s) = p.h_seg[s];
      }
      for (auto s { 0u }; s <= nseg; ++s) {
        h_start(s) = p.h_start[s];
      }
      Kokkos::deep_copy(p.d_seg, h_seg);
      Kokkos::deep_copy(p.d_start, h_start);
    }

    // cached plan equal to `p` for `key`, uploading `p` if there is none
    auto cached(int key, Plan&& p) -> Plan& {
      auto& variants = m_plans[key];
      for (auto& v : variants) {
        if (v.same_as(p)) {
          return v;
        }
      }
      // tables of a replaced plan may still be read by queued kernels
      Kokkos::fence("HaloExchange: segment table update");
      upload(p);
      if (variants.size() < MAX_VARIANTS) {
        variants.push_back(std::move(p));
        return variants.back();
      }
      auto& slot = m_next_slot[key];
      auto& v    = variants[slot];
      slot       = (slot + 1u) % MAX_VARIANTS;
      v          = std::move(p);
      return v;
    }

    void ensure_buffers(int64_t nsend, int64_t nrecv) {
      if ((static_cast<int64_t>(m_send.extent(0)) < nsend) or
          (static_cast<int64_t>(m_recv.extent(0)) < nrecv)) {
        // buffers being replaced may still be read by queued kernels
        Kokkos::fence("HaloExchange: buffer resize");
        if (static_cast<int64_t>(m_send.extent(0)) < nsend) {
          m_send = Kokkos::View<real_t*> {
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "halo_send"),
            static_cast<std::size_t>(nsend)
          };
        }
        if (static_cast<int64_t>(m_recv.extent(0)) < nrecv) {
          m_recv = Kokkos::View<real_t*> {
            Kokkos::view_alloc(Kokkos::WithoutInitializing, "halo_recv"),
            static_cast<std::size_t>(nrecv)
          };
        }
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        m_send_h = Kokkos::create_mirror_view(m_send);
        m_recv_h = Kokkos::create_mirror_view(m_recv);
#endif
      }
    }

    void launch(const Plan& p, const seg_range_t& grp) {
      if (grp.second <= grp.first) {
        return;
      }
      const int64_t e0    = p.h_start[grp.first];
      const int64_t count = p.h_start[grp.second] - e0;
      if (count <= 0) {
        return;
      }
      Kokkos::parallel_for(
        "HaloExchange",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, Kokkos::IndexType<int64_t>>(
          0,
          count),
        HaloSegments_kernel { p.d_seg, p.d_start, m_send, m_recv, grp.first, grp.second, e0 });
    }

  public:
    HaloExchanger() = default;

    /**
     * @brief Exchange the blocks of `flds` over all directions of `dirs`.
     * @param key identifies the call site (tags + mode); its segment table is
     *        cached across calls
     * @param additive false: receive blocks overwrite dst (ghost fill);
     *        true: received and wrapped blocks are summed into dst one
     *        direction at a time, in the order of `dirs`
     */
    void Exchange(int                               key,
                  const std::vector<HaloDirection>& dirs,
                  const std::vector<HaloField>&     flds,
                  bool                              additive) {
      if (dirs.empty() or flds.empty()) {
        return;
      }
      auto& p = cached(key, build(dirs, flds, additive));
      ensure_buffers(p.send_total, p.recv_total);

      launch(p, p.out);

      const bool remote = (not p.sends.empty()) or (not p.recvs.empty());
      if (remote) {
#if defined(DEVICE_ENABLED)
        // Intel GPUs: every kernel writing a send buffer must complete
        // before MPI reads it
        Kokkos::fence("HaloExchange: before MPI");
#endif
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        if (p.send_total > 0) {
          const auto rng = std::make_pair(static_cast<std::size_t>(0),
                                          static_cast<std::size_t>(p.send_total));
          Kokkos::deep_copy(Kokkos::subview(m_send_h, rng),
                            Kokkos::subview(m_send, rng));
        }
        real_t* const sptr = m_send_h.data();
        real_t* const rptr = m_recv_h.data();
#else
        real_t* const sptr = m_send.data();
        real_t* const rptr = m_recv.data();
#endif
        m_reqs.assign(p.sends.size() + p.recvs.size(), MPI_REQUEST_NULL);
        std::size_t q = 0u;
        for (const auto& m : p.recvs) {
          raise::ErrorIf(m.count > static_cast<int64_t>(INT_MAX),
                         "HaloExchange: message too large",
                         HERE);
          MPI_Irecv(rptr + m.off,
                    static_cast<int>(m.count),
                    mpi::get_type<real_t>(),
                    m.rank,
                    TAG_BASE + m.tag,
                    MPI_COMM_WORLD,
                    &m_reqs[q++]);
        }
        for (const auto& m : p.sends) {
          raise::ErrorIf(m.count > static_cast<int64_t>(INT_MAX),
                         "HaloExchange: message too large",
                         HERE);
          MPI_Isend(sptr + m.off,
                    static_cast<int>(m.count),
                    mpi::get_type<real_t>(),
                    m.rank,
                    TAG_BASE + m.tag,
                    MPI_COMM_WORLD,
                    &m_reqs[q++]);
        }
        MPI_Waitall(static_cast<int>(m_reqs.size()), m_reqs.data(), MPI_STATUSES_IGNORE);
#if defined(DEVICE_ENABLED) && !defined(GPU_AWARE_MPI)
        if (p.recv_total > 0) {
          const auto rng = std::make_pair(static_cast<std::size_t>(0),
                                          static_cast<std::size_t>(p.recv_total));
          Kokkos::deep_copy(Kokkos::subview(m_recv, rng),
                            Kokkos::subview(m_recv_h, rng));
        }
#endif
      }

      if (not additive) {
        launch(p, p.in);
      } else {
        for (const auto& grp : p.per_dir) {
          launch(p, grp);
        }
      }
    }

    /**
     * @brief Persistent zeroed accumulator shaped like `like` (zeroing is
     *        asynchronous; the buffer is reallocated when the extents change).
     */
    template <unsigned short N>
    auto Accumulator(int slot, const ndfield_t<D, N>& like) -> ndfield_t<D, N> {
      auto& store = [this]() -> std::map<int, ndfield_t<D, N>>& {
        if constexpr (N == 6) {
          return m_acc6;
        } else {
          static_assert(N == 3, "HaloExchange: accumulators are 3- or 6-component");
          return m_acc3;
        }
      }();
      auto& acc     = store[slot];
      bool  matches = (acc.data() != nullptr);
      for (auto r { 0u }; matches and r < static_cast<unsigned>(D); ++r) {
        matches = (acc.extent(r) == like.extent(r));
      }
      if (not matches) {
        Kokkos::fence("HaloExchange: accumulator resize");
        if constexpr (D == Dim::_1D) {
          acc = ndfield_t<D, N> { "halo_accumulator", like.extent(0) };
        } else if constexpr (D == Dim::_2D) {
          acc = ndfield_t<D, N> { "halo_accumulator", like.extent(0), like.extent(1) };
        } else {
          acc = ndfield_t<D, N> { "halo_accumulator",
                                  like.extent(0),
                                  like.extent(1),
                                  like.extent(2) };
        }
      } else {
        Kokkos::deep_copy(Kokkos::DefaultExecutionSpace {}, acc, ZERO);
      }
      return acc;
    }
  };

} // namespace comm

#endif // FRAMEWORK_DOMAIN_COMM_HALO_EXCHANGE_HPP

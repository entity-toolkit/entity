/**
 * @file output/render/png.h
 * @brief Self-contained, dependency-free PNG (8-bit RGBA) encoder
 * @implements
 *   - out::write_png
 * @namespaces:
 *   - out::
 * @note
 * Header-only. Emits a valid PNG using stored (uncompressed) DEFLATE blocks
 * wrapped in a zlib stream, with per-scanline filter type 0 (None). This keeps
 * the encoder tiny and provably correct at the cost of compression ratio; the
 * resulting files are still orders of magnitude smaller than the full-field
 * dumps the renderer is meant to replace. A drop-in stronger encoder (e.g. a
 * fixed-Huffman DEFLATE, or vendored stb_image_write) can replace the IDAT
 * producer without touching callers.
 */

#ifndef OUTPUT_RENDER_PNG_H
#define OUTPUT_RENDER_PNG_H

#include "global.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace out {

  namespace png_hidden {

    inline auto crc32(const uint8_t* data, std::size_t len) -> uint32_t {
      static uint32_t table[256];
      static bool     ready = false;
      if (not ready) {
        for (uint32_t n = 0; n < 256; ++n) {
          uint32_t c = n;
          for (int k = 0; k < 8; ++k) {
            c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
          }
          table[n] = c;
        }
        ready = true;
      }
      uint32_t c = 0xFFFFFFFFu;
      for (std::size_t i = 0; i < len; ++i) {
        c = table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
      }
      return c ^ 0xFFFFFFFFu;
    }

    inline auto adler32(const uint8_t* data, std::size_t len) -> uint32_t {
      constexpr uint32_t MOD = 65521u;
      uint32_t           a = 1u, b = 0u;
      for (std::size_t i = 0; i < len; ++i) {
        a = (a + data[i]) % MOD;
        b = (b + a) % MOD;
      }
      return (b << 16) | a;
    }

    inline void put_u32_be(std::vector<uint8_t>& v, uint32_t x) {
      v.push_back(static_cast<uint8_t>((x >> 24) & 0xFFu));
      v.push_back(static_cast<uint8_t>((x >> 16) & 0xFFu));
      v.push_back(static_cast<uint8_t>((x >> 8) & 0xFFu));
      v.push_back(static_cast<uint8_t>(x & 0xFFu));
    }

    inline void write_chunk(std::vector<uint8_t>&       out,
                            const char                  type[4],
                            const std::vector<uint8_t>& data) {
      put_u32_be(out, static_cast<uint32_t>(data.size()));
      std::vector<uint8_t> typed_data;
      typed_data.reserve(4 + data.size());
      for (int i = 0; i < 4; ++i) {
        typed_data.push_back(static_cast<uint8_t>(type[i]));
      }
      typed_data.insert(typed_data.end(), data.begin(), data.end());
      out.insert(out.end(), typed_data.begin(), typed_data.end());
      put_u32_be(out, crc32(typed_data.data(), typed_data.size()));
    }

    // zlib stream wrapping `raw` in stored (BTYPE=00) DEFLATE blocks
    inline auto zlib_store(const std::vector<uint8_t>& raw) -> std::vector<uint8_t> {
      std::vector<uint8_t> z;
      z.push_back(0x78); // CMF: CM=8, CINFO=7
      z.push_back(0x01); // FLG: makes (CMF<<8 | FLG) % 31 == 0, no dict, level 0
      std::size_t          off    = 0;
      const std::size_t    n      = raw.size();
      constexpr std::size_t BLOCK = 65535u;
      if (n == 0) {
        z.push_back(0x01); // final, stored
        z.push_back(0x00);
        z.push_back(0x00);
        z.push_back(0xFF);
        z.push_back(0xFF);
      }
      while (off < n) {
        const std::size_t len   = (n - off > BLOCK) ? BLOCK : (n - off);
        const bool        final = (off + len >= n);
        z.push_back(final ? 0x01 : 0x00);
        const uint16_t l  = static_cast<uint16_t>(len);
        const uint16_t nl = static_cast<uint16_t>(~l);
        z.push_back(static_cast<uint8_t>(l & 0xFFu));
        z.push_back(static_cast<uint8_t>((l >> 8) & 0xFFu));
        z.push_back(static_cast<uint8_t>(nl & 0xFFu));
        z.push_back(static_cast<uint8_t>((nl >> 8) & 0xFFu));
        z.insert(z.end(), raw.begin() + off, raw.begin() + off + len);
        off += len;
      }
      put_u32_be(z, adler32(raw.data(), raw.size()));
      return z;
    }

  } // namespace png_hidden

  /**
   * @brief Write an 8-bit RGBA buffer to a PNG file.
   * @param path output file path
   * @param width image width in pixels
   * @param height image height in pixels
   * @param rgba pointer to width*height*4 bytes, row-major, top-left origin
   * @return true on success
   */
  inline auto write_png(const path_t&  path,
                        int            width,
                        int            height,
                        const uint8_t* rgba) -> bool {
    using namespace png_hidden;
    const std::size_t    w = static_cast<std::size_t>(width);
    const std::size_t    h = static_cast<std::size_t>(height);
    // build filtered raw scanlines: each row prefixed with filter byte 0 (None)
    std::vector<uint8_t> raw;
    raw.reserve(h * (1 + w * 4));
    for (std::size_t y = 0; y < h; ++y) {
      raw.push_back(0x00);
      const uint8_t* row = rgba + y * w * 4;
      raw.insert(raw.end(), row, row + w * 4);
    }

    std::vector<uint8_t> file;
    // PNG signature
    const uint8_t sig[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
    file.insert(file.end(), sig, sig + 8);

    // IHDR
    std::vector<uint8_t> ihdr;
    put_u32_be(ihdr, static_cast<uint32_t>(width));
    put_u32_be(ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8); // bit depth
    ihdr.push_back(6); // color type: RGBA
    ihdr.push_back(0); // compression
    ihdr.push_back(0); // filter
    ihdr.push_back(0); // interlace
    write_chunk(file, "IHDR", ihdr);

    // IDAT
    write_chunk(file, "IDAT", zlib_store(raw));

    // IEND
    write_chunk(file, "IEND", {});

    std::ofstream f(path, std::ios::binary);
    if (not f.good()) {
      return false;
    }
    f.write(reinterpret_cast<const char*>(file.data()),
            static_cast<std::streamsize>(file.size()));
    return f.good();
  }

} // namespace out

#endif // OUTPUT_RENDER_PNG_H

/*
* Copyright (c) 2025 XLAB
* All rights reserved.
*
* This file is part of trueform (trueform.polydera.com)
*
* Licensed for noncommercial use under the PolyForm Noncommercial
* License 1.0.0.
* Commercial licensing available via info@polydera.com.
*
* Author: Žiga Sajovic
*/
#pragma once
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points_buffer.hpp"
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <fstream>

namespace tf::io {
class obj_reader {
public:
  // Read OBJ with dynamic Ngon (mixed polygon sizes)
  template <typename Index, typename RealT, std::size_t Dims>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &out_points,
            tf::offset_block_buffer<Index, Index> &out_faces) -> bool {
    // Load entire file into memory
    tf::buffer<char> file_data;
    if (!_load_file(path, file_data))
      return false;

    const char *data_ptr = file_data.begin();
    const char *data_end = file_data.end();

    std::uint64_t vertex_count{}, face_count{}, face_index_count{};
    _count_elements(data_ptr, data_end, vertex_count, face_count,
                    face_index_count);

    // Pre-allocate points
    const std::size_t points_base = out_points.size();
    out_points.allocate(points_base + static_cast<std::size_t>(vertex_count));

    // Pre-allocate faces (offset_block_buffer internals)
    auto &offsets = out_faces.offsets_buffer();
    auto &data = out_faces.data_buffer();
    const std::size_t offsets_base = offsets.size();
    const std::size_t data_base = data.size();

    // Initialize offsets array
    if (offsets_base == 0) {
      offsets.allocate(static_cast<std::size_t>(face_count) + 1);
      offsets[0] = 0;
    } else {
      offsets.reallocate(offsets_base + static_cast<std::size_t>(face_count));
    }
    data.allocate(data_base + static_cast<std::size_t>(face_index_count));

    // Read vertices and faces
    if (!_read_data_dynamic<Index, RealT, Dims>(
            data_ptr, data_end, out_points, points_base, offsets, data,
            offsets_base, data_base))
      return false;

    return true;
  }

  // Read OBJ with fixed Ngon (e.g., triangles only, quads only)
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &out_points,
            tf::blocked_buffer<Index, Ngon> &out_faces) -> bool {
    // Load entire file into memory
    tf::buffer<char> file_data;
    if (!_load_file(path, file_data))
      return false;

    const char *data_ptr = file_data.begin();
    const char *data_end = file_data.end();

    std::uint64_t vertex_count{}, face_count{}, face_index_count{};
    _count_elements(data_ptr, data_end, vertex_count, face_count,
                    face_index_count);

    // Validate all faces have correct Ngon
    if (face_index_count != face_count * Ngon)
      return false; // Mixed polygon sizes not allowed for fixed Ngon

    // Pre-allocate points
    const std::size_t points_base = out_points.size();
    out_points.allocate(points_base + static_cast<std::size_t>(vertex_count));

    // Pre-allocate faces
    const std::size_t faces_base = out_faces.size();
    out_faces.allocate(faces_base + static_cast<std::size_t>(face_count));

    // Read vertices and faces
    if (!_read_data_fixed<Index, RealT, Dims, Ngon>(
            data_ptr, data_end, out_points, points_base, out_faces, faces_base))
      return false;

    return true;
  }

private:
  // ---------- File helpers ----------
  static auto _load_file(std::string_view path, tf::buffer<char> &out) -> bool {
    std::ifstream f(std::string(path), std::ios::binary | std::ios::ate);
    if (!f)
      return false;

    const auto size = f.tellg();
    if (size <= 0)
      return false;

    out.allocate(static_cast<std::size_t>(size));
    f.seekg(0, std::ios::beg);
    if (!f.read(out.begin(), size))
      return false;

    return true;
  }

  // ---------- Counting (first pass) ----------
  static auto _count_elements(const char *p, const char *end,
                              std::uint64_t &vertex_count,
                              std::uint64_t &face_count,
                              std::uint64_t &face_index_count) -> void {
    vertex_count = 0;
    face_count = 0;
    face_index_count = 0;

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      if (p[0] == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Vertex line
        ++vertex_count;
        p = _skip_line(p, end);
      } else if (p[0] == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Face line - count indices
        p += 2;
        std::uint64_t indices_in_face = 0;
        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;

          // Skip to next whitespace
          while (p < end && *p != ' ' && *p != '\t' && *p != '\n' &&
                 *p != '\r' && *p != '#')
            ++p;

          ++indices_in_face;
        }
        if (indices_in_face >= 3) {
          ++face_count;
          face_index_count += indices_in_face;
        }
        p = _skip_line(p, end);
      } else {
        p = _skip_line(p, end);
      }
    }
  }

  // ---------- Reading data: dynamic Ngon ----------
  template <typename Index, typename RealT, std::size_t Dims>
  static auto _read_data_dynamic(const char *p, const char *end,
                                 tf::points_buffer<RealT, Dims> &points,
                                 std::size_t points_base,
                                 tf::buffer<Index> &offsets,
                                 tf::buffer<Index> &data,
                                 std::size_t offsets_base,
                                 std::size_t data_base) -> bool {
    auto points_iter = points.begin() + points_base;
    std::size_t current_offset_idx =
        offsets_base == 0 ? 1 : offsets_base; // Start after initial 0
    std::size_t current_data_idx = data_base;
    Index current_data_offset =
        offsets_base == 0 ? 0 : static_cast<Index>(data_base);

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      if (p[0] == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Parse vertex
        p += 2;
        float x{}, y{}, z{};
        if (!_parse_three_floats(p, end, x, y, z))
          return false;

        auto &&pt = *points_iter++;
        pt[0] = static_cast<RealT>(x);
        pt[1] = static_cast<RealT>(y);
        if constexpr (Dims > 2)
          pt[2] = static_cast<RealT>(z);

        p = _skip_line(p, end);

      } else if (p[0] == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Parse face
        p += 2;
        std::size_t indices_in_face = 0;

        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;

          int vertex_index{};
          if (!_parse_face_index(p, end, vertex_index))
            return false;

          // Convert 1-based to 0-based
          if (vertex_index <= 0)
            return false;
          data[current_data_idx++] = static_cast<Index>(vertex_index - 1);
          ++indices_in_face;
        }

        if (indices_in_face >= 3) {
          current_data_offset += static_cast<Index>(indices_in_face);
          offsets[current_offset_idx++] = current_data_offset;
        }
        p = _skip_line(p, end);
      } else {
        p = _skip_line(p, end);
      }
    }

    return true;
  }

  // ---------- Reading data: fixed Ngon ----------
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  static auto _read_data_fixed(const char *p, const char *end,
                               tf::points_buffer<RealT, Dims> &points,
                               std::size_t points_base,
                               tf::blocked_buffer<Index, Ngon> &faces,
                               std::size_t faces_base) -> bool {
    auto points_iter = points.begin() + points_base;
    auto faces_iter = faces.begin() + faces_base;

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      if (p[0] == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Parse vertex
        p += 2;
        float x{}, y{}, z{};
        if (!_parse_three_floats(p, end, x, y, z))
          return false;

        auto &&pt = *points_iter++;
        pt[0] = static_cast<RealT>(x);
        pt[1] = static_cast<RealT>(y);
        if constexpr (Dims > 2)
          pt[2] = static_cast<RealT>(z);

        p = _skip_line(p, end);

      } else if (p[0] == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        // Parse face - only read first Ngon indices, ignore extras
        p += 2;
        auto &&face = *faces_iter;
        std::size_t idx_count = 0;

        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;

          int vertex_index{};
          if (!_parse_face_index(p, end, vertex_index))
            return false;

          // Only write up to Ngon indices (skip extras safely)
          if (idx_count < Ngon) {
            // Convert 1-based to 0-based
            if (vertex_index <= 0)
              return false;
            face[idx_count] = static_cast<Index>(vertex_index - 1);
          }
          ++idx_count;
        }

        if (idx_count >= Ngon) {
          ++faces_iter;
        } else {
          return false; // Not enough indices for this Ngon
        }
        p = _skip_line(p, end);
      } else {
        p = _skip_line(p, end);
      }
    }

    return true;
  }

  // ---------- Parse helpers ----------
  static auto _parse_three_floats(const char *&p, const char *end, float &x,
                                  float &y, float &z) -> bool {
    (void)end; // strtof doesn't use end pointer
    char *endp;

    p = _skip_ws(p, end);
    x = std::strtof(p, &endp);
    if (endp == p)
      return false;
    p = endp;

    p = _skip_ws(p, end);
    y = std::strtof(p, &endp);
    if (endp == p)
      return false;
    p = endp;

    p = _skip_ws(p, end);
    z = std::strtof(p, &endp);
    if (endp == p)
      return false;
    p = endp;

    return true;
  }

  // Parse face index (handles "123", "123/45", "123//67", "123/45/67")
  // Advances pointer past the entire index specifier
  static auto _parse_face_index(const char *&p, const char *end,
                                int &vertex_index) -> bool {
    auto res = std::from_chars(p, end, vertex_index);
    if (res.ec != std::errc{})
      return false;
    p = res.ptr;

    // Skip texture and normal indices (after slashes)
    if (p < end && *p == '/') {
      ++p; // Skip first slash
      if (p < end && *p == '/') {
        ++p; // Skip second slash (format: v//vn)
      } else if (p < end && *p >= '0' && *p <= '9') {
        // Skip texture index
        int dummy{};
        res = std::from_chars(p, end, dummy);
        if (res.ec == std::errc{})
          p = res.ptr;
        if (p < end && *p == '/') {
          ++p; // Skip to normal index
        }
      }
      // Skip normal index if present
      if (p < end && *p >= '0' && *p <= '9') {
        int dummy{};
        res = std::from_chars(p, end, dummy);
        if (res.ec == std::errc{})
          p = res.ptr;
      }
    }

    return true;
  }

  static auto _skip_ws(const char *p, const char *end) -> const char * {
    while (p < end && (*p == ' ' || *p == '\t'))
      ++p;
    return p;
  }

  static auto _skip_line(const char *p, const char *end) -> const char * {
    while (p < end && *p != '\n' && *p != '\r')
      ++p;
    while (p < end && (*p == '\n' || *p == '\r'))
      ++p;
    return p;
  }
};
} // namespace tf::io

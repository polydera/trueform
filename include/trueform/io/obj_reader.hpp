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
#include "../core/range.hpp"
#include "./external/fast_float.hpp"
#include <charconv>
#include <cstdlib>
#include <fstream>

namespace tf::io {
class obj_reader {
public:
  // Read OBJ with dynamic Ngon (mixed polygon sizes)
  template <typename Index, typename RealT, std::size_t Dims>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &out_points,
            tf::offset_block_buffer<Index, Index> &out_faces) -> bool {
    tf::buffer<char> file_data;
    if (!_load_file(path, file_data))
      return false;
    return _read_single_pass_dynamic<Index, RealT, Dims>(
        file_data.begin(), file_data.end(), file_data.size(), out_points,
        out_faces);
  }

  // Read OBJ from memory buffer with dynamic Ngon
  template <typename Index, typename RealT, std::size_t Dims>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::points_buffer<RealT, Dims> &out_points,
            tf::offset_block_buffer<Index, Index> &out_faces) -> bool {
    return _read_single_pass_dynamic<Index, RealT, Dims>(
        data.begin(), data.end(), data.size(), out_points, out_faces);
  }

  // Read OBJ from memory buffer with fixed Ngon
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::points_buffer<RealT, Dims> &out_points,
            tf::blocked_buffer<Index, Ngon> &out_faces) -> bool {
    return _read_single_pass_fixed<Index, RealT, Dims, Ngon>(
        data.begin(), data.end(), data.size(), out_points, out_faces);
  }

  // Read OBJ with fixed Ngon (e.g., triangles only, quads only)
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  auto read(std::string_view path, tf::points_buffer<RealT, Dims> &out_points,
            tf::blocked_buffer<Index, Ngon> &out_faces) -> bool {
    tf::buffer<char> file_data;
    if (!_load_file(path, file_data))
      return false;
    return _read_single_pass_fixed<Index, RealT, Dims, Ngon>(
        file_data.begin(), file_data.end(), file_data.size(), out_points,
        out_faces);
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

  // ---------- Single-pass fixed Ngon ----------
  template <typename Index, typename RealT, std::size_t Dims, std::size_t Ngon>
  static auto
  _read_single_pass_fixed(const char *p, const char *end, std::size_t file_size,
                          tf::points_buffer<RealT, Dims> &out_points,
                          tf::blocked_buffer<Index, Ngon> &out_faces) -> bool {
    auto est = file_size / 28;
    auto points_base = out_points.size();
    auto faces_base = out_faces.size();
    out_points.reserve(points_base + est);
    out_faces.reserve(faces_base + est);

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      if (p[0] == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        float x{}, y{}, z{};
        if (!_parse_three_floats(p, end, x, y, z)) {
          out_points.clear();
          out_faces.clear();
          return false;
        }
        if constexpr (Dims == 3)
          out_points.emplace_back(static_cast<RealT>(x), static_cast<RealT>(y),
                                  static_cast<RealT>(z));
        else
          out_points.emplace_back(static_cast<RealT>(x), static_cast<RealT>(y));
        p = _skip_line(p, end);
      } else if (p[0] == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        std::array<Index, Ngon> face{};
        std::size_t idx_count = 0;
        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;
          int vertex_index{};
          if (!_parse_face_index(p, end, vertex_index)) {
            out_points.clear();
            out_faces.clear();
            return false;
          }
          if (vertex_index <= 0) {
            out_points.clear();
            out_faces.clear();
            return false;
          }
          if (idx_count < Ngon)
            face[idx_count] = static_cast<Index>(vertex_index - 1);
          ++idx_count;
        }
        if (idx_count != Ngon) {
          out_points.clear();
          out_faces.clear();
          return false;
        }
        out_faces.push_back(face);
        p = _skip_line(p, end);
      } else {
        p = _skip_line(p, end);
      }
    }
    return true;
  }

  // ---------- Single-pass dynamic Ngon ----------
  template <typename Index, typename RealT, std::size_t Dims>
  static auto
  _read_single_pass_dynamic(const char *p, const char *end,
                            std::size_t file_size,
                            tf::points_buffer<RealT, Dims> &out_points,
                            tf::offset_block_buffer<Index, Index> &out_faces)
      -> bool {
    // Reserve based on file size estimates:
    // vertex line ~30 bytes, face line ~20 bytes, ~60% vertices
    auto est = file_size / 28;
    out_points.reserve(est);
    auto &offsets = out_faces.offsets_buffer();
    auto &data = out_faces.data_buffer();
    if (offsets.size() == 0)
      offsets.push_back(0);
    offsets.reserve(est);
    data.reserve(est * 3);

    Index current_data_offset = offsets[offsets.size() - 1];

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      if (p[0] == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        float x{}, y{}, z{};
        if (!_parse_three_floats(p, end, x, y, z))
          return false;
        if constexpr (Dims == 3)
          out_points.emplace_back(static_cast<RealT>(x), static_cast<RealT>(y),
                                  static_cast<RealT>(z));
        else
          out_points.emplace_back(static_cast<RealT>(x), static_cast<RealT>(y));
        p = _skip_line(p, end);
      } else if (p[0] == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        std::size_t indices_in_face = 0;
        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;
          int vertex_index{};
          if (!_parse_face_index(p, end, vertex_index))
            return false;
          if (vertex_index <= 0)
            return false;
          data.push_back(static_cast<Index>(vertex_index - 1));
          ++indices_in_face;
        }
        if (indices_in_face >= 3) {
          current_data_offset += static_cast<Index>(indices_in_face);
          offsets.push_back(current_data_offset);
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
    p = _skip_ws(p, end);
    auto r = tf::external::fast_float::from_chars(p, end, x);
    if (r.ec != std::errc{})
      return false;
    p = r.ptr;

    p = _skip_ws(p, end);
    r = tf::external::fast_float::from_chars(p, end, y);
    if (r.ec != std::errc{})
      return false;
    p = r.ptr;

    p = _skip_ws(p, end);
    r = tf::external::fast_float::from_chars(p, end, z);
    if (r.ec != std::errc{})
      return false;
    p = r.ptr;

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

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
#include "../core/algorithm/make_unique_index_map.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/algorithm/reduce.hpp"
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/points_buffer.hpp"
#include "../core/range.hpp"
#include "../core/views/mapped_range.hpp"
#include "./external/fast_float.hpp"
#include "./obj_file.hpp"
#include <array>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

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

  // Complete-mode: positions + normals + textures + groups + objects.
  template <typename Index, typename RealT>
  auto read(std::string_view path, tf::obj_file<Index, RealT> &out) -> bool {
    tf::buffer<char> file_data;
    if (!_load_file(path, file_data))
      return false;
    return _read_complete<Index, RealT>(file_data.begin(), file_data.end(),
                                        out);
  }

  template <typename Index, typename RealT>
  auto read(tf::range<const char *, tf::dynamic_size> data,
            tf::obj_file<Index, RealT> &out) -> bool {
    return _read_complete<Index, RealT>(data.begin(), data.end(), out);
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
        RealT x{}, y{}, z{};
        if (!_parse_three_scalars(p, end, x, y, z)) {
          out_points.clear();
          out_faces.clear();
          return false;
        }
        if constexpr (Dims == 3)
          out_points.emplace_back(x, y, z);
        else
          out_points.emplace_back(x, y);
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
        RealT x{}, y{}, z{};
        if (!_parse_three_scalars(p, end, x, y, z))
          return false;
        if constexpr (Dims == 3)
          out_points.emplace_back(x, y, z);
        else
          out_points.emplace_back(x, y);
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

  // ---------- Complete-mode parser ----------
  enum class _face_mode { unknown, v, v_vt, v_vn, v_vt_vn };

  // Parse a single face vertex reference and emit (v, vt, vn) into `out`,
  // -1 in slots not present. Validates that the slot pattern matches `mode`
  // (or sets `mode` if currently `unknown`). Returns false on parse error
  // or pattern mismatch.
  static auto _parse_face_triplet(const char *&p, const char *end,
                                  _face_mode &mode,
                                  std::array<int, 3> &out) -> bool {
    int v = 0, vt = 0, vn = 0;
    bool has_vt = false, has_vn = false;

    auto res = std::from_chars(p, end, v);
    if (res.ec != std::errc{} || v == 0)
      return false;
    p = res.ptr;

    if (p < end && *p == '/') {
      ++p;
      if (p < end && *p == '/') {
        // v//vn
        ++p;
        res = std::from_chars(p, end, vn);
        if (res.ec != std::errc{} || vn == 0)
          return false;
        p = res.ptr;
        has_vn = true;
      } else {
        // v/vt[/vn]
        res = std::from_chars(p, end, vt);
        if (res.ec != std::errc{} || vt == 0)
          return false;
        p = res.ptr;
        has_vt = true;
        if (p < end && *p == '/') {
          ++p;
          res = std::from_chars(p, end, vn);
          if (res.ec != std::errc{} || vn == 0)
            return false;
          p = res.ptr;
          has_vn = true;
        }
      }
    }

    auto observed = !has_vt && !has_vn ? _face_mode::v
                  : has_vt && !has_vn  ? _face_mode::v_vt
                  : !has_vt && has_vn  ? _face_mode::v_vn
                                       : _face_mode::v_vt_vn;
    if (mode == _face_mode::unknown)
      mode = observed;
    else if (mode != observed)
      return false;

    out[0] = v - 1;
    out[1] = has_vt ? vt - 1 : -1;
    out[2] = has_vn ? vn - 1 : -1;
    return true;
  }

  // Read the next whitespace-delimited token after `p`, advancing past it.
  // Empty (line end / comment / EOF) → returns empty string_view, leaves `p`.
  static auto _parse_token(const char *&p, const char *end) -> std::string_view {
    p = _skip_ws(p, end);
    auto start = p;
    while (p < end && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r' &&
           *p != '#')
      ++p;
    return {start, static_cast<std::size_t>(p - start)};
  }

  template <typename Index, typename RealT>
  static auto _read_complete(const char *p, const char *end,
                             tf::obj_file<Index, RealT> &out) -> bool {
    static_assert(std::is_floating_point_v<RealT>,
                  "obj_reader requires a floating-point Real type");
    out = tf::obj_file<Index, RealT>{};

    tf::buffer<tf::point<RealT, 3>> lib_v;
    tf::buffer<tf::point<RealT, 2>> lib_vt;
    tf::buffer<tf::point<RealT, 3>> lib_vn;

    tf::buffer<std::array<int, 3>> flat_triplets;
    tf::buffer<Index> face_offsets; // size = n_faces + 1
    face_offsets.push_back(0);
    tf::buffer<Index> face_group_id;
    tf::buffer<Index> face_object_id;

    std::vector<std::string> group_names;
    std::vector<std::string> object_names;
    Index current_group = Index(-1);
    Index current_object = Index(-1);
    bool group_needs_default = false;
    bool object_needs_default = false;

    _face_mode mode = _face_mode::unknown;

    while (p < end) {
      p = _skip_ws(p, end);
      if (p >= end)
        break;

      char c0 = p[0];
      if (c0 == 'v' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        RealT x{}, y{}, z{};
        if (!_parse_three_scalars(p, end, x, y, z))
          return false;
        lib_v.emplace_back(x, y, z);
        p = _skip_line(p, end);
      } else if (c0 == 'v' && p + 2 < end && p[1] == 't' &&
                 (p[2] == ' ' || p[2] == '\t')) {
        p += 3;
        RealT u{}, v{};
        if (!_parse_two_scalars(p, end, u, v))
          return false;
        lib_vt.emplace_back(u, v);
        p = _skip_line(p, end);
      } else if (c0 == 'v' && p + 2 < end && p[1] == 'n' &&
                 (p[2] == ' ' || p[2] == '\t')) {
        p += 3;
        RealT x{}, y{}, z{};
        if (!_parse_three_scalars(p, end, x, y, z))
          return false;
        lib_vn.emplace_back(x, y, z);
        p = _skip_line(p, end);
      } else if (c0 == 'g' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        auto name = _parse_token(p, end);
        if (name.empty())
          name = std::string_view{"default"};
        group_names.emplace_back(name);
        current_group = static_cast<Index>(group_names.size() - 1);
        p = _skip_line(p, end);
      } else if (c0 == 'o' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        auto name = _parse_token(p, end);
        if (name.empty())
          name = std::string_view{"default"};
        object_names.emplace_back(name);
        current_object = static_cast<Index>(object_names.size() - 1);
        p = _skip_line(p, end);
      } else if (c0 == 'f' && p + 1 < end && (p[1] == ' ' || p[1] == '\t')) {
        p += 2;
        int corner_count = 0;
        while (p < end && *p != '\n' && *p != '\r') {
          p = _skip_ws(p, end);
          if (p >= end || *p == '\n' || *p == '\r' || *p == '#')
            break;
          std::array<int, 3> triplet{};
          if (!_parse_face_triplet(p, end, mode, triplet))
            return false;
          flat_triplets.push_back(triplet);
          ++corner_count;
        }
        if (corner_count < 3)
          return false;
        face_offsets.push_back(face_offsets.back() +
                               static_cast<Index>(corner_count));
        face_group_id.push_back(current_group);
        face_object_id.push_back(current_object);
        if (current_group == Index(-1))
          group_needs_default = true;
        if (current_object == Index(-1))
          object_needs_default = true;
        p = _skip_line(p, end);
      } else {
        // Comment, mtllib, usemtl, s, or anything else: skip.
        p = _skip_line(p, end);
      }
    }

    if (lib_v.size() == 0 || flat_triplets.size() == 0)
      return false;

    // Validate triplet indices in range.
    auto v_n = static_cast<int>(lib_v.size());
    auto vt_n = static_cast<int>(lib_vt.size());
    auto vn_n = static_cast<int>(lib_vn.size());
    bool has_vt = (mode == _face_mode::v_vt || mode == _face_mode::v_vt_vn);
    bool has_vn = (mode == _face_mode::v_vn || mode == _face_mode::v_vt_vn);
    auto in_range = tf::make_mapped_range(flat_triplets, [&](const auto &t) {
      if (t[0] < 0 || t[0] >= v_n)
        return false;
      if (has_vt && (t[1] < 0 || t[1] >= vt_n))
        return false;
      if (has_vn && (t[2] < 0 || t[2] >= vn_n))
        return false;
      return true;
    });
    if (!tf::reduce(
            in_range,
            [](bool a, bool b) { return a && b; }, true, tf::checked))
      return false;

    // Dedup unique (v, vt, vn) triplets.
    tf::index_map_buffer<Index> im;
    tf::make_unique_index_map(flat_triplets, im);
    auto n_unique = static_cast<Index>(im.kept_ids().size());

    // Build aligned points / normals / textures.
    auto &points = out.polygons.points_buffer();
    points.allocate(n_unique);
    if (has_vn)
      out.normals.allocate(n_unique);
    if (has_vt)
      out.textures.allocate(n_unique);
    tf::parallel_for_each(
        tf::enumerate(im.kept_ids()),
        [&](auto pair) {
          auto &&[i, kept] = pair;
          auto &t = flat_triplets[kept];
          points[i] = lib_v[t[0]];
          if (has_vn) {
            auto &n = lib_vn[t[2]];
            out.normals.data_buffer()[3 * i + 0] = n[0];
            out.normals.data_buffer()[3 * i + 1] = n[1];
            out.normals.data_buffer()[3 * i + 2] = n[2];
          }
          if (has_vt) {
            auto &uv = lib_vt[t[1]];
            out.textures.data_buffer()[2 * i + 0] = uv[0];
            out.textures.data_buffer()[2 * i + 1] = uv[1];
          }
        },
        tf::checked);

    // Build face data: offsets move directly; corner indices remap via im.f().
    auto &faces = out.polygons.faces_buffer();
    faces.offsets_buffer() = std::move(face_offsets);
    faces.data_buffer().allocate(flat_triplets.size());
    tf::parallel_for_each(
        tf::enumerate(im.f()),
        [&](auto pair) {
          auto &&[i, val] = pair;
          faces.data_buffer()[i] = static_cast<Index>(val);
        },
        tf::checked);

    // Resolve face_groups / face_objects, inserting "default" lazily if
    // any face was emitted before the first `g` / `o`.
    auto resolve_labels = [](tf::buffer<Index> &src,
                             std::vector<std::string> &names,
                             bool needs_default, tf::buffer<Index> &dst,
                             std::vector<std::string> &dst_names) {
      if (names.empty())
        return; // file had no directives, dst stays empty
      if (!needs_default) {
        dst = std::move(src);
        dst_names = std::move(names);
        return;
      }
      dst_names.reserve(names.size() + 1);
      dst_names.emplace_back("default");
      for (auto &n : names)
        dst_names.push_back(std::move(n));
      dst.allocate(src.size());
      tf::parallel_for_each(
          tf::enumerate(src),
          [&](auto pair) {
            auto &&[i, id] = pair;
            dst[i] = (id == Index(-1)) ? Index(0) : Index(id + 1);
          },
          tf::checked);
    };
    resolve_labels(face_group_id, group_names, group_needs_default,
                   out.face_groups, out.group_names);
    resolve_labels(face_object_id, object_names, object_needs_default,
                   out.face_objects, out.object_names);

    return true;
  }

  template <typename Scalar>
  static auto _parse_two_scalars(const char *&p, const char *end, Scalar &u,
                                 Scalar &v) -> bool {
    p = _skip_ws(p, end);
    auto r = tf::external::fast_float::from_chars(p, end, u);
    if (r.ec != std::errc{})
      return false;
    p = r.ptr;
    p = _skip_ws(p, end);
    r = tf::external::fast_float::from_chars(p, end, v);
    if (r.ec != std::errc{})
      return false;
    p = r.ptr;
    return true;
  }

  // ---------- Parse helpers ----------
  template <typename Scalar>
  static auto _parse_three_scalars(const char *&p, const char *end, Scalar &x,
                                   Scalar &y, Scalar &z) -> bool {
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

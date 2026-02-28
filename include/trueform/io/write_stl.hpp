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
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/frame_of.hpp"
#include "../core/policy/normals.hpp"
#include "../core/polygons.hpp"
#include "../core/static_size.hpp"
#include "../core/transformed.hpp"
#include "../core/views/zip.hpp"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>

namespace tf {

namespace io {

// Binary STL triangle struct (50 bytes total)
// Using byte array for portability instead of #pragma pack
struct stl_triangle {
  std::uint8_t data[50]; // Raw bytes: 12 (normal) + 36 (vertices) + 2 (attr)

  auto set_normal(float x, float y, float z) -> void {
    std::memcpy(data + 0, &x, 4);
    std::memcpy(data + 4, &y, 4);
    std::memcpy(data + 8, &z, 4);
  }

  auto set_vertex(std::size_t vertex_idx, float x, float y, float z) -> void {
    std::size_t offset = 12 + (vertex_idx * 12);
    std::memcpy(data + offset + 0, &x, 4);
    std::memcpy(data + offset + 4, &y, 4);
    std::memcpy(data + offset + 8, &z, 4);
  }

  auto set_attr(std::uint16_t value) -> void {
    std::memcpy(data + 48, &value, 2);
  }
};
static_assert(sizeof(stl_triangle) == 50, "stl_triangle must be 50 bytes");

} // namespace io

/// @ingroup io
/// @brief Serialize polygons to a binary STL buffer.
///
/// Builds the complete binary STL representation in memory.
/// If the polygons have normals, they are written; otherwise zero normals.
/// When tagged with a frame, points and normals are transformed.
///
/// @tparam Byte The byte type for the output buffer (default: char).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D triangles).
/// @return Buffer containing the binary STL data.
template <typename Byte = char, typename Policy>
auto write_stl_to_buffer(const tf::polygons<Policy> &polygons)
    -> tf::buffer<Byte> {
  static_assert(sizeof(Byte) == 1, "Byte type must be 1 byte");
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_stl requires 3D polygons");
  using polygon_t = decltype(polygons[0]);
  static_assert(tf::static_size_v<polygon_t> == 3,
                "write_stl requires triangular polygons (size 3)");

  const auto &frame = tf::frame_of(polygons);
  std::uint32_t triangle_count = static_cast<std::uint32_t>(polygons.size());
  std::size_t total_size = 84 + (50 * static_cast<std::size_t>(triangle_count));

  tf::buffer<Byte> output;
  output.allocate(total_size);

  // 80-byte zero header
  std::memset(output.data(), 0, 80);
  // 4-byte triangle count (little-endian)
  std::memcpy(output.data() + 80, &triangle_count, 4);

  // View triangle region as stl_triangle array
  auto *triangles =
      reinterpret_cast<io::stl_triangle *>(output.data() + 84);
  auto tri_range = tf::make_range(triangles, triangle_count);

  if constexpr (tf::has_normals_policy<Policy>) {
    tf::parallel_for_each(
        tf::zip(polygons, polygons.normals(), tri_range),
        [&frame](auto tuple) {
          auto &&[polygon, normal, out_struct] = tuple;
          auto tn = tf::transformed_normal(normal, frame);
          out_struct.set_normal(static_cast<float>(tn[0]),
                                static_cast<float>(tn[1]),
                                static_cast<float>(tn[2]));
          for (std::size_t i = 0; i < 3; ++i) {
            auto tp = tf::transformed(polygon[i], frame);
            out_struct.set_vertex(i, static_cast<float>(tp[0]),
                                  static_cast<float>(tp[1]),
                                  static_cast<float>(tp[2]));
          }
          out_struct.set_attr(0);
        },
        tf::checked);
  } else {
    tf::parallel_for_each(
        tf::zip(polygons, tri_range),
        [&frame](auto tuple) {
          auto &&[polygon, out_struct] = tuple;
          out_struct.set_normal(0.0f, 0.0f, 0.0f);
          for (std::size_t i = 0; i < 3; ++i) {
            auto tp = tf::transformed(polygon[i], frame);
            out_struct.set_vertex(i, static_cast<float>(tp[0]),
                                  static_cast<float>(tp[1]),
                                  static_cast<float>(tp[2]));
          }
          out_struct.set_attr(0);
        },
        tf::checked);
  }

  return output;
}

/// @ingroup io
/// @brief Write polygons to binary STL file.
///
/// Writes binary STL format. If the polygons have normals,
/// they are written; otherwise zero normals are used.
/// When the polygons are tagged with a frame, points and normals
/// are transformed before writing.
///
/// @note Files < 500MB use the buffered path via write_stl_to_buffer;
///       larger files use sequential streaming.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D triangles).
/// @param filename Output filename (.stl appended if missing).
/// @return true if write succeeded, false otherwise.
template <typename Policy>
auto write_stl(const tf::polygons<Policy> &polygons, std::string filename)
    -> bool {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_stl requires 3D polygons");
  using polygon_t = decltype(polygons[0]);
  static_assert(tf::static_size_v<polygon_t> == 3,
                "write_stl requires triangular polygons (size 3)");

  // Ensure filename ends with .stl
  if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".stl") {
    filename += ".stl";
  }

  std::uint32_t triangle_count = static_cast<std::uint32_t>(polygons.size());
  std::size_t file_size = 84 + (50 * static_cast<std::size_t>(triangle_count));
  constexpr std::size_t max_buffer_size = 500 * 1024 * 1024; // 500 MB

  if (file_size < max_buffer_size) {
    // ========== PARALLEL BUFFERED PATH ==========
    auto buf = write_stl_to_buffer(polygons);

    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;
    file.write(buf.data(), static_cast<std::streamsize>(buf.size()));
    return !!file;

  } else {
    // ========== STREAMING PATH (>= 500MB) ==========
    const auto &frame = tf::frame_of(polygons);

    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    char header[80] = {};
    file.write(header, 80);
    if (!file) return false;

    file.write(reinterpret_cast<const char *>(&triangle_count), 4);
    if (!file) return false;

    io::stl_triangle tri;

    if constexpr (tf::has_normals_policy<Policy>) {
      for (const auto &[triangle, normal] :
           tf::zip(polygons, polygons.normals())) {
        auto tn = tf::transformed_normal(normal, frame);
        tri.set_normal(static_cast<float>(tn[0]),
                       static_cast<float>(tn[1]),
                       static_cast<float>(tn[2]));
        for (std::size_t i = 0; i < 3; ++i) {
          auto tp = tf::transformed(triangle[i], frame);
          tri.set_vertex(i, static_cast<float>(tp[0]),
                         static_cast<float>(tp[1]),
                         static_cast<float>(tp[2]));
        }
        tri.set_attr(0);
        file.write(reinterpret_cast<const char *>(&tri.data), 50);
        if (!file) return false;
      }
    } else {
      for (const auto &triangle : polygons) {
        tri.set_normal(0.0f, 0.0f, 0.0f);
        for (std::size_t i = 0; i < 3; ++i) {
          auto tp = tf::transformed(triangle[i], frame);
          tri.set_vertex(i, static_cast<float>(tp[0]),
                         static_cast<float>(tp[1]),
                         static_cast<float>(tp[2]));
        }
        tri.set_attr(0);
        file.write(reinterpret_cast<const char *>(&tri.data), 50);
        if (!file) return false;
      }
    }

    return true;
  }
}

} // namespace tf

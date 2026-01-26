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

/// @ingroup io
/// @brief Write polygons to binary STL file.
///
/// Writes binary STL format. If the polygons have normals,
/// they are written; otherwise zero normals are used.
/// When the polygons are tagged with a frame, points and normals
/// are transformed before writing.
///
/// @note Files < 500MB use parallel buffered writing; larger files use streaming.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D triangles).
/// @param filename Output filename (.stl appended if missing).
/// @return true if write succeeded, false otherwise.
template <typename Policy>
auto write_stl(const tf::polygons<Policy> &polygons, std::string filename)
    -> bool {
  // Validate dimensions (must be 3D)
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_stl requires 3D polygons");

  // Validate polygon size (must be triangles)
  using polygon_t = decltype(polygons[0]);
  static_assert(tf::static_size_v<polygon_t> == 3,
                "write_stl requires triangular polygons (size 3)");

  // Binary STL triangle struct (50 bytes total)
  // Using byte array for portability instead of #pragma pack
  struct stl_triangle {
    std::uint8_t data[50]; // Raw bytes: 12 (normal) + 36 (vertices) + 2 (attr)

    // Write normal at bytes 0-11
    auto set_normal(float x, float y, float z) -> void {
      std::memcpy(data + 0, &x, 4);
      std::memcpy(data + 4, &y, 4);
      std::memcpy(data + 8, &z, 4);
    }

    // Write vertex at bytes 12 + (vertex_idx * 12) + (coord_idx * 4)
    auto set_vertex(std::size_t vertex_idx, float x, float y, float z) -> void {
      std::size_t offset = 12 + (vertex_idx * 12);
      std::memcpy(data + offset + 0, &x, 4);
      std::memcpy(data + offset + 4, &y, 4);
      std::memcpy(data + offset + 8, &z, 4);
    }

    // Write attribute at bytes 48-49
    auto set_attr(std::uint16_t value) -> void {
      std::memcpy(data + 48, &value, 2);
    }
  };
  static_assert(sizeof(stl_triangle) == 50, "stl_triangle must be 50 bytes");

  // Ensure filename ends with .stl
  if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".stl") {
    filename += ".stl";
  }

  // Get frame for transformation (identity if no frame policy)
  const auto &frame = tf::frame_of(polygons);

  // Count triangles
  std::uint32_t triangle_count = static_cast<std::uint32_t>(polygons.size());

  // Calculate file size: 80 byte header + 4 byte count + 50 bytes per triangle
  std::size_t file_size = 84 + (50 * static_cast<std::size_t>(triangle_count));
  constexpr std::size_t max_buffer_size = 500 * 1024 * 1024; // 500 MB

  // Choose path based on file size
  if (file_size < max_buffer_size) {
    // ========== PARALLEL BUFFERED PATH ==========
    // For files < 500MB, buffer entire file and fill in parallel

    // Allocate buffer for all triangles
    tf::buffer<stl_triangle> triangles;
    triangles.allocate(triangle_count);

    // Fill triangles buffer in parallel
    if constexpr (tf::has_normals_policy<Policy>) {
      // With normals - zip polygons, normals, and output buffer

      tf::parallel_for_each(
          tf::zip(polygons, polygons.normals(), triangles),
          [&frame](auto tuple) {
            auto &&[polygon, normal, out_struct] = tuple;

            // Transform and write normal
            auto transformed_normal = tf::transformed_normal(normal, frame);
            out_struct.set_normal(static_cast<float>(transformed_normal[0]),
                                  static_cast<float>(transformed_normal[1]),
                                  static_cast<float>(transformed_normal[2]));

            // Transform and write vertices
            for (std::size_t i = 0; i < 3; ++i) {
              auto transformed_pt = tf::transformed(polygon[i], frame);
              out_struct.set_vertex(i, static_cast<float>(transformed_pt[0]),
                                    static_cast<float>(transformed_pt[1]),
                                    static_cast<float>(transformed_pt[2]));
            }

            // Attribute (always 0)
            out_struct.set_attr(0);
          }, tf::checked);
    } else {

      tf::parallel_for_each(tf::zip(polygons, triangles), [&frame](auto tuple) {
        auto &&[polygon, out_struct] = tuple;

        // Zero normal
        out_struct.set_normal(0.0f, 0.0f, 0.0f);

        // Transform and write vertices
        for (std::size_t i = 0; i < 3; ++i) {
          auto transformed_pt = tf::transformed(polygon[i], frame);
          out_struct.set_vertex(i, static_cast<float>(transformed_pt[0]),
                                static_cast<float>(transformed_pt[1]),
                                static_cast<float>(transformed_pt[2]));
        }

        // Attribute (always 0)
        out_struct.set_attr(0);
      }, tf::checked);
    }

    // Open file and write everything at once
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      return false;
    }

    // Write 80-byte header (zeros)
    char header[80] = {};
    file.write(header, 80);
    if (!file) {
      return false;
    }

    // Write triangle count
    file.write(reinterpret_cast<const char *>(&triangle_count), 4);
    if (!file) {
      return false;
    }

    // Write all triangles in one operation (data is already in byte array format)
    file.write(reinterpret_cast<const char *>(triangles.begin()),
               triangle_count * 50);
    if (!file) {
      return false;
    }

    return true;

  } else {
    // ========== STREAMING PATH ==========
    // For files >= 500MB, use sequential struct-based writes

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
      return false;
    }

    // Write 80-byte header (zeros)
    char header[80] = {};
    file.write(header, 80);
    if (!file) {
      return false;
    }

    // Write triangle count
    file.write(reinterpret_cast<const char *>(&triangle_count), 4);
    if (!file) {
      return false;
    }

    // Write triangles sequentially using struct for efficiency
    stl_triangle tri;

    if constexpr (tf::has_normals_policy<Policy>) {
      // With normals
      for (const auto &[triangle, normal] :
           tf::zip(polygons, polygons.normals())) {
        // Transform and write normal
        auto transformed_normal = tf::transformed_normal(normal, frame);
        tri.set_normal(static_cast<float>(transformed_normal[0]),
                       static_cast<float>(transformed_normal[1]),
                       static_cast<float>(transformed_normal[2]));

        // Transform and write vertices
        for (std::size_t i = 0; i < 3; ++i) {
          auto transformed_pt = tf::transformed(triangle[i], frame);
          tri.set_vertex(i, static_cast<float>(transformed_pt[0]),
                         static_cast<float>(transformed_pt[1]),
                         static_cast<float>(transformed_pt[2]));
        }

        // Attribute (always 0)
        tri.set_attr(0);

        // Write triangle as single struct (50 bytes)
        file.write(reinterpret_cast<const char *>(&tri.data), 50);
        if (!file) {
          return false;
        }
      }
    } else {
      // Without normals - write zeros
      for (const auto &triangle : polygons) {
        // Zero normal
        tri.set_normal(0.0f, 0.0f, 0.0f);

        // Transform and write vertices
        for (std::size_t i = 0; i < 3; ++i) {
          auto transformed_pt = tf::transformed(triangle[i], frame);
          tri.set_vertex(i, static_cast<float>(transformed_pt[0]),
                         static_cast<float>(transformed_pt[1]),
                         static_cast<float>(transformed_pt[2]));
        }

        // Attribute (always 0)
        tri.set_attr(0);

        // Write triangle as single struct (50 bytes)
        file.write(reinterpret_cast<const char *>(&tri.data), 50);
        if (!file) {
          return false;
        }
      }
    }

    return true;
  }
}

} // namespace tf

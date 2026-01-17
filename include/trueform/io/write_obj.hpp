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
#include "../core/polygons.hpp"
#include "../core/transformed.hpp"
#include "../core/views/enumerate.hpp"
#include "./float_to_chars.hpp"
#include <charconv>
#include <fstream>
#include <string>

namespace tf {

/// @ingroup io
/// @brief Write polygons to ASCII OBJ file.
///
/// Writes OBJ format with 1-based indices.
/// When the polygons are tagged with a frame, points are transformed
/// before writing.
///
/// @note Uses parallel writing with offset buffers for optimal speed.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D).
/// @param filename Output filename (.obj appended if missing).
/// @return true if write succeeded, false otherwise.
template <typename Policy>
auto write_obj(const tf::polygons<Policy> &polygons, std::string filename)
    -> bool {
  // Validate dimensions (must be 3D)
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_obj requires 3D polygons");

  // Ensure filename ends with .obj
  if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".obj") {
    filename += ".obj";
  }

  // Get frame for transformation (identity if no frame policy)
  const auto &frame = tf::frame_of(polygons);

  const std::size_t num_points = polygons.points().size();
  const std::size_t num_faces = polygons.faces().size();

  if (num_points == 0 || num_faces == 0)
    return false;

  // ========== PASS 1: Compute line sizes into offset arrays ==========

  // Allocate offset buffers (sizes go into [1..n], then prefix sum)
  tf::buffer<std::size_t> point_offsets;
  tf::buffer<std::size_t> face_offsets;
  point_offsets.allocate(num_points + 1);
  face_offsets.allocate(num_faces + 1);

  // Compute point line sizes in parallel (store in [1..n])
  // Format: "v x y z\n"
  tf::parallel_for_each(
      tf::enumerate(polygons.points()),
      [&point_offsets, &frame](auto pair) {
        auto &&[idx, point] = pair;
        auto transformed_pt = tf::transformed(point, frame);

        // Compute size: "v " + x + " " + y + " " + z + "\n"
        char temp[128];
        std::size_t size = 2; // "v "

        char *end = io::float_to_chars(temp, temp + 64,
                                        static_cast<float>(transformed_pt[0]));
        size += end - temp;
        size += 1; // " "

        end = io::float_to_chars(temp, temp + 64,
                                 static_cast<float>(transformed_pt[1]));
        size += end - temp;
        size += 1; // " "

        end = io::float_to_chars(temp, temp + 64,
                                 static_cast<float>(transformed_pt[2]));
        size += end - temp;
        size += 1; // "\n"

        point_offsets[idx + 1] = size;
      },
      tf::checked);

  // Compute face line sizes in parallel (store in [1..n])
  // Format: "f i1 i2 i3 ...\n"
  tf::parallel_for_each(
      tf::enumerate(polygons.faces()),
      [&face_offsets](auto pair) {
        auto &&[idx, face] = pair;

        // Compute size: "f" + for each index: " " + index
        char temp[32];
        std::size_t size = 1; // "f"

        for (const auto &vertex_idx : face) {
          size += 1; // " "
          // OBJ uses 1-based indices
          auto res =
              std::to_chars(temp, temp + 32, static_cast<int>(vertex_idx) + 1);
          size += res.ptr - temp;
        }
        size += 1; // "\n"

        face_offsets[idx + 1] = size;
      },
      tf::checked);

  // ========== Convert sizes to offsets (in-place prefix sum) ==========

  // Point offsets
  point_offsets[0] = 0;
  for (std::size_t i = 1; i <= num_points; ++i) {
    point_offsets[i] += point_offsets[i - 1];
  }

  // Face offsets (start after all points)
  face_offsets[0] = point_offsets[num_points];
  for (std::size_t i = 1; i <= num_faces; ++i) {
    face_offsets[i] += face_offsets[i - 1];
  }

  const std::size_t total_size = face_offsets[num_faces];

  // ========== Allocate output buffer ==========

  tf::buffer<char> output;
  output.allocate(total_size);

  // ========== PASS 2: Write in parallel ==========

  // Write points in parallel
  tf::parallel_for_each(
      tf::enumerate(polygons.points()),
      [&output, &point_offsets, &frame](auto pair) {
        auto &&[idx, point] = pair;
        auto transformed_pt = tf::transformed(point, frame);

        char *ptr = &output[point_offsets[idx]];
        *ptr++ = 'v';
        *ptr++ = ' ';

        ptr = io::float_to_chars(ptr, ptr + 64,
                                 static_cast<float>(transformed_pt[0]));
        *ptr++ = ' ';

        ptr = io::float_to_chars(ptr, ptr + 64,
                                 static_cast<float>(transformed_pt[1]));
        *ptr++ = ' ';

        ptr = io::float_to_chars(ptr, ptr + 64,
                                 static_cast<float>(transformed_pt[2]));
        *ptr++ = '\n';
      },
      tf::checked);

  // Write faces in parallel
  tf::parallel_for_each(
      tf::enumerate(polygons.faces()),
      [&output, &face_offsets](auto pair) {
        auto &&[idx, face] = pair;

        char *ptr = &output[face_offsets[idx]];
        *ptr++ = 'f';

        for (const auto &vertex_idx : face) {
          *ptr++ = ' ';
          // OBJ uses 1-based indices
          auto res =
              std::to_chars(ptr, ptr + 32, static_cast<int>(vertex_idx) + 1);
          ptr = res.ptr;
        }
        *ptr++ = '\n';
      },
      tf::checked);

  // ========== Write to file ==========

  std::ofstream file(filename, std::ios::binary);
  if (!file)
    return false;

  file.write(output.begin(), static_cast<std::streamsize>(total_size));
  if (!file)
    return false;

  return true;
}

} // namespace tf

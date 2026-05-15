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
#include "./banner.hpp"
#include "./float_to_chars.hpp"
#include <charconv>
#include <cstring>
#include <fstream>
#include <string>

namespace tf {

/// @ingroup io
/// @brief Serialize polygons to an ASCII OBJ buffer.
///
/// Builds the complete OBJ representation in memory using a parallel
/// two-pass algorithm (compute sizes, prefix sum, parallel write).
/// When the polygons are tagged with a frame, points are transformed.
///
/// @tparam Byte The byte type for the output buffer (default: char).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D).
/// @return Buffer containing the ASCII OBJ data, or empty buffer on failure.
template <typename Byte = char, typename Policy>
auto write_obj_to_buffer(const tf::polygons<Policy> &polygons)
    -> tf::buffer<Byte> {
  static_assert(sizeof(Byte) == 1, "Byte type must be 1 byte");
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_obj requires 3D polygons");

  const auto &frame = tf::frame_of(polygons);

  const std::size_t num_points = polygons.points().size();
  const std::size_t num_faces = polygons.faces().size();

  if (num_points == 0 || num_faces == 0)
    return {};

  // ========== PASS 1: Compute line sizes into offset arrays ==========

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

        char temp[128];
        std::size_t size = 2; // "v "

        char *end = io::scalar_to_chars(temp, temp + 64, transformed_pt[0]);
        size += end - temp;
        size += 1; // " "

        end = io::scalar_to_chars(temp, temp + 64, transformed_pt[1]);
        size += end - temp;
        size += 1; // " "

        end = io::scalar_to_chars(temp, temp + 64, transformed_pt[2]);
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

        char temp[32];
        std::size_t size = 1; // "f"

        for (const auto &vertex_idx : face) {
          size += 1; // " "
          auto res =
              std::to_chars(temp, temp + 32, static_cast<int>(vertex_idx) + 1);
          size += res.ptr - temp;
        }
        size += 1; // "\n"

        face_offsets[idx + 1] = size;
      },
      tf::checked);

  // ========== Convert sizes to offsets (in-place prefix sum) ==========

  // Reserve room for the leading "# trueform ...\n" banner line.
  const std::size_t banner_size = std::strlen(io::trueform_banner) + 3; // "# " + banner + "\n"
  point_offsets[0] = banner_size;
  for (std::size_t i = 1; i <= num_points; ++i) {
    point_offsets[i] += point_offsets[i - 1];
  }

  face_offsets[0] = point_offsets[num_points];
  for (std::size_t i = 1; i <= num_faces; ++i) {
    face_offsets[i] += face_offsets[i - 1];
  }

  const std::size_t total_size = face_offsets[num_faces];

  // ========== Allocate output buffer ==========

  tf::buffer<Byte> output;
  output.allocate(total_size);

  // Write banner: "# trueform - ...\n"
  {
    char *ptr = reinterpret_cast<char *>(output.data());
    *ptr++ = '#';
    *ptr++ = ' ';
    auto blen = std::strlen(io::trueform_banner);
    std::memcpy(ptr, io::trueform_banner, blen);
    ptr += blen;
    *ptr = '\n';
  }

  // ========== PASS 2: Write in parallel ==========

  // Write points in parallel
  tf::parallel_for_each(
      tf::enumerate(polygons.points()),
      [&output, &point_offsets, &frame](auto pair) {
        auto &&[idx, point] = pair;
        auto transformed_pt = tf::transformed(point, frame);

        char *ptr = reinterpret_cast<char *>(&output[point_offsets[idx]]);
        *ptr++ = 'v';
        *ptr++ = ' ';

        ptr = io::scalar_to_chars(ptr, ptr + 64, transformed_pt[0]);
        *ptr++ = ' ';

        ptr = io::scalar_to_chars(ptr, ptr + 64, transformed_pt[1]);
        *ptr++ = ' ';

        ptr = io::scalar_to_chars(ptr, ptr + 64, transformed_pt[2]);
        *ptr++ = '\n';
      },
      tf::checked);

  // Write faces in parallel
  tf::parallel_for_each(
      tf::enumerate(polygons.faces()),
      [&output, &face_offsets](auto pair) {
        auto &&[idx, face] = pair;

        char *ptr = reinterpret_cast<char *>(&output[face_offsets[idx]]);
        *ptr++ = 'f';

        for (const auto &vertex_idx : face) {
          *ptr++ = ' ';
          auto res =
              std::to_chars(ptr, ptr + 32, static_cast<int>(vertex_idx) + 1);
          ptr = res.ptr;
        }
        *ptr++ = '\n';
      },
      tf::checked);

  return output;
}

/// @ingroup io
/// @brief Write polygons to ASCII OBJ file.
///
/// Writes OBJ format with 1-based indices.
/// When the polygons are tagged with a frame, points are transformed
/// before writing.
///
/// @tparam Policy The policy type of the polygons.
/// @param polygons The @ref tf::polygons to write (must be 3D).
/// @param filename Output filename (.obj appended if missing).
/// @return true if write succeeded, false otherwise.
template <typename Policy>
auto write_obj(const tf::polygons<Policy> &polygons, std::string filename)
    -> bool {
  static_assert(tf::coordinate_dims_v<Policy> == 3,
                "write_obj requires 3D polygons");

  if (filename.size() < 4 || filename.substr(filename.size() - 4) != ".obj") {
    filename += ".obj";
  }

  auto buf = write_obj_to_buffer(polygons);
  if (buf.size() == 0)
    return false;

  std::ofstream file(filename, std::ios::binary);
  if (!file)
    return false;

  file.write(reinterpret_cast<const char *>(buf.data()),
             static_cast<std::streamsize>(buf.size()));
  return !!file;
}

} // namespace tf

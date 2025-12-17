/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/polygons_buffer.hpp"
#include "./obj_reader.hpp"

namespace tf {
// Read OBJ file with dynamic Ngon (supports mixed polygon sizes)
template <typename Index>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, float, 3, tf::dynamic_size> {
  tf::polygons_buffer<Index, float, 3, tf::dynamic_size> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out.points_buffer(), out.faces_buffer())) {
    return {}; // Return empty on error
  }
  return out;
}

// Read OBJ file with fixed Ngon (e.g., triangles only, quads only)
template <typename Index, std::size_t Ngon>
auto read_obj(std::string_view file_path)
    -> tf::polygons_buffer<Index, float, 3, Ngon> {
  tf::polygons_buffer<Index, float, 3, Ngon> out;
  tf::io::obj_reader reader;
  if (!reader.read(file_path, out.points_buffer(), out.faces_buffer())) {
    return {}; // Return empty on error
  }
  return out;
}
} // namespace tf

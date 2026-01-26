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
#include "../clean/soup/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "./stl_point_collector.hpp"

namespace tf {

/// @ingroup io
/// @brief Read STL file into triangular polygons.
///
/// Reads binary or ASCII STL format automatically.
/// Deduplicates vertices during loading via @ref tf::clean::polygon_soup.
///
/// @tparam Index The index type (defaults to int).
/// @param file_path Path to the STL file.
/// @return A @ref tf::polygons_buffer containing 3D triangular mesh with float coordinates.
template <typename Index = int>
auto read_stl(std::string_view file_path)
    -> tf::polygons_buffer<Index, float, 3, 3> {
  tf::buffer<float> buffer;
  tf::io::stl_point_collector collector;
  collector.read(file_path, buffer);
  tf::clean::polygon_soup<Index, float, 3, 3> cleaned;
  cleaned.build(std::move(buffer));
  return cleaned;
}
} // namespace tf

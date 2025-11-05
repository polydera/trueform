/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../clean/soup/polygons.hpp"
#include "../core/polygons_buffer.hpp"
#include "./stl_point_collector.hpp"

namespace tf {
template <typename Index>
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

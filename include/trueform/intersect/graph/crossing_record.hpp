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

#include "../../exact/int128.hpp"

#include <array>
#include <cstdint>
#include <tuple>

namespace tf::intersect::graph {

template <typename Index> struct face_id {
  short tag;
  Index object;

  auto operator<(const face_id &o) const {
    return std::tie(tag, object) < std::tie(o.tag, o.object);
  }
  auto operator==(const face_id &o) const {
    return tag == o.tag && object == o.object;
  }
};

template <typename Index> struct crossing_record {
  Index edge_a;
  Index edge_b;
  Index point_a;
  Index point_b;
  std::array<face_id<Index>, 3> triple;
  enum type_t : uint8_t { ee, ve, vv } type;
};

template <typename Index> struct edge_split_entry {
  Index edge_id;
  Index point_id;

  auto operator<(const edge_split_entry &o) const {
    return std::tie(edge_id, point_id) < std::tie(o.edge_id, o.point_id);
  }
  auto operator==(const edge_split_entry &o) const {
    return edge_id == o.edge_id && point_id == o.point_id;
  }
};

template <typename Index> struct split_point {
  Index point_id;
  tf::exact::int128 t;
};

} // namespace tf::intersect::graph

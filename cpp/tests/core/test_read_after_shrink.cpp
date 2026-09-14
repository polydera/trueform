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
#include "carriers.hpp"
#include "nd_array.hpp"

#include "trueform/cpp/arrangement/mesh_arrangements.hpp"
#include "trueform/cpp/arrangement/polygon_arrangements.hpp"
#include "trueform/cpp/clean/mesh.hpp"
#include "trueform/cpp/csg/outer_shell.hpp"
#include "trueform/cpp/geometry/area.hpp"
#include "trueform/cpp/geometry/laplacian_smoothed.hpp"
#include "trueform/cpp/geometry/mean_edge_length.hpp"
#include "trueform/cpp/geometry/normals.hpp"
#include "trueform/cpp/geometry/point_normals.hpp"
#include "trueform/cpp/geometry/positively_oriented.hpp"
#include "trueform/cpp/geometry/principal_curvatures.hpp"
#include "trueform/cpp/geometry/reverse_winding.hpp"
#include "trueform/cpp/geometry/sharp_edges.hpp"
#include "trueform/cpp/geometry/volume.hpp"
#include "trueform/cpp/intersect/intersection_curves.hpp"
#include "trueform/cpp/intersect/self_intersection_curves.hpp"
#include "trueform/cpp/io/write_obj.hpp"
#include "trueform/cpp/io/write_stl.hpp"
#include "trueform/cpp/reindex/by_ids.hpp"
#include "trueform/cpp/spatial/intersects.hpp"
#include "trueform/cpp/topology/boundary_curves.hpp"
#include "trueform/cpp/topology/boundary_edges.hpp"
#include "trueform/cpp/topology/boundary_paths.hpp"
#include "trueform/cpp/topology/euler_characteristic.hpp"
#include "trueform/cpp/topology/is_closed.hpp"
#include "trueform/cpp/topology/is_manifold.hpp"
#include "trueform/cpp/topology/k_rings.hpp"
#include "trueform/cpp/topology/non_manifold_edges.hpp"
#include "trueform/cpp/topology/orient_faces_consistently.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace {

using shrink_owned_t =
    tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float>;

/// Eight points under two faces, then three points under the same two faces:
/// the second face names points 5, 6 and 7, which the geometry no longer has.
auto shrink_out_of_reach() -> shrink_owned_t {
  shrink_owned_t owned{
      tf::cpp::test::polygons_of<tf::cpp::default_index_t, float>(
          {0, 1, 2, 5, 6, 7},
          {0,  0,  0,   10, 20,  30,  20, 40,  60,  30, 60,  90,
           40, 80, 120, 50, 100, 150, 60, 120, 180, 70, 140, 210})};
  auto &coordinates = owned.polygons.points_buffer().data_buffer();
  coordinates.allocate(9);
  const float shrunk[] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  std::copy(std::begin(shrunk), std::end(shrunk), coordinates.begin());
  owned.cache.points_changed();
  return owned;
}

template <typename Read> auto shrink_refuses(Read &&read) -> bool {
  const auto owned = shrink_out_of_reach();
  const auto mesh = owned.mesh();
  try {
    static_cast<void>(std::forward<Read>(read)(mesh));
  } catch (const std::out_of_range &) {
    return true;
  }
  return false;
}

} // namespace

/// THE DOOR'S LAW: a read refuses only what a later restatement of the points
/// put out of reach — and EVERY read refuses it. An entry that reads the
/// geometry without asking the cache walks coordinates the mesh does not have.
TEST_CASE("every mesh read refuses points a later set_points put out of reach",
          "[cpp][core][mesh][door][validation]") {
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::area(value); }));
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::volume(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::mean_edge_length(value); }));
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::is_closed(value); }));
  CHECK(
      shrink_refuses([](auto &value) { return tf::cpp::is_manifold(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::euler_characteristic(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::non_manifold_edges(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::boundary_edges(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::boundary_paths(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::boundary_curves(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::k_rings(value, 1, true); }));
  CHECK(shrink_refuses([](auto &value) {
    return tf::cpp::sharp_edges(value, tf::rad<float>{0.5f});
  }));
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::normals(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::point_normals(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::principal_curvatures(value, 2); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::laplacian_smoothed(value, 1, 0.5f); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::orient_faces_consistently(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::positively_oriented(value, true); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::reverse_winding(value); }));
  CHECK(shrink_refuses([](auto &value) {
    return tf::cpp::cleaned_mesh(value, 0.0f, true, true);
  }));
  CHECK(shrink_refuses([](auto &value) {
    return tf::cpp::reindexed_by_ids(
        value,
        tf::cpp::test::make_nd_array<tf::cpp::default_index_t>({0}, {1}));
  }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::intersects(value, value); }));
  CHECK(
      shrink_refuses([](auto &value) { return tf::cpp::outer_shell(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::polygon_arrangements(value); }));
  CHECK(shrink_refuses(
      [](auto &value) { return tf::cpp::self_intersection_curves(value); }));
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::write_obj(value); }));
  CHECK(shrink_refuses([](auto &value) { return tf::cpp::write_stl(value); }));
}

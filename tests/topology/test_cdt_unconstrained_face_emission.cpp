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
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/delaunay_vertex_policy.hpp>
#include <trueform/topology/cdt/materialize_unconstrained_delaunay_faces.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using test_index = std::uint32_t;

struct test_edge {
  test_index vertex;
  test_index prev;
  test_index next;
};

struct test_site {
  test_index output;
};

struct test_owner {
  using index_type = test_index;
  using vertex_policy = tf::topology::cdt::original_input_vertex_policy;

  static constexpr index_type none = std::numeric_limits<index_type>::max();

  std::vector<test_edge> _edges;
  std::vector<test_site> _sites;
  tf::blocked_buffer<index_type, 3> _faces;
  tf::buffer<std::size_t> _face_offsets;
  tf::buffer<std::uint8_t> _face_visited;
};

template <typename ExecutionPolicy>
auto verify_triangle_emission(std::size_t edge_count) -> void {
  test_owner owner;
  owner._sites = {{10}, {11}, {12}};
  owner._edges.assign(edge_count, {test_owner::none, 0, 0});

  // Interior orbit: 0 -> 2 -> 4. Exterior orbit: 1 -> 5 -> 3.
  owner._edges[0] = {0, 5, 0};
  owner._edges[1] = {1, 2, 0};
  owner._edges[2] = {1, 1, 0};
  owner._edges[3] = {2, 4, 0};
  owner._edges[4] = {2, 3, 0};
  owner._edges[5] = {0, 0, 0};

  tf::topology::cdt::materialize_unconstrained_delaunay_faces<ExecutionPolicy>(
      owner, test_index(1));

  REQUIRE(owner._faces.size() == 1);
  REQUIRE(owner._faces[0][0] == 10);
  REQUIRE(owner._faces[0][1] == 11);
  REQUIRE(owner._faces[0][2] == 12);

  // The owner's visit bytes carry the traversal; topology identities are
  // left intact.
  REQUIRE(owner._edges[0].vertex == 0);
  REQUIRE(owner._edges[1].vertex == 1);
  REQUIRE(owner._edges[2].vertex == 1);
  REQUIRE(owner._edges[3].vertex == 2);
  REQUIRE(owner._edges[4].vertex == 2);
  REQUIRE(owner._edges[5].vertex == 0);
}

} // namespace

TEST_CASE("serial unconstrained face emission keeps topology identities",
          "[constrained_delaunay]") {
  verify_triangle_emission<tf::topology::cdt::serial_delaunay_execution_policy>(
      6);
}

TEST_CASE("parallel unconstrained face emission keeps topology identities",
          "[constrained_delaunay]") {
  verify_triangle_emission<
      tf::topology::cdt::parallel_delaunay_execution_policy>(100000);
}

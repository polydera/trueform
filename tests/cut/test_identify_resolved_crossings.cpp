/**
 * @file test_identify_resolved_crossings.cpp
 * @brief tf::cut::identify_resolved_crossings — the wave's last word on a
 *        crossing it can file no split for.
 *
 * A region whose walk passes over one physical edge twice crosses itself
 * below the lattice, and the position of that crossing on the transported
 * grid IS a vertex that already exists: there is no split to broadcast, so
 * the wave stalls and a preserved build would refuse forever. The resolving
 * build completes by creating a point, and that point is kept only because
 * the same snap that defines split identity everywhere else names an
 * existing vertex for it. One parent naming an endpoint is enough; two that
 * name different vertices state a weld; a crossing genuinely interior to
 * both parents names nothing, and the region refuses as it did before.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <trueform/core/buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/core/views/constant.hpp>
#include <trueform/cut/detail/region_triangulation_types.hpp>
#include <trueform/cut/face_regions.hpp>
#include <trueform/cut/impl/identify_resolved_crossings.hpp>
#include <trueform/cut/impl/triangulate_region.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/vertex.hpp>
#include <trueform/topology/topo_id.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>

namespace {

using Index = int;
using vertex_t = tf::intersect::graph::vertex<Index>;
using vsource = tf::intersect::graph::vertex_source;
using triangle_t = std::array<vertex_t, 3>;
using position_t = std::array<int, 2>;
using constraint_t = std::array<Index, 2>;
using vertex_key_t = std::array<Index, 2>;

constexpr Index n_tags = 2;
constexpr Index tag = 0;

auto original(Index id) -> vertex_t {
  return vertex_t{vsource::original, id, {0, tf::topo_type::face}};
}

auto created(Index id) -> vertex_t {
  return vertex_t{vsource::created, id, {0, tf::topo_type::face}};
}

/// Everything a region carries into tf::cut::triangulate_region. Its walk
/// is stated in created vertices so the lattice positions are the ones
/// written here, with no conversion between them and the test.
template <typename Int> struct region_case {
  tf::buffer<tf::point<Int, 3>> extra_points;
  tf::buffer<vertex_t> loop;
  tf::buffer<Index> hole_ids;
  tf::points_buffer<Int, 3> intersection_points;
  tf::face_regions<Index, Int> regions;
  tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> splits;
  tf::buffer<tf::cut::detail::region_triangulation_merge<Index>> merges;
  tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
};

template <typename Int>
auto make_region_case(std::initializer_list<position_t> positions)
    -> region_case<Int> {
  region_case<Int> region;
  Index id = 0;
  for (const auto &position : positions) {
    region.extra_points.push_back(
        {Int(position[0]), Int(position[1]), Int(0)});
    region.loop.push_back(created(id++));
  }
  return region;
}

template <typename Int>
auto triangulate(region_case<Int> &region, bool resolve_crossings,
                 tf::buffer<triangle_t> &triangles) -> bool {
  const std::array<tf::point<Int, 3>, 3> face_points{
      {{Int(0), Int(0), Int(0)},
       {Int(65536), Int(0), Int(0)},
       {Int(0), Int(65536), Int(0)}}};
  auto apply_to_face = [](Index, Index, const auto &f) {
    const std::array<Index, 3> face{0, 1, 2};
    f(face);
  };
  auto get_mesh_point = [&](Index, Index id) -> tf::point<Int, 3> {
    return face_points[std::size_t(id)];
  };
  return tf::cut::triangulate_region(
      region.regions, tf::intersect::graph::face_descriptor<Index>{tag, 0},
      tf::make_range(region.loop), tf::make_range(region.hole_ids),
      apply_to_face, get_mesh_point, region.intersection_points.points(),
      n_tags, Index(0), region.extra_points, region.splits, region.merges,
      region.workspace, triangles, true,
      resolve_crossings ? tf::cut::detail::region_build::resolve
                        : tf::cut::detail::region_build::preserve);
}

/// The one output vertex the build created, negative unless there is
/// exactly one. Its `kept_ids()` entry is the sentinel `f().size()`, which
/// is how a point with no input of its own is stated.
template <typename Int>
auto created_output(
    const tf::cut::detail::region_triangulation_workspace<Index, Int>
        &workspace) -> Index {
  const auto &kept_ids = workspace.tri.index_map().kept_ids();
  const Index sentinel = Index(workspace.tri.index_map().f().size());
  Index found = Index(-1);
  for (Index output = 0; output < Index(kept_ids.size()); ++output)
    if (kept_ids[std::size_t(output)] == sentinel)
      found = found == Index(-1) ? output : Index(-2);
  return found;
}

/// A workspace assembled the way triangulate_region assembles one, carried
/// through the resolving build and the weld pass that precedes the
/// identification. Constraints are whole edges, so each spans its parent
/// entirely and every vertex id is its own local id.
template <typename Int> struct crossing_case {
  tf::buffer<tf::point<Int, 3>> points;
  tf::buffer<vertex_t> vertices;
  tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
  Index created = Index(-1);
};

template <typename Int>
auto resolve_crossing(std::initializer_list<position_t> positions,
                      std::initializer_list<constraint_t> constraints)
    -> crossing_case<Int> {
  using param_t = typename tf::exact::meta<Int>::param_type;
  crossing_case<Int> state;
  Index id = 0;
  for (const auto &position : positions) {
    state.points.push_back({Int(position[0]), Int(position[1]), Int(0)});
    state.workspace.pts.emplace_back(Int(position[0]), Int(position[1]));
    state.vertices.push_back(original(id++));
  }
  Index slot = 0;
  for (const auto &constraint : constraints) {
    const auto a = state.vertices[std::size_t(constraint[0])];
    const auto b = state.vertices[std::size_t(constraint[1])];
    state.workspace.cons.push_back(constraint[0]);
    state.workspace.cons.push_back(constraint[1]);
    state.workspace.cons_verts.push_back({a, b});
    state.workspace.cons_slot.push_back(slot++);
    state.workspace.cons_sub.push_back(char(0));
    state.workspace.cons_parent.push_back({a, b});
    state.workspace.cons_t.push_back({param_t(0), param_t(0)});
  }

  auto edges = tf::make_edges(tf::make_blocked_range<2>(tf::make_range(
      state.workspace.cons.data(), state.workspace.cons.size())));
  if (!state.workspace.tri.build(tf::make_points(state.workspace.pts), edges,
                                 tf::make_constant_range(true, edges.size()),
                                 true))
    return state;

  const Index n_input = Index(state.vertices.size());
  const auto &input_to_output = state.workspace.tri.index_map().f();
  const Index n_final = Index(state.workspace.tri.points().size());
  state.workspace.rep.allocate(std::size_t(n_final));
  std::fill(state.workspace.rep.begin(), state.workspace.rep.end(), Index(-1));
  for (Index input = 0; input < n_input; ++input) {
    auto &representative =
        state.workspace.rep[std::size_t(input_to_output[std::size_t(input)])];
    if (representative == Index(-1) || input < representative)
      representative = input;
  }
  state.created = created_output(state.workspace);

  auto vertex_at = [&](std::size_t index) -> const vertex_t & {
    return state.vertices[index];
  };
  auto get_point = [&](const vertex_t &vertex) -> tf::point<Int, 3> {
    return state.points[std::size_t(vertex.id)];
  };
  tf::cut::identify_resolved_crossings<Index, Int>(
      state.workspace, n_tags, tag, n_input, vertex_at, get_point);
  return state;
}

} // namespace

TEMPLATE_TEST_CASE("triangulate_region: the resolving pass keeps a region the "
                   "preserved one refuses forever",
                   "[cut][identify_resolved_crossings]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  // The walk crosses its own bottom edge at that edge's midpoint, which no
  // snap can put on an end, sixteen units below the notch tip, which is
  // under one snap step of the edge the tip sits on. Exactly one parent
  // names the crossing, and neither of them has a split to file.
  auto region = make_region_case<Int>(
      {{0, 0}, {2048, 0}, {1024, 16}, {1024, -1008}});
  tf::buffer<triangle_t> triangles;

  CHECK_FALSE(triangulate(region, false, triangles));
  CHECK(triangles.size() == 0u);

  REQUIRE(triangulate(region, true, triangles));
  const Index output = created_output(region.workspace);
  REQUIRE(output >= 0);
  const Index representative = region.workspace.rep[std::size_t(output)];
  REQUIRE(representative >= 0);
  CHECK(region.workspace.ihm.kept_ids()[std::size_t(representative)] ==
        created(2));

  // The lobe on the far side of the crossing has two corners on one vertex
  // once the crossing is identified, so it is an edge and not a triangle.
  REQUIRE(triangles.size() == 1u);
  std::array<Index, 3> corners{triangles[0][0].id, triangles[0][1].id,
                               triangles[0][2].id};
  std::sort(corners.begin(), corners.end());
  CHECK(corners == std::array<Index, 3>{0, 2, 3});
}

TEMPLATE_TEST_CASE("triangulate_region: a crossing no parent can name leaves "
                   "the region refusing",
                   "[cut][identify_resolved_crossings]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  auto region =
      make_region_case<Int>({{0, 0}, {2048, 0}, {1024, 512}, {1024, -512}});
  tf::buffer<triangle_t> triangles;

  CHECK_FALSE(triangulate(region, false, triangles));
  CHECK_FALSE(triangulate(region, true, triangles));
  const Index output = created_output(region.workspace);
  REQUIRE(output >= 0);
  CHECK(region.workspace.rep[std::size_t(output)] == Index(-1));
}

TEMPLATE_TEST_CASE("identify_resolved_crossings: one parent naming an "
                   "endpoint identifies the created point",
                   "[cut][identify_resolved_crossings]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  // The crossing lies sixteen units from one end of the edge it splits,
  // under one snap step of that edge, and halfway along the edge it
  // crosses, which is a position on that one and not an end.
  auto state = resolve_crossing<Int>(
      {{0, 0}, {1024, 0}, {512, -16}, {512, 1008}}, {{0, 1}, {2, 3}});
  REQUIRE(state.created >= 0);
  CHECK(state.workspace.rep[std::size_t(state.created)] == Index(2));
  CHECK(state.workspace.merges.size() == 0u);
}

TEMPLATE_TEST_CASE("identify_resolved_crossings: two parents naming two "
                   "vertices report the weld",
                   "[cut][identify_resolved_crossings]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  // The crossed edge is shorter than one ulp of the real it came from, so
  // it has no interior at this resolution and the crossing halfway along it
  // is its far end. The crossing edge names its own near end, and the two
  // are one point under this precision.
  auto state = resolve_crossing<Int>(
      {{0, -32}, {0, 32}, {-16, 0}, {1008, 0}}, {{0, 1}, {2, 3}});
  REQUIRE(state.created >= 0);
  CHECK(state.workspace.rep[std::size_t(state.created)] == Index(1));
  REQUIRE(state.workspace.merges.size() == 1u);
  CHECK(state.workspace.merges[0].from == vertex_key_t{tag, 2});
  CHECK(state.workspace.merges[0].to_key == vertex_key_t{tag, 1});
  CHECK(state.workspace.merges[0].to == original(1));
}

TEMPLATE_TEST_CASE("identify_resolved_crossings: a crossing interior to both "
                   "parents stays unmapped",
                   "[cut][identify_resolved_crossings]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  auto state = resolve_crossing<Int>(
      {{0, 0}, {1024, 0}, {512, -512}, {512, 512}}, {{0, 1}, {2, 3}});
  REQUIRE(state.created >= 0);
  CHECK(state.workspace.rep[std::size_t(state.created)] == Index(-1));
  CHECK(state.workspace.merges.size() == 0u);
}

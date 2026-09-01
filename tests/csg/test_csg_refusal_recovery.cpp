/**
 * @file test_csg_refusal_recovery.cpp
 * @brief The refusal wave, end to end: tf::csg_graph::failed and the mesh
 *
 * A plane whose triangulation refuses holds no product, and the recovery wave
 * is what hands it its rewritten rows and triangulates it again. What the wave
 * cannot recover the arrangement publishes in `failed()`, and this graph
 * forwards it — so a boolean answered off an empty `failed()` stands on an
 * arrangement every carrier of which holds its product.
 *
 * The scene strands a plane: a 16-stack sphere and a 4x4 tessellated cutting
 * plane, whose equator ring lands ON the plane once the FLOAT lattice
 * quantizes it. Its stranding is a property of the ARRANGEMENT alone, so both
 * readings of the cutting plane — a sheet that separates and a volume operand
 * — reach it, and both are asserted here. The double lattice never puts the
 * ring on the plane; that arm holds trivially.
 *
 * The sheet reading is the one with a boolean to check: the plane cuts the
 * sphere into two closed capped halves, each half of the polyhedral sphere's
 * own volume (~2.06 at 16x32), and the two halves account for all of it.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/geometry/ensure_positive_orientation.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>

#include "type_traits.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace {

/// The stranding scene: the sphere's equator ring is a ring of the mesh
/// because the stack count is even, and the cutting plane is tessellated, so
/// the contact is a whole tiled region rather than one edge.
template <typename Index, typename Real>
auto sphere_on_plane(int stacks, int segments, Real height) {
  auto sphere = tf::make_sphere_mesh<Index, Real>(Real(1), stacks, segments);
  tf::ensure_positive_orientation(sphere.polygons());
  auto plane = tf::make_plane_mesh<Index, Real>(Real(4), Real(4), 4, 4);
  for (auto point : plane.points_buffer())
    point[2] = point[2] + height;
  return std::make_pair(std::move(sphere), std::move(plane));
}

auto one_sheet() -> std::vector<int> { return {1}; }

template <typename Mesh> auto recovery_volume_of(const Mesh &mesh) -> double {
  return mesh.size() == 0 ? 0.0 : double(tf::signed_volume(mesh.polygons()));
}

} // namespace

// ============================================================================
// The sheet reading: the boolean answered over the stranding scene.
// ============================================================================
TEMPLATE_TEST_CASE("csg refusal: the sheet-cut sphere is whole",
                   "[csg][sheets][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto scene = sphere_on_plane<index_t, real_t>(16, 32, real_t(0));
  auto sheets = one_sheet();
  auto op0 = tf::test::make_tagged_operand(scene.first);
  auto op1 = tf::test::make_tagged_operand(scene.second);
  auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                              tf::test::sheets_of(sheets), {});

  REQUIRE(graph.failed().size() == 0);
  REQUIRE(graph.arrangement().failed().size() == 0);

  // the operand is the polyhedron, not the sphere it approximates: the halves
  // are compared against the volume the input mesh itself encloses
  const double whole = recovery_volume_of(scene.first);
  auto upper = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
  auto lower = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));

  REQUIRE(upper.size() > 0);
  REQUIRE(lower.size() > 0);
  REQUIRE(tf::is_closed(upper.polygons()));
  REQUIRE(tf::is_closed(lower.polygons()));
  REQUIRE(tf::is_manifold(upper.polygons()));
  REQUIRE(tf::is_manifold(lower.polygons()));
  // an even stack count puts a ring ON the cut, so the two halves are mirror
  // images and the cut loses nothing
  REQUIRE_THAT(recovery_volume_of(upper),
               Catch::Matchers::WithinRel(whole * 0.5, 0.01));
  REQUIRE_THAT(recovery_volume_of(lower),
               Catch::Matchers::WithinRel(whole * 0.5, 0.01));
  REQUIRE_THAT(recovery_volume_of(upper) + recovery_volume_of(lower),
               Catch::Matchers::WithinRel(whole, 0.01));
}

// ============================================================================
// The volume reading of the same scene: the stranding is the arrangement's,
// so it is reached without declaring any sheet.
// ============================================================================
TEMPLATE_TEST_CASE("csg refusal: the same scene as plain operands",
                   "[csg][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto scene = sphere_on_plane<index_t, real_t>(16, 32, real_t(0));
  auto op0 = tf::test::make_tagged_operand(scene.first);
  auto op1 = tf::test::make_tagged_operand(scene.second);
  auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                              tf::test::no_sheets(), {});

  REQUIRE(graph.failed().size() == 0);
  REQUIRE(graph.arrangement().failed().size() == 0);
}

// ============================================================================
// The neighbourhood of the trigger, answered as booleans: nothing strands and
// every cut closes.
// ============================================================================
TEMPLATE_TEST_CASE("csg refusal: the stranding ladder closes",
                   "[csg][sheets][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  // the stack count either side of the trigger (an odd count has no equator
  // ring at all), the segment count either side of it, and the plane swept
  // one tessellation pitch above and below the equator
  const int stacks[] = {14, 15, 16, 17, 18};
  const int segments[] = {30, 31, 33, 34};
  const real_t heights[] = {real_t(-0.0833333), real_t(0.0833333),
                            real_t(0.25)};
  auto sheets = one_sheet();

  const auto check = [&](int stacks_at, int segments_at, real_t height) {
    auto scene = sphere_on_plane<index_t, real_t>(stacks_at, segments_at,
                                                  height);
    auto op0 = tf::test::make_tagged_operand(scene.first);
    auto op1 = tf::test::make_tagged_operand(scene.second);
    auto graph = tf::test::build_pair_csg_graph(
        op0.form(), op1.form(), tf::test::sheets_of(sheets), {});
    REQUIRE(graph.failed().size() == 0);
    const double whole = recovery_volume_of(scene.first);
    auto upper = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
    auto lower = tf::test::csg_mesh_of(graph, tf::csg::intersection(0, 1));
    REQUIRE(upper.size() > 0);
    REQUIRE(lower.size() > 0);
    REQUIRE(tf::is_closed(upper.polygons()));
    REQUIRE(tf::is_closed(lower.polygons()));
    REQUIRE_THAT(recovery_volume_of(upper) + recovery_volume_of(lower),
                 Catch::Matchers::WithinRel(whole, 0.01));
  };

  for (const int stack : stacks) {
    INFO("stacks " << stack);
    check(stack, 32, real_t(0));
  }
  for (const int segment : segments) {
    INFO("segments " << segment);
    check(16, segment, real_t(0));
  }
  for (const real_t height : heights) {
    INFO("height " << double(height));
    check(16, 32, height);
  }
}

// ============================================================================
// The average path: the surface is read where nothing ever refuses.
// ============================================================================
TEMPLATE_TEST_CASE("csg refusal: a scene that never refuses",
                   "[csg][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto s0 = tf::make_sphere_mesh<index_t, real_t>(real_t(1), 24, 24);
  auto s1 = tf::make_sphere_mesh<index_t, real_t>(real_t(1), 24, 24);
  tf::ensure_positive_orientation(s0.polygons());
  tf::ensure_positive_orientation(s1.polygons());
  for (auto point : s1.points_buffer())
    point[0] = point[0] + real_t(0.7);

  auto op0 = tf::test::make_tagged_operand(s0);
  auto op1 = tf::test::make_tagged_operand(s1);
  auto graph = tf::test::build_pair_csg_graph(op0.form(), op1.form(),
                                              tf::test::no_sheets(), {});
  REQUIRE(graph.failed().size() == 0);
  REQUIRE(graph.arrangement().failed().size() == 0);
  REQUIRE(graph.arrangement().arrangement().census().refusals == 0);

  auto m = tf::test::csg_mesh_of(graph, tf::csg::merge(0, 1));
  REQUIRE(tf::is_closed(m.polygons()));
  REQUIRE(tf::is_manifold(m.polygons()));
}

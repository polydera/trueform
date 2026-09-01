/**
 * @file test_plane_refusal_recovery.cpp
 * @brief The refusal wave's completeness surface: tf::arrangement_graph::failed
 *
 * A plane that refuses its triangulation holds no product. The recovery wave
 * states what the refusal saw, substitutes the identities it closes, and hands
 * the carrier its rewritten rows to be triangulated again — and what the wave
 * cannot recover is published, per plane, in `failed()`. These cases stand on
 * that surface:
 *
 *   RECOVERY   a scene that strands a plane — a 16-stack sphere against a
 *              4x4 tessellated plane, whose equator ring lands ON the
 *              plane once the FLOAT lattice quantizes it. One plane
 *              refuses, one further round recovers it, and `failed()` is
 *              empty. On the double lattice the ring misses the plane,
 *              nothing refuses, and the same assertions hold trivially.
 *   LADDER     the same scene's neighbourhood — the stack and segment
 *              counts either side of the trigger, and the cutting plane
 *              swept through the sphere — where nothing may strand.
 *   AVERAGE    a scene that never refuses, so the surface is read where
 *              it costs nothing.
 *   STRANDED   the other side of the surface, stated by hand: a carrier
 *              with no constraints refuses forever, and the arrangement
 *              publishes THAT PLANE while every other plane keeps its
 *              product. No pipeline input is known to reach it, so the
 *              fixture world states it directly.
 *
 * Every case also asserts THE COMPLETENESS LAW that `failed()` empty is a
 * claim about: every plane that carries constraints emitted the triangulation
 * of them. A carrier that bounds no area emits nothing and satisfies that law
 * by doing so; it is subject to it like every other.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "arrangement_builders.hpp"
#include "plane_arrangement_generators.hpp"
#include "tagged_operand.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/range.hpp>
#include <trueform/arrangement/arrangement_config.hpp>
#include <trueform/arrangement/make_arrangement_graph.hpp>
#include <trueform/arrangement/planes/plane_arrangement.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/geometry/ensure_positive_orientation.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_plane_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>

#include "type_traits.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace {

/// What the wave costs this scene on one lattice. The float lattice is the
/// one that puts the sphere's equator ring on the cutting plane, so it is the
/// only one that refuses; the double lattice never reaches the wave.
template <typename Real> struct wave_cost;
template <> struct wave_cost<float> {
  static constexpr std::size_t rounds = 2;
  static constexpr std::size_t refusals = 1;
};
template <> struct wave_cost<double> {
  static constexpr std::size_t rounds = 1;
  static constexpr std::size_t refusals = 0;
};

/// THE COMPLETENESS LAW, read off the published surface: with `failed()`
/// empty, every carrier holds its product. A plane bounds area unless its
/// exact frame is degenerate — a LINE the local arrangement already resolved,
/// which emits nothing by design — and it carries constraints unless its
/// definition block is empty.
template <typename Graph>
auto require_every_carrier_holds_its_product(const Graph &graph) -> void {
  const auto &arrangement = graph.arrangement();
  const auto &world = graph.world();
  const auto tickets = arrangement.plane_tickets();
  using index_t = typename Graph::index_type;
  for (index_t plane = 0; plane < arrangement.n_planes(); ++plane) {
    const auto range = arrangement.plane_range(plane);
    if (range[1] != range[0])
      continue;
    const auto ticket =
        tickets.size() == 0 ? index_t(-1) : tickets[std::size_t(plane)];
    if (ticket != index_t(-1)) {
      REQUIRE(arrangement.current_plane_def_rows(ticket).size() == 0);
      continue;
    }
    REQUIRE(plane < arrangement.n_base_planes());
    const auto &frame = world.frame(plane);
    const bool line = frame.plane_n[0] == 0 && frame.plane_n[1] == 0 &&
                      frame.plane_n[2] == 0;
    REQUIRE((line || world.plane_edges(plane).size() == 0));
  }
}

/// The stranding scene: the sphere's equator ring is a ring of the mesh
/// because the stack count is even, and the cutting plane is tessellated so
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

} // namespace

// ============================================================================
// The stranding scene, and the wave that recovers it.
// ============================================================================
TEMPLATE_TEST_CASE("plane refusal: the stranded scene recovers",
                   "[cut][planes][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto scene = sphere_on_plane<index_t, real_t>(16, 32, real_t(0));
  auto op0 = tf::test::make_tagged_operand(scene.first);
  auto op1 = tf::test::make_tagged_operand(scene.second);
  auto graph = tf::test::build_pair_arrangement(op0.form(), op1.form(), {});

  REQUIRE(graph.failed().size() == 0);
  // the fixture is only worth its runtime while it still reaches the wave
  REQUIRE(graph.arrangement().census().refusals == wave_cost<real_t>::refusals);
  REQUIRE(graph.arrangement().census().rounds == wave_cost<real_t>::rounds);
  require_every_carrier_holds_its_product(graph);
}

// ============================================================================
// Its neighbourhood: the trigger is one point of a family, and no member of
// the family may strand.
// ============================================================================
TEMPLATE_TEST_CASE("plane refusal: the stranding ladder holds",
                   "[cut][planes][refusal]",
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

  for (const int stack : stacks) {
    auto scene = sphere_on_plane<index_t, real_t>(stack, 32, real_t(0));
    auto op0 = tf::test::make_tagged_operand(scene.first);
    auto op1 = tf::test::make_tagged_operand(scene.second);
    auto graph = tf::test::build_pair_arrangement(op0.form(), op1.form(), {});
    INFO("stacks " << stack);
    REQUIRE(graph.failed().size() == 0);
    require_every_carrier_holds_its_product(graph);
  }
  for (const int segment : segments) {
    auto scene = sphere_on_plane<index_t, real_t>(16, segment, real_t(0));
    auto op0 = tf::test::make_tagged_operand(scene.first);
    auto op1 = tf::test::make_tagged_operand(scene.second);
    auto graph = tf::test::build_pair_arrangement(op0.form(), op1.form(), {});
    INFO("segments " << segment);
    REQUIRE(graph.failed().size() == 0);
    require_every_carrier_holds_its_product(graph);
  }
  for (const real_t height : heights) {
    auto scene = sphere_on_plane<index_t, real_t>(16, 32, height);
    auto op0 = tf::test::make_tagged_operand(scene.first);
    auto op1 = tf::test::make_tagged_operand(scene.second);
    auto graph = tf::test::build_pair_arrangement(op0.form(), op1.form(), {});
    INFO("height " << double(height));
    REQUIRE(graph.failed().size() == 0);
    require_every_carrier_holds_its_product(graph);
  }
}

// ============================================================================
// The average path: two boxes that cut each other cleanly.
// ============================================================================
TEMPLATE_TEST_CASE("plane refusal: a scene that never refuses",
                   "[cut][planes][refusal]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto b0 = tf::make_box_mesh<index_t, real_t>(real_t(1), real_t(1), real_t(1));
  auto b1 = tf::make_box_mesh<index_t, real_t>(real_t(1), real_t(1), real_t(1));
  for (auto point : b1.points_buffer()) {
    point[0] = point[0] + real_t(0.5);
    point[1] = point[1] + real_t(0.25);
  }
  auto op0 = tf::test::make_tagged_operand(b0);
  auto op1 = tf::test::make_tagged_operand(b1);
  auto graph = tf::test::build_pair_arrangement(op0.form(), op1.form(), {});

  REQUIRE(graph.failed().size() == 0);
  REQUIRE(graph.arrangement().census().refusals == 0);
  REQUIRE(graph.arrangement().census().rounds == 1);
  require_every_carrier_holds_its_product(graph);
}

// ============================================================================
// The other side of the surface: a carrier the wave cannot recover.
// ============================================================================
TEMPLATE_TEST_CASE("plane refusal: a constraint-less carrier is published",
                   "[cut][planes][refusal]", tf::exact::int32,
                   tf::exact::int64) {
  using Int = TestType;
  using index_t = tf::test::plane_index_t;

  // the shared-boundary world with the second plane's edge block emptied: it
  // still carries a face, so it is a carrier, and it states no constraint, so
  // its triangulation refuses and the wave has nothing to state for it
  auto world = tf::test::make_plane_shared_boundary_world<Int>();
  auto &edges = world.input.tables().edges();
  const auto kept = edges.offsets_buffer()[1];
  edges.data_buffer().erase_till_end(edges.data_buffer().begin() +
                                     std::ptrdiff_t(kept));
  edges.offsets_buffer()[2] = kept;

  const auto get_original = [&](int, index_t point) {
    return world.points[std::size_t(point)];
  };
  tf::arrangement::plane_arrangement<index_t, Int> product;
  product.build(world.input, index_t(0), get_original, get_original,
                world.vertex_offsets);

  REQUIRE(product.failed().size() == 1);
  REQUIRE(product.failed()[0] == 1);
  REQUIRE(product.census().failed_planes == 1);
  // the failure is that plane's alone: the arrangement keeps every product it
  // did state
  REQUIRE(product.plane_range(0)[1] > product.plane_range(0)[0]);
  REQUIRE(product.plane_range(1)[1] == product.plane_range(1)[0]);
  REQUIRE(product.face_range(1)[1] == product.face_range(1)[0]);
}

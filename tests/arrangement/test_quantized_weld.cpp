/**
 * @file test_quantized_weld.cpp
 * @brief Two float-distinct vertices that quantize onto one lattice point
 *        leave the exposed stream, and the arrangement carrying them stays
 *        watertight.
 *
 * The fixture is an octahedron whose top apex is split into a needle —
 * two vertices a sub-lattice-step apart, joined by two sliver hinge
 * faces — cut by a small box through the apex region so the slivers'
 * regions carry both needle endpoints. The identity layer never sees
 * the pair (same-tag originals in a pair arrangement); the plane
 * arrangement does, and whatever it decides about them, no consumer of
 * the exposed stream may still be handed a retired original.
 *
 * The premise is asserted, not assumed: the test checks the pipeline's
 * own converter maps the two vertices to the same lattice point, so a
 * converter-scale change fails loudly here instead of silently testing
 * nothing.
 *
 * The weld's own steps are unit-tested against the canonicalize wave in
 * tests/cut/planes/test_plane_canonicalize_wave.cpp. This file is the
 * end-to-end oracle over the public graph.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <vector>

namespace {

using weld_index_t = int;
using weld_real_t = double;
using Mesh = tf::polygons_buffer<weld_index_t, weld_real_t, 3, 3>;

/// Octahedron with the top apex split into a needle of length `eps`
/// along x: fans re-anchored per side, two sliver hinges close it.
/// The needle sits AT x = 0, where doubles are dense enough to place
/// two distinct values inside one lattice cell (~2.4e-19 for this
/// bbox); away from zero the coordinate ulp exceeds the lattice step
/// and no distinct doubles can collide.
auto needle_octahedron(weld_real_t eps, bool with_spectator = false) -> Mesh {
  Mesh m;
  auto &pts = m.points_buffer();
  pts.emplace_back(weld_real_t(0), weld_real_t(0),
                   weld_real_t(1));                      // 0: apex t1
  pts.emplace_back(eps, weld_real_t(0), weld_real_t(1)); // 1: apex t2
  pts.emplace_back(weld_real_t(1), weld_real_t(0), weld_real_t(0));  // 2: a
  pts.emplace_back(weld_real_t(0), weld_real_t(1), weld_real_t(0));  // 3: b
  pts.emplace_back(weld_real_t(-1), weld_real_t(0), weld_real_t(0)); // 4: c
  pts.emplace_back(weld_real_t(0), weld_real_t(-1), weld_real_t(0)); // 5: d
  pts.emplace_back(weld_real_t(0), weld_real_t(0),
                   weld_real_t(-1)); // 6: bottom u
  if (with_spectator) {
    pts.emplace_back(weld_real_t(0.05), weld_real_t(0.02),
                     weld_real_t(0.93)); // 7
    pts.emplace_back(weld_real_t(0.02), weld_real_t(0.05),
                     weld_real_t(0.93)); // 8
  }
  auto &f = m.faces_buffer();
  if (with_spectator) {
    // Face 0 lies wholly inside the cutter and has no intersection record.
    // Its apex corner can enter the arrangement only through the twin.
    f.emplace_back(0, 7, 8);
    f.emplace_back(0, 2, 7);
    f.emplace_back(2, 3, 7);
    f.emplace_back(3, 8, 7);
    f.emplace_back(3, 0, 8);
  } else
    f.emplace_back(0, 2, 3); // t1 fan: a->b
  f.emplace_back(0, 3, 4);   // t1 fan: b->c
  f.emplace_back(1, 4, 5);   // t2 fan: c->d
  f.emplace_back(1, 5, 2);   // t2 fan: d->a
  f.emplace_back(0, 4, 1);   // hinge t1-c-t2
  f.emplace_back(1, 2, 0);   // hinge t2-a-t1
  f.emplace_back(6, 3, 2);   // bottom: b->a
  f.emplace_back(6, 4, 3);   // bottom: c->b
  f.emplace_back(6, 5, 4);   // bottom: d->c
  f.emplace_back(6, 2, 5);   // bottom: a->d
  return m;
}

/// A small box through the apex region: its side planes cut the fan
/// faces AND the sliver hinges (the hinge lines cross x = +-0.25 at
/// z = 0.75, interior to the box's [0.65, 1.15] z-span), so the cut
/// regions on the apex side carry both twins.
auto apex_cutter() -> Mesh {
  return tf::make_box_mesh<weld_index_t, weld_real_t>(
      weld_real_t(0.5), weld_real_t(0.5), weld_real_t(0.5));
}

auto frame_at(weld_real_t z) -> tf::frame<weld_real_t, 3> {
  return tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<weld_real_t, 3>{weld_real_t(0), weld_real_t(0), z}));
}

} // namespace

TEST_CASE("quantized weld: sub-lattice twins leave the exposed stream and the "
          "raw arrangement stays closed",
          "[cut][weld][global]") {
  const weld_real_t eps = weld_real_t(1e-19);
  auto needle = needle_octahedron(eps);
  REQUIRE(tf::is_closed(needle.polygons()));
  REQUIRE(tf::is_manifold(needle.polygons()));
  REQUIRE(double(tf::signed_volume(needle.polygons())) > 0.0);

  auto cutter = apex_cutter();
  auto identity = frame_at(weld_real_t(0));
  auto lift = frame_at(weld_real_t(0.9));
  std::vector<tf::test::tagged_operand<weld_index_t, weld_real_t>> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(
      needle, tf::transformation<weld_real_t, 3>(identity.transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      cutter, tf::transformation<weld_real_t, 3>(lift.transformation())));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  // Premise: the pipeline's own converter collapses the twins.
  const auto q0 = graph.converter().convert(needle.polygons().points()[0]);
  const auto q1 = graph.converter().convert(needle.polygons().points()[1]);
  REQUIRE(q0 == q1);
  REQUIRE(needle.polygons().points()[0] != needle.polygons().points()[1]);

  // Clean ownership boundary: every later consumer reads the exposed stream
  // and is allowed to trust its ids without a substitution table, so neither
  // retired original may still appear as a corner.
  const auto &arrangement = graph.arrangement();
  for (const auto &triangle : arrangement.global().exposed_tris())
    for (const auto &corner : triangle)
      REQUIRE_FALSE(
          (corner.source == tf::intersect::graph::vertex_source::original &&
           (corner.id == weld_index_t(0) || corner.id == weld_index_t(1))));

  // Every source face incident to either retired original belongs to the cut
  // portion, including faces that were geometrically untouched by the cutter.
  tf::face_membership<weld_index_t> membership;
  membership.build(needle.polygons());
  auto is_cut_face = [&](weld_index_t face) {
    for (const auto &descriptor : arrangement.global().exposed_descriptors())
      if (weld_index_t(descriptor.tag) == weld_index_t(0) &&
          descriptor.object == face)
        return descriptor.plane != weld_index_t(-1);
    return false;
  };
  for (const auto face : membership[weld_index_t(0)])
    REQUIRE(is_cut_face(weld_index_t(face)));
  for (const auto face : membership[weld_index_t(1)])
    REQUIRE(is_cut_face(weld_index_t(face)));

  // The oracle: the RAW arrangement is watertight — no cleaned(), closure is
  // topological, so a retired original still referenced by an uncut face
  // would surface as boundary edges here.
  auto raw = tf::test::csg_mesh_of(graph);
  REQUIRE(tf::is_closed(raw.polygons()));
  REQUIRE(tf::make_boundary_edges(raw.polygons()).size() == 0);

  // And the boolean read over the same graph stays a solid.
  auto diff = tf::test::csg_mesh_of(graph, tf::csg::difference(0, 1));
  REQUIRE(tf::is_closed(diff.polygons()));
  REQUIRE(tf::is_manifold(diff.polygons()));
}

TEST_CASE("a face wholly inside the cutter enters through the twin it carries",
          "[cut][weld][global][wave]") {
  const weld_real_t eps = weld_real_t(1e-19);
  auto needle = needle_octahedron(eps, true);
  REQUIRE(tf::is_closed(needle.polygons()));
  REQUIRE(tf::is_manifold(needle.polygons()));

  auto cutter = apex_cutter();
  auto identity = frame_at(weld_real_t(0));
  auto lift = frame_at(weld_real_t(0.9));
  std::vector<tf::test::tagged_operand<weld_index_t, weld_real_t>> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(
      needle, tf::transformation<weld_real_t, 3>(identity.transformation())));
  operands.push_back(tf::test::make_tagged_operand(
      cutter, tf::transformation<weld_real_t, 3>(lift.transformation())));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  const auto &arrangement = graph.arrangement();

  REQUIRE(arrangement.converter().convert(needle.polygons().points()[0]) ==
          arrangement.converter().convert(needle.polygons().points()[1]));

  // Face 0 carries no intersection record of its own; it is cut because the
  // twin its apex corner sits on was resolved.
  bool spectator_is_cut = false;
  for (const auto &descriptor : arrangement.global().exposed_descriptors())
    if (weld_index_t(descriptor.tag) == weld_index_t(0) &&
        descriptor.object == weld_index_t(0))
      spectator_is_cut = descriptor.plane != weld_index_t(-1);
  REQUIRE(spectator_is_cut);

  for (const auto &triangle : arrangement.global().exposed_tris())
    for (const auto &corner : triangle)
      REQUIRE_FALSE(
          (corner.source == tf::intersect::graph::vertex_source::original &&
           (corner.id == weld_index_t(0) || corner.id == weld_index_t(1))));

  auto raw = tf::test::csg_mesh_of(graph);
  REQUIRE(tf::is_closed(raw.polygons()));
  REQUIRE(tf::make_boundary_edges(raw.polygons()).size() == 0);
}

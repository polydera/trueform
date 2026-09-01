/**
 * @file test_intersection_primitives.cpp
 * @brief tf::polygon_intersections — the five contact primitives.
 *
 * One minimal fixture per contact type, each built so the pair states
 * that type and nothing else. Every case asserts the classification —
 * the two FEATURES the point was delivered at, which is what states the
 * type for a contact whose pair holds one point and therefore no pair
 * record — together with what that classification obliges in the
 * identity surface: a vertex contact names a kind-V point, an edge
 * contact lists its point on the canonical carrier, a face contact names
 * no carrier at all.
 *
 * Fixtures use integer coordinates so the converter is the identity and
 * every incidence is exact on the lattice, not merely close.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/edge_parameter.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/records/tagged_intersection.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/topo_type.hpp>

#include "input_lattice_for.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace {

using primitives_index_t = int;
using primitives_mesh_t = tf::polygons_buffer<primitives_index_t, int, 3, 3>;
using primitives_int_t = tf::exact::int32;
using primitives_wide_t = tf::exact::meta<primitives_int_t>::T2;
using primitives_ibp_t =
    tf::polygon_intersections<primitives_index_t, int, primitives_int_t>;

auto make_primitives_mesh(
    const std::vector<std::array<int, 3>> &pts,
    const std::vector<std::array<primitives_index_t, 3>> &faces)
    -> primitives_mesh_t {
  primitives_mesh_t mesh;
  mesh.points_buffer().allocate(pts.size());
  mesh.faces_buffer().allocate(faces.size());
  for (std::size_t i = 0; i < pts.size(); ++i)
    for (int d = 0; d < 3; ++d)
      mesh.points()[i][d] = pts[i][d];
  for (std::size_t i = 0; i < faces.size(); ++i)
    for (int d = 0; d < 3; ++d)
      mesh.faces()[i][d] = faces[i][d];
  return mesh;
}

struct primitives_fixture {
  primitives_mesh_t mesh;
  tf::aabb_tree<primitives_index_t, int, 3> tree;
  decltype(tf::make_face_membership(
      std::declval<primitives_mesh_t &>().polygons())) fm;
  decltype(tf::make_manifold_edge_link(
      std::declval<primitives_mesh_t &>().polygons())) mel;

  explicit primitives_fixture(primitives_mesh_t m)
      : mesh(std::move(m)), tree(mesh.polygons(), tf::config_tree(4, 4)),
        fm(tf::make_face_membership(mesh.polygons())),
        mel(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() {
    return mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);
  }
};

auto primitives_ratio(int num, int den)
    -> tf::exact::edge_parameter<primitives_int_t> {
  return {primitives_wide_t(num), primitives_wide_t(den)};
}

/// One face's claim on a point, from whichever currency states it: a pair
/// record names the point at its own face as well as the pair it stands
/// on, a delivery names only the point. The loop reads both, so a fixture
/// about identity reads both.
struct primitives_claim {
  int tag;
  primitives_index_t object;
  tf::topo_id<primitives_index_t> target;
  primitives_index_t id;
};

auto primitives_claims(const primitives_ibp_t &ibp)
    -> std::vector<primitives_claim> {
  std::vector<primitives_claim> out;
  for (const auto &r : ibp.flat_intersections())
    out.push_back({int(r.tag), r.object, r.target, r.id});
  for (const auto &d : ibp.flat_deliveries())
    out.push_back({int(d.tag), d.object, d.target, d.id});
  return out;
}

/// The contact a point states, read off the two FEATURES it was
/// delivered at. The pair is unordered, and the delivery names one
/// feature per face the contact reaches, so the distinct labels a point
/// was delivered at ARE its classification — stated for every contact,
/// including the ones whose pair holds a single point and therefore no
/// pair record.
enum class contact { vv, ve, vf, ee, ef, other };

auto contact_of(tf::topo_type a, tf::topo_type b) -> contact {
  if (a > b)
    std::swap(a, b);
  if (a == tf::topo_type::vertex && b == tf::topo_type::vertex)
    return contact::vv;
  if (a == tf::topo_type::vertex && b == tf::topo_type::edge)
    return contact::ve;
  if (a == tf::topo_type::vertex && b == tf::topo_type::face)
    return contact::vf;
  if (a == tf::topo_type::edge && b == tf::topo_type::edge)
    return contact::ee;
  if (a == tf::topo_type::edge && b == tf::topo_type::face)
    return contact::ef;
  return contact::other;
}

/// Points per contact type, indexed by `contact`.
auto census(const primitives_ibp_t &ibp) -> std::array<int, 6> {
  std::map<primitives_index_t, std::set<tf::topo_type>> labels;
  for (const auto &c : primitives_claims(ibp))
    labels[c.id].insert(c.target.label);
  std::array<int, 6> counts{};
  for (const auto &entry : labels) {
    const auto &set = entry.second;
    REQUIRE(set.size() <= 2);
    const auto a = *set.begin();
    const auto b = *set.rbegin();
    ++counts[std::size_t(contact_of(a, b))];
  }
  return counts;
}

/// The canonical points listed on carrier (u, v), which must be a
/// carrier the build published exactly once.
auto primitives_carrier_block(const primitives_ibp_t &ibp, primitives_index_t u,
                              primitives_index_t v)
    -> std::vector<primitives_index_t> {
  auto carriers = ibp.edge_carriers();
  std::vector<primitives_index_t> found;
  int hits = 0;
  for (std::size_t g = 0; g < carriers.size(); ++g) {
    if (carriers[g].u != u || carriers[g].v != v)
      continue;
    ++hits;
    found.clear();
    for (auto id : ibp.edge_splits()[g])
      found.push_back(id);
  }
  REQUIRE(hits == 1);
  return found;
}

/// A pierce of A's interior by one edge of B, and its mirror: A's plane
/// cuts B's triangle along a chord whose two crossing edges land one
/// inside A and one outside it, and A's own crossings land outside B.
auto pierced_face() -> primitives_mesh_t {
  return make_primitives_mesh({{{0, 0, 0}}, {{12, 0, 0}}, {{0, 12, 0}}},
                              {{{0, 1, 2}}});
}
auto piercing_blade() -> primitives_mesh_t {
  return make_primitives_mesh({{{2, 2, -6}}, {{2, 2, 6}}, {{6, 2, 3}}},
                              {{{0, 1, 2}}});
}

/// One transversal edge crossing another transversal edge at (4, 4, 0).
auto crossed_edge() -> primitives_mesh_t {
  return make_primitives_mesh({{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}},
                              {{{0, 1, 2}}});
}
auto crossing_edge() -> primitives_mesh_t {
  return make_primitives_mesh({{{4, 4, -4}}, {{4, 4, 4}}, {{20, 20, 5}}},
                              {{{0, 1, 2}}});
}

} // namespace

TEST_CASE("intersection primitives: an edge through a face interior is EF",
          "[intersect][primitives]") {
  primitives_fixture a(pierced_face());
  primitives_fixture b(piercing_blade());

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  // Both of B's crossing edges pierce A; A's own crossings miss B.
  auto counts = census(ibp);
  REQUIRE(counts[std::size_t(contact::ef)] == 2);
  REQUIRE(counts == std::array<int, 6>{0, 0, 0, 0, 2, 0});

  // A pierce is a kind-E point, and its carrier is the piercing EDGE —
  // the face side of the contact carries no identity of its own.
  REQUIRE(ibp.n_vertex_points() == 0);
  REQUIRE(ibp.n_points() == 2);
  REQUIRE(ibp.edge_carriers().size() == 2);

  auto blade = primitives_carrier_block(ibp, 3, 4);
  REQUIRE(blade.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(blade[0]),
                                       primitives_ratio(1, 2)) == 0);
  auto flank = primitives_carrier_block(ibp, 3, 5);
  REQUIRE(flank.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(flank[0]),
                                       primitives_ratio(2, 3)) == 0);
  REQUIRE(blade[0] != flank[0]);

  for (const auto &c : primitives_claims(ibp))
    REQUIRE(c.id >= ibp.n_vertex_points());
}

TEST_CASE("intersection primitives: two edges meeting in space are EE",
          "[intersect][primitives]") {
  primitives_fixture a(crossed_edge());
  primitives_fixture b(crossing_edge());

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{0, 0, 0, 1, 0, 0});

  // One point on TWO carriers: an edge-edge contact splits both edges,
  // and the classes the two constructions produce collapse into one.
  REQUIRE(ibp.n_vertex_points() == 0);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.edge_carriers().size() == 2);

  auto crossed = primitives_carrier_block(ibp, 1, 2);
  auto crossing = primitives_carrier_block(ibp, 3, 4);
  REQUIRE(crossed.size() == 1);
  REQUIRE(crossing.size() == 1);
  REQUIRE(crossed[0] == crossing[0]);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(crossed[0]),
                                       primitives_ratio(1, 2)) == 0);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(crossing[0]),
                                       primitives_ratio(1, 2)) == 0);
}

TEST_CASE("intersection primitives: an edge through a vertex is VE",
          "[intersect][primitives]") {
  // B's first edge runs (0,0,-4) -> (0,0,4) straight through A's vertex
  // 0; B's other crossing lands outside A's wedge.
  primitives_fixture a(make_primitives_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  primitives_fixture b(make_primitives_mesh(
      {{{0, 0, -4}}, {{0, 0, 4}}, {{-6, -3, 1}}}, {{{0, 1, 2}}}));

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{0, 1, 0, 0, 0, 0});

  // The vertex wins the identity — a pierce landing on an original
  // vertex IS that vertex — and it still subdivides the edge it lies on.
  REQUIRE(ibp.n_vertex_points() == 1);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.vertex_anchor(0).tag == 0);
  REQUIRE(ibp.vertex_anchor(0).vid == 0);

  REQUIRE(ibp.edge_carriers().size() == 1);
  auto pierced = primitives_carrier_block(ibp, 3, 4);
  REQUIRE(pierced.size() == 1);
  REQUIRE(pierced[0] == 0);

  for (const auto &c : primitives_claims(ibp))
    REQUIRE(c.id == 0);
}

TEST_CASE("intersection primitives: a vertex inside a face is VF",
          "[intersect][primitives]") {
  primitives_fixture a(make_primitives_mesh(
      {{{-8, -8, 0}}, {{8, -8, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  primitives_fixture b(make_primitives_mesh(
      {{{0, 0, 0}}, {{0, -6, 10}}, {{6, 4, 10}}}, {{{0, 1, 2}}}));

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{0, 0, 1, 0, 0, 0});

  // A vertex resting on a face interior names its own vertex point and
  // subdivides nothing: no edge of either face carries it.
  REQUIRE(ibp.n_vertex_points() == 1);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.vertex_anchor(0).tag == 1);
  REQUIRE(ibp.vertex_anchor(0).vid == 0);
  REQUIRE(ibp.edge_carriers().size() == 0);

  for (const auto &c : primitives_claims(ibp))
    REQUIRE(c.id == 0);
}

TEST_CASE("intersection primitives: two coincident vertices are VV",
          "[intersect][primitives]") {
  primitives_fixture a(make_primitives_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  primitives_fixture b(make_primitives_mesh(
      {{{0, 0, 0}}, {{0, -8, 8}}, {{-8, 0, 8}}}, {{{0, 1, 2}}}));

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{1, 0, 0, 0, 0, 0});

  // Coincident originals are identified before carriers are keyed, so
  // the two vertex ids are one class anchored at its lowest member.
  REQUIRE(ibp.n_vertex_points() == 1);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.canonical_vertex(0, 0) == 0);
  REQUIRE(ibp.canonical_vertex(1, 0) == 0);
  REQUIRE(ibp.vertex_anchor(0).tag == 0);
  REQUIRE(ibp.vertex_anchor(0).vid == 0);
  REQUIRE(ibp.edge_carriers().size() == 0);

  for (const auto &c : primitives_claims(ibp))
    REQUIRE(c.id == 0);
}

TEST_CASE("intersection primitives: a coplanar overlap is stamped EE",
          "[intersect][primitives]") {
  // Two triangles in z = 0 overlapping as a hexagram: six coplanar edge
  // crossings, no vertex of either inside or on the other.
  primitives_fixture a(make_primitives_mesh(
      {{{0, 0, 0}}, {{12, 0, 0}}, {{6, 10, 0}}}, {{{0, 1, 2}}}));
  primitives_fixture b(make_primitives_mesh(
      {{{0, 6, 0}}, {{12, 6, 0}}, {{6, -4, 0}}}, {{{0, 1, 2}}}));

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{0, 0, 0, 6, 0, 0});

  // The coplanarity is the PAIR's fact, so every record of the pair
  // carries it.
  for (const auto &rec : ibp.flat_intersections())
    REQUIRE((rec.flags & tf::intersect::coplanar_pair_flag) != 0);

  REQUIRE(ibp.n_vertex_points() == 0);
  REQUIRE(ibp.n_points() == 6);
  REQUIRE(ibp.edge_carriers().size() == 6);
  for (std::size_t g = 0; g < ibp.edge_carriers().size(); ++g)
    REQUIRE(ibp.edge_splits()[g].size() == 2);
}

TEST_CASE("intersection primitives: a form states its own crossings",
          "[intersect][primitives][self]") {
  // Two faces of ONE form, sharing no vertex, cutting each other.
  primitives_fixture a(make_primitives_mesh({{{0, 0, 0}},
                                             {{10, 0, 0}},
                                             {{0, 10, 0}},
                                             {{2, 2, -5}},
                                             {{2, 2, 5}},
                                             {{20, 2, 3}}},
                                            {{{0, 1, 2}}, {{3, 4, 5}}}));

  primitives_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), 0.0);
  ibp.build(a.form(), ibp_lattice, tf::intersect_mode::primitives);

  auto counts = census(ibp);
  REQUIRE(counts == std::array<int, 6>{0, 0, 0, 0, 2, 0});
  for (const auto &c : primitives_claims(ibp))
    REQUIRE(c.tag == 0);

  REQUIRE(ibp.n_vertex_points() == 0);
  REQUIRE(ibp.n_points() == 2);
  REQUIRE(ibp.edge_carriers().size() == 2);

  auto blade = primitives_carrier_block(ibp, 3, 4);
  REQUIRE(blade.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(blade[0]),
                                       primitives_ratio(1, 2)) == 0);
  auto pierced = primitives_carrier_block(ibp, 1, 2);
  REQUIRE(pierced.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(pierced[0]),
                                       primitives_ratio(1, 5)) == 0);
}

TEST_CASE("intersection primitives: a shared feature is not a contact",
          "[intersect][primitives][self]") {
  // The same two triangles and the same coordinates twice: as two forms
  // the touching corner is a VV contact, as one form it is a shared
  // vertex id and states nothing at all.
  primitives_fixture split_a(make_primitives_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  primitives_fixture split_b(make_primitives_mesh(
      {{{0, 0, 0}}, {{0, -8, 8}}, {{-8, 0, 8}}}, {{{0, 1, 2}}}));

  primitives_ibp_t apart;
  const auto apart_lattice = tf::test::input_lattice_for(split_a.form(), split_b.form(), 0.0);
  apart.build(split_a.form(), split_b.form(), apart_lattice, tf::intersect_mode::primitives);
  REQUIRE(census(apart) == std::array<int, 6>{1, 0, 0, 0, 0, 0});

  primitives_fixture joined(make_primitives_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}, {{0, -8, 8}}, {{-8, 0, 8}}},
      {{{0, 1, 2}}, {{0, 3, 4}}}));

  primitives_ibp_t together;
  const auto together_lattice = tf::test::input_lattice_for(joined.form(), 0.0);
  together.build(joined.form(), together_lattice, tf::intersect_mode::primitives);
  REQUIRE(primitives_claims(together).size() == 0);
  REQUIRE(together.n_points() == 0);
}

TEST_CASE("intersection primitives: sos states every contact as edge-face",
          "[intersect][primitives][sos]") {
  // The SoS arm has one primitive: the fan pierce. A transversal pierce
  // survives it unchanged, and an exact edge-edge meeting — which the
  // conforming arm calls EE — reads as edge-face here too.
  primitives_fixture pierced(pierced_face());
  primitives_fixture blade(piercing_blade());

  primitives_ibp_t pierce;
  const auto pierce_lattice = tf::test::input_lattice_for(pierced.form(), blade.form(), 0.0);
  pierce.build(pierced.form(), blade.form(), pierce_lattice, tf::intersect_mode::sos);
  auto pierce_counts = census(pierce);
  REQUIRE(pierce_counts[std::size_t(contact::ef)] == 2);
  REQUIRE(pierce_counts == std::array<int, 6>{0, 0, 0, 0, 2, 0});

  primitives_fixture crossed(crossed_edge());
  primitives_fixture crossing(crossing_edge());

  primitives_ibp_t meeting;
  const auto meeting_lattice = tf::test::input_lattice_for(crossed.form(), crossing.form(), 0.0);
  meeting.build(crossed.form(), crossing.form(), meeting_lattice, tf::intersect_mode::sos);
  auto meeting_counts = census(meeting);
  REQUIRE(meeting_counts[std::size_t(contact::ef)] != 0);
  for (const auto &rec : meeting.flat_intersections())
    REQUIRE(contact_of(rec.target.label, rec.target_other.label) ==
            contact::ef);
}

/**
 * @file test_tolerance_contact_holes.cpp
 * @brief Which contact pathways a band closes, and which it does not.
 *
 * A band is spent on the INPUT: each vertex moves at most the band onto a
 * lattice point of the planes ITS OWN faces state, and the classification
 * that follows is exact on the mesh that produced. So a gap closes when
 * the two sides come to stand on the same names, and only then.
 *
 * Each pathway is pinned from both sides: the exact fixture proves the
 * scene states the contact at all, and the gapped fixture — the same scene
 * with the contact opened by one lattice unit and a band that spans it —
 * says whether the band recovers it. The four that recover it do so
 * because the two sides name one plane; the two that do not are the
 * contract's price, and they say so where they stand.
 *
 * Coordinates are integers, so the converter is the identity and the
 * exact cases are incident on the lattice rather than merely close.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/intersect/intersect_config.hpp>
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

using holes_index_t = int;
using holes_mesh_t = tf::polygons_buffer<holes_index_t, int, 3, 3>;
using holes_int_t = tf::exact::int32;
using holes_ibp_t = tf::polygon_intersections<holes_index_t, int, holes_int_t>;

/// The gap a "sub-tolerance" fixture opens, and a band that spans it with
/// room to spare — six lattice units against a gap of one, on a mesh whose
/// coordinates are already the lattice's.
constexpr int gap = 1;
constexpr double spanning_tolerance = 6.0;

auto make_holes_mesh(const std::vector<std::array<int, 3>> &pts,
                     const std::vector<std::array<holes_index_t, 3>> &faces)
    -> holes_mesh_t {
  holes_mesh_t mesh;
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

struct holes_fixture {
  holes_mesh_t mesh;
  tf::aabb_tree<holes_index_t, int, 3> tree;
  decltype(tf::make_face_membership(
      std::declval<holes_mesh_t &>().polygons())) fm;
  decltype(tf::make_manifold_edge_link(
      std::declval<holes_mesh_t &>().polygons())) mel;

  explicit holes_fixture(holes_mesh_t m)
      : mesh(std::move(m)), tree(mesh.polygons(), tf::config_tree(4, 4)),
        fm(tf::make_face_membership(mesh.polygons())),
        mel(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() {
    return mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);
  }
};

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

/// Points per contact type. A point delivered at more than two distinct
/// feature labels is not a primitive contact and is counted as `other`.
auto census(const holes_ibp_t &ibp) -> std::array<int, 6> {
  std::map<holes_index_t, std::set<tf::topo_type>> labels;
  for (const auto &r : ibp.flat_intersections())
    labels[r.id].insert(r.target.label);
  for (const auto &d : ibp.flat_deliveries())
    labels[d.id].insert(d.target.label);
  std::array<int, 6> counts{};
  for (const auto &entry : labels) {
    const auto &set = entry.second;
    if (set.size() > 2) {
      ++counts[std::size_t(contact::other)];
      continue;
    }
    const auto a = *set.begin();
    const auto b = *set.rbegin();
    ++counts[std::size_t(contact_of(a, b))];
  }
  return counts;
}

/// The census of one scene at one tolerance.
auto contacts(holes_mesh_t left, holes_mesh_t right, double tolerance)
    -> std::array<int, 6> {
  holes_fixture a(std::move(left));
  holes_fixture b(std::move(right));
  holes_ibp_t ibp;
  const auto lattice =
      tf::test::input_lattice_for(a.form(), b.form(), tolerance);
  ibp.build(a.form(), b.form(), lattice,
            tf::intersect_config{tf::intersect_mode::primitives, tolerance});
  return census(ibp);
}

auto n_of(const std::array<int, 6> &c, contact k) -> int {
  return c[std::size_t(k)];
}


// ---- the six scenes, each parameterised by the gap it opens ----

/// A wedge in the plane z = 0; every scene below contacts it. The
/// features are two orders of magnitude larger than the band, so a
/// contact keeps its own type instead of collapsing onto a nearer one.
auto host() -> holes_mesh_t {
  return make_holes_mesh({{{0, 0, 0}}, {{1200, 0, 0}}, {{0, 1200, 0}}},
                         {{{0, 1, 2}}});
}

/// B's corner meets the host's corner at the origin.
auto corner_meeting(int rise) -> holes_mesh_t {
  return make_holes_mesh({{{0, 0, rise}}, {{0, -1200, 600}}, {{-1200, 0, 600}}},
                         {{{0, 1, 2}}});
}

/// B's corner meets the interior of the host's edge (0,0)-(1200,0). B
/// lies in the plane y = 0, which CONTAINS that edge, so the contact is
/// the corner against the edge and nothing else.
auto corner_on_edge(int rise) -> holes_mesh_t {
  return make_holes_mesh({{{600, 0, rise}}, {{300, 0, 1000}}, {{900, 0, 1000}}},
                         {{{0, 1, 2}}});
}

/// B's edge lies ALONG the host's edge, overlapping its interior.
auto edge_along_edge(int rise) -> holes_mesh_t {
  return make_holes_mesh(
      {{{300, 0, rise}}, {{900, 0, rise}}, {{600, -800, rise}}}, {{{0, 1, 2}}});
}

/// B's corner meets the host's face interior.
auto corner_in_face(int rise) -> holes_mesh_t {
  return make_holes_mesh(
      {{{300, 300, rise}}, {{300, 300, 900}}, {{700, 600, 900}}},
      {{{0, 1, 2}}});
}

/// B repeats the host's own triangle.
auto coincident_wall(int rise) -> holes_mesh_t {
  return make_holes_mesh({{{0, 0, rise}}, {{1200, 0, rise}}, {{0, 1200, rise}}},
                         {{{0, 1, 2}}});
}

/// B's edge crosses the host's plane, landing well inside the face.
auto grazing_crossing(int rise) -> holes_mesh_t {
  return make_holes_mesh({{{500, 200 + rise, -600}},
                          {{500, 200 + rise, 600}},
                          {{900, 600 + rise, 600}}},
                         {{{0, 1, 2}}});
}

/// THE PATHWAY IS OPEN when the gapped scene states the SAME census as
/// the exact one: the same contacts, of the same types, in the same
/// numbers. A pathway that answers one and not the other is a HOLE — the
/// contact exists at the stated tolerance and no record names it.
///
/// WHAT CLOSES A GAP IS AGREEMENT OF NAMES. A tolerance moves each vertex
/// at most the band onto a lattice point of the planes ITS OWN faces
/// state; two features become one when the names they stand on agree, and
/// the classification below is exact on what the move produced. A vertex
/// is never drawn onto ANOTHER form's feature just for being near it —
/// that is the contract's price, and the scenes below state which
/// pathways it keeps and which it does not.
auto same_census(holes_mesh_t exact_left, holes_mesh_t exact_right,
                 holes_mesh_t gap_left, holes_mesh_t gap_right, contact kind,
                 int at_least) -> bool {
  const auto e = contacts(std::move(exact_left), std::move(exact_right), 0.0);
  const auto g = contacts(std::move(gap_left), std::move(gap_right),
                          spanning_tolerance);
  return n_of(e, kind) >= at_least && e == g;
}

} // namespace

TEST_CASE("contact holes: a corner meeting a corner", "[intersect][tolerance]") {
  CHECK(same_census(host(), corner_meeting(0), host(), corner_meeting(gap),
                    contact::vv, 1));
}

TEST_CASE("contact holes: a corner meeting an edge", "[intersect][tolerance]") {
  // The exact scene states the incidence.
  CHECK(n_of(contacts(host(), corner_on_edge(0), 0.0), contact::ve) >= 1);
  // The gapped one does not, and cannot: B's corner lies in B's own plane
  // `y = 0` and already stands on it, so the placement leaves it where it
  // is — the host's EDGE is not a plane B names.
  CHECK(contacts(host(), corner_on_edge(gap), spanning_tolerance) ==
        contacts(host(), corner_on_edge(gap), 0.0));
}

TEST_CASE("contact holes: an edge lying along an edge",
          "[intersect][tolerance]") {
  // Two collinear edges overlapping over a span meet at the span's two
  // ends, and each end is a corner ON the other edge — so this contact is
  // stated as the two vertex-edge incidences bounding the overlap.
  CHECK(same_census(host(), edge_along_edge(0), host(), edge_along_edge(gap),
                    contact::ve, 2));
}

TEST_CASE("contact holes: a corner meeting a face", "[intersect][tolerance]") {
  // The exact scene states the incidence.
  CHECK(n_of(contacts(host(), corner_in_face(0), 0.0), contact::vf) >= 1);
  // The gapped one does not: B's corner meets the planes of B's own faces,
  // none of which is the host's `z = 0`, so it is placed on those and stays
  // off the host's face.
  CHECK(contacts(host(), corner_in_face(gap), spanning_tolerance) ==
        contacts(host(), corner_in_face(gap), 0.0));
}

TEST_CASE("contact holes: a wall coincident with a wall",
          "[intersect][tolerance]") {
  CHECK(same_census(host(), coincident_wall(0), host(), coincident_wall(gap),
                    contact::vv, 3));
}

TEST_CASE("contact holes: a crossing through the face interior",
          "[intersect][tolerance]") {
  CHECK(same_census(host(), grazing_crossing(0), host(), grazing_crossing(gap),
                    contact::ef, 2));
}

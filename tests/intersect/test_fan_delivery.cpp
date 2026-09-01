/**
 * @file test_fan_delivery.cpp
 * @brief What the duplication fan must deliver, stated per requirement.
 *
 * The fan states two different facts and they have two carriers: the POINT
 * IDENTITY a face must substitute at a feature of its own — a per-face fact,
 * costing the SUM of the two feature fans — and the PAIR INTERACTION a
 * chord, a coplanar row and the edge-def builders read per (face, cut)
 * group, which costs only the pairs those consumers still read. These
 * fixtures hold each fact against its own carrier, so a change to how the
 * fan is carried is answerable by them rather than by an end-to-end face
 * count.
 *
 * Integer coordinates throughout, so the converter is the identity and every
 * incidence is exact on the lattice.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/topo_type.hpp>
#include <trueform/trueform.hpp>

#include "input_lattice_for.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using fan_delivery_index_t = int;
using fan_delivery_mesh_t =
    tf::polygons_buffer<fan_delivery_index_t, int, 3, 3>;
using fan_delivery_int_t = tf::exact::int32;
using fan_delivery_ibp_t =
    tf::polygon_intersections<fan_delivery_index_t, int, fan_delivery_int_t>;

auto make_fan_delivery_mesh(
    const std::vector<std::array<int, 3>> &pts,
    const std::vector<std::array<fan_delivery_index_t, 3>> &faces)
    -> fan_delivery_mesh_t {
  fan_delivery_mesh_t mesh;
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

struct fan_delivery_fixture {
  fan_delivery_mesh_t mesh;
  tf::aabb_tree<fan_delivery_index_t, int, 3> tree;
  decltype(tf::make_face_membership(
      std::declval<fan_delivery_mesh_t &>().polygons())) fm;
  decltype(tf::make_manifold_edge_link(
      std::declval<fan_delivery_mesh_t &>().polygons())) mel;

  explicit fan_delivery_fixture(fan_delivery_mesh_t m)
      : mesh(std::move(m)), tree(mesh.polygons(), tf::config_tree(4, 4)),
        fm(tf::make_face_membership(mesh.polygons())),
        mel(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() {
    return mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);
  }
};

/// One face's claim on a point, from whichever currency states it: a pair
/// record names the point at its own face as well as the pair it stands
/// on, a delivery names only the point. The loop reads both, so a fixture
/// about identity reads both.
struct claim {
  int tag;
  fan_delivery_index_t object;
  tf::topo_id<fan_delivery_index_t> target;
  fan_delivery_index_t id;
};

auto claims(const fan_delivery_ibp_t &ibp) -> std::vector<claim> {
  std::vector<claim> out;
  for (const auto &r : ibp.flat_intersections())
    out.push_back({int(r.tag), r.object, r.target, r.id});
  for (const auto &d : ibp.flat_deliveries())
    out.push_back({int(d.tag), d.object, d.target, d.id});
  return out;
}

/// REQUIREMENT: BOTH ORIENTATIONS. Every consumer reads a pair from its own
/// side, so a record and its swapped twin are one fact stated twice. Whatever
/// carries the fan must keep both.
auto check_both_orientations(const fan_delivery_ibp_t &ibp) -> void {
  using key_t = std::tuple<int, fan_delivery_index_t, int, fan_delivery_index_t,
                           int, fan_delivery_index_t, int, fan_delivery_index_t,
                           fan_delivery_index_t>;
  std::set<key_t> present;
  for (const auto &r : ibp.flat_intersections())
    present.insert({int(r.tag), r.object, int(r.tag_other), r.object_other,
                    int(r.target.label), r.target.id,
                    int(r.target_other.label), r.target_other.id, r.id});
  for (const auto &r : ibp.flat_intersections()) {
    const key_t mirrored{int(r.tag_other),      r.object_other,
                         int(r.tag),            r.object,
                         int(r.target_other.label), r.target_other.id,
                         int(r.target.label),   r.target.id,
                         r.id};
    INFO("record tag=" << int(r.tag) << " object=" << r.object
                       << " other=" << int(r.tag_other) << ":"
                       << r.object_other << " id=" << r.id);
    REQUIRE(present.count(mirrored) == 1);
  }
}

/// REQUIREMENT: IDENTITY SYNCHRONIZATION. A point sitting at an original
/// vertex is ONE canonical name, so every face that names that vertex as its
/// own target must state the same id — that is what makes a cut face and its
/// uncut neighbour agree on the corner. The identity currency carries this,
/// and it carries it for every incident face, including those whose pairs
/// hold no chord.
auto check_vertex_identity_synchronized(
    const fan_delivery_ibp_t &ibp,
    const std::vector<fan_delivery_mesh_t *> &meshes) -> void {
  std::map<std::pair<int, fan_delivery_index_t>, fan_delivery_index_t> named;
  for (const auto &c : claims(ibp)) {
    if (c.target.label != tf::topo_type::vertex)
      continue;
    const auto face = meshes[std::size_t(c.tag)]->polygons().faces()[c.object];
    const auto vertex = fan_delivery_index_t(face[std::size_t(c.target.id)]);
    const auto key = std::make_pair(c.tag, vertex);
    auto it = named.find(key);
    INFO("tag=" << c.tag << " vertex=" << vertex << " face=" << c.object);
    if (it == named.end())
      named.emplace(key, c.id);
    else
      REQUIRE(it->second == c.id);
  }
  REQUIRE(!named.empty());
}

/// The faces that received a point, per tag — THE FACE CARRIER the consumers
/// walk. `find_loop_index` searches its descriptors for a face named as the
/// OTHER side of a chord, so a face a surviving chord points at must be in it.
auto delivered_faces(const fan_delivery_ibp_t &ibp, int tag)
    -> std::set<fan_delivery_index_t> {
  std::set<fan_delivery_index_t> out;
  for (const auto &c : claims(ibp))
    if (c.tag == tag)
      out.insert(c.object);
  return out;
}

/// REQUIREMENT: EVERY CHORD'S PARTNER IS REACHABLE. A group of two or more
/// distinct points forms a chord whose emission reads the partner face's own
/// loop, so that partner must itself be on the carrier.
auto check_chord_partners_delivered(const fan_delivery_ibp_t &ibp) -> void {
  std::map<std::tuple<int, fan_delivery_index_t, int, fan_delivery_index_t>,
           std::set<fan_delivery_index_t>>
      groups;
  for (const auto &r : ibp.flat_intersections())
    groups[{int(r.tag), r.object, int(r.tag_other), r.object_other}].insert(
        r.id);
  std::map<int, std::set<fan_delivery_index_t>> delivered;
  for (const auto &c : claims(ibp))
    delivered[c.tag].insert(c.object);
  for (const auto &entry : groups) {
    if (entry.second.size() < 2)
      continue;
    const auto other_tag = std::get<2>(entry.first);
    const auto other_object = std::get<3>(entry.first);
    INFO("chord group " << std::get<0>(entry.first) << ":"
                        << std::get<1>(entry.first) << " -> " << other_tag
                        << ":" << other_object);
    REQUIRE(delivered[other_tag].count(other_object) == 1);
  }
}

/// REQUIREMENT: THE PAIR CURRENCY COSTS ITS READERS. A pair group is read
/// for the chords its points bound and, when the pair is coplanar, for the
/// row that pools the two faces into one plane. A group holding a single
/// point bounds no chord and, uncoplanar, is read for nothing — so it must
/// not be there at all.
auto check_no_inert_pairs(const fan_delivery_ibp_t &ibp) -> void {
  std::map<std::tuple<int, fan_delivery_index_t, int, fan_delivery_index_t>,
           std::set<fan_delivery_index_t>>
      points;
  std::map<std::tuple<int, fan_delivery_index_t, int, fan_delivery_index_t>,
           bool>
      coplanar;
  for (const auto &r : ibp.flat_intersections()) {
    const auto key =
        std::make_tuple(int(r.tag), r.object, int(r.tag_other), r.object_other);
    points[key].insert(r.id);
    coplanar[key] = coplanar[key] ||
                    (r.flags & tf::intersect::coplanar_pair_flag) != 0;
  }
  for (const auto &entry : points) {
    INFO("pair " << std::get<0>(entry.first) << ":" << std::get<1>(entry.first)
                 << " -> " << std::get<2>(entry.first) << ":"
                 << std::get<3>(entry.first));
    REQUIRE((entry.second.size() >= 2 || coplanar[entry.first]));
  }
}

auto build_pair(fan_delivery_fixture &a, fan_delivery_fixture &b)
    -> fan_delivery_ibp_t {
  fan_delivery_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_config{
                tf::intersect_mode::primitives |
                tf::intersect_mode::resolve_crossing_contours});
  return ibp;
}

/// A hexagonal pyramid: apex at `apex`, its rim a hexagon at `z`, closed by a
/// rim fan through the base centre. The apex has valence SIX, which is the
/// population the fan multiplies against the other side's.
auto hex_pyramid(int z, int apex_z) -> fan_delivery_mesh_t {
  std::vector<std::array<int, 3>> pts{{0, 0, apex_z}};
  const int rim[6][2] = {{6, 0}, {3, 6}, {-3, 6}, {-6, 0}, {-3, -6}, {3, -6}};
  for (auto &r : rim)
    pts.push_back({r[0], r[1], z});
  pts.push_back({0, 0, z});
  std::vector<std::array<fan_delivery_index_t, 3>> faces;
  for (fan_delivery_index_t k = 0; k < 6; ++k) {
    const fan_delivery_index_t n = 1 + (k + 1) % 6;
    faces.push_back({0, 1 + k, n});
    faces.push_back({7, n, 1 + k});
  }
  return make_fan_delivery_mesh(pts, faces);
}

} // namespace

// FAMILY E — two meshes touching at a SINGLE vertex, valence six on both
// sides. This is the class the census names: one proven contact, and a
// delivery population that is the PRODUCT of the two fans. The contact
// carries no chord, so nothing may be cut, and every incident face on both
// sides must still name the shared point.
TEST_CASE("fan: two solids meeting at one vertex, valence six both sides",
          "[intersect][fan][vv]") {
  fan_delivery_fixture a(hex_pyramid(-8, 0));
  fan_delivery_fixture b(hex_pyramid(8, 0));
  auto ibp = build_pair(a, b);

  REQUIRE(claims(ibp).size() > 0);
  check_both_orientations(ibp);
  std::vector<fan_delivery_mesh_t *> meshes{&a.mesh, &b.mesh};
  check_vertex_identity_synchronized(ibp, meshes);
  check_chord_partners_delivered(ibp);
  check_no_inert_pairs(ibp);

  // every face at the shared apex, on BOTH sides, learns the contact
  REQUIRE(delivered_faces(ibp, 0).size() >= 6);
  REQUIRE(delivered_faces(ibp, 1).size() >= 6);

  // the only pairs that survive are the ones a consumer still reads —
  // here the opposed side faces, which are exactly coplanar through the
  // apex — and between them they already name all twelve faces, so the
  // identity currency is the DIFFERENCE and states nothing at all
  std::set<std::pair<int, fan_delivery_index_t>> claimed;
  for (const auto &c : claims(ibp))
    claimed.emplace(c.tag, c.object);
  REQUIRE(claimed.size() == 12);
  REQUIRE(ibp.flat_deliveries().size() == 0);
  for (const auto &r : ibp.flat_intersections())
    REQUIRE((r.flags & tf::intersect::coplanar_pair_flag) != 0);

  // a single-point contact bounds no chord: the arrangement keeps both
  // solids whole
  std::vector<decltype(a.form())> forms{a.form(), b.form()};
  auto [arr, tags, faces] = tf::make_mesh_arrangements(tf::make_range(forms));
  REQUIRE(tf::is_closed(arr.polygons()));
  REQUIRE(std::size_t(arr.polygons().size()) ==
          std::size_t(a.mesh.polygons().size() + b.mesh.polygons().size()));
}

// FAMILY C — the INERT class, stated directly. A fan of faces around a
// touched vertex where exactly ONE pair actually crosses: every incident face
// must still name the created point, and no face may gain a chord it did not
// earn.
TEST_CASE("fan: a touched vertex delivers identity without inventing chords",
          "[intersect][fan][inert]") {
  fan_delivery_fixture a(hex_pyramid(-8, 0));
  // a blade through one of A's sides, meeting the apex
  fan_delivery_fixture b(make_fan_delivery_mesh(
      {{0, 0, 0}, {20, 1, -20}, {20, -1, -20}, {20, 0, -6}},
      {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}}));
  auto ibp = build_pair(a, b);

  check_both_orientations(ibp);
  check_chord_partners_delivered(ibp);
  check_no_inert_pairs(ibp);
  std::vector<fan_delivery_mesh_t *> meshes{&a.mesh, &b.mesh};
  check_vertex_identity_synchronized(ibp, meshes);

  // the apex is shared by all six of A's side faces; each must name it
  std::set<fan_delivery_index_t> apex_faces;
  for (const auto &c : claims(ibp)) {
    if (c.tag != 0 || c.target.label != tf::topo_type::vertex)
      continue;
    const auto face = a.mesh.polygons().faces()[c.object];
    if (fan_delivery_index_t(face[std::size_t(c.target.id)]) ==
        fan_delivery_index_t(0))
      apex_faces.insert(c.object);
  }
  REQUIRE(apex_faces.size() >= 6);
}

// FAMILY A — a self pair sharing vertices, with a crossing that continues
// THROUGH the shared vertex. This is the sentinel path: the shared vertices
// are delivered as sentinel ids around the vertex's own face fan and resolved
// after the generate pass.
TEST_CASE("fan: self pair sharing a vertex delivers it around the whole fan",
          "[intersect][fan][self][sentinel]") {
  // two blades of ONE mesh meeting along a shared vertex and crossing
  auto m = make_fan_delivery_mesh({{0, 0, 0},
                                   {10, 0, 0},
                                   {0, 10, 0},
                                   {-10, 0, 0},
                                   {0, -10, 0},
                                   {0, 0, 10},
                                   {0, 0, -10},
                                   {6, 6, 0},
                                   {-6, -6, 0}},
                                  {{0, 1, 2},
                                   {0, 2, 3},
                                   {0, 3, 4},
                                   {0, 4, 1},
                                   {0, 5, 7},
                                   {0, 7, 6},
                                   {0, 6, 8},
                                   {0, 8, 5}});
  fan_delivery_fixture f(m);
  fan_delivery_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(f.form(), 0.0);
  ibp.build(f.form(), ibp_lattice, tf::intersect_config{tf::intersect_mode::primitives |
                                           tf::intersect_mode::within});

  // the self family's emissions are the discovery site of its own
  // shared-vertex deliveries, so its fan is not a product the pair gate
  // prunes — @ref check_no_inert_pairs does not hold here, by design
  REQUIRE(ibp.flat_intersections().size() > 0);
  check_both_orientations(ibp);
  std::vector<fan_delivery_mesh_t *> meshes{&f.mesh};
  check_vertex_identity_synchronized(ibp, meshes);
  check_chord_partners_delivered(ibp);

  // the shared vertex 0 sits on all eight faces; the delivery must reach
  // every one of them
  REQUIRE(delivered_faces(ibp, 0).size() == 8);
}

// FAMILY B — a NON-MANIFOLD edge: three faces meet on one edge and a cutter
// lands on it. The edge's representative is ONE of the three faces, so the
// kernel states the contact once; the expansion walks `face_edge_neighbors`
// rather than the manifold peer, and every incident face must be told the
// point AND the slot it sits on.
TEST_CASE("fan: a record on a three-face edge reaches every incident face",
          "[intersect][fan][non-manifold]") {
  // edge (0,1) carries three faces
  auto m = make_fan_delivery_mesh(
      {{0, 0, 0}, {12, 0, 0}, {6, 8, 0}, {6, -8, 0}, {6, 0, 8}},
      {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}});
  fan_delivery_fixture a(m);
  // B's own VERTEX rests on the interior of that edge and B goes nowhere near
  // the three faces otherwise: the contact is stated once, against the EDGE,
  // so the only route to all three incident faces is the edge expansion
  fan_delivery_fixture b(make_fan_delivery_mesh(
      {{6, 0, 0}, {6, 20, 20}, {6, -20, 20}, {30, 0, 20}},
      {{0, 1, 2}, {0, 2, 3}, {0, 3, 1}, {1, 3, 2}}));
  auto ibp = build_pair(a, b);

  REQUIRE(claims(ibp).size() > 0);
  check_both_orientations(ibp);
  check_chord_partners_delivered(ibp);
  check_no_inert_pairs(ibp);

  // all three faces on the non-manifold edge must be told
  REQUIRE(delivered_faces(ibp, 0).size() == 3);

  // and each must be told it CONFORMALLY: one point, sitting on that face's
  // own instance of the shared edge. This is what the expansion states and
  // the kernel cannot — a non-manifold edge elects a single representative
  // face, so only one of the three ever emits.
  std::set<fan_delivery_index_t> conformal;
  std::set<fan_delivery_index_t> ids;
  for (const auto &c : claims(ibp)) {
    if (c.tag != 0 || c.target.label != tf::topo_type::edge)
      continue;
    const auto face = a.mesh.polygons().faces()[c.object];
    const auto u = fan_delivery_index_t(face[std::size_t(c.target.id)]);
    const auto v =
        fan_delivery_index_t(face[std::size_t((c.target.id + 1) % 3)]);
    if (std::min(u, v) != fan_delivery_index_t(0) ||
        std::max(u, v) != fan_delivery_index_t(1))
      continue;
    conformal.insert(c.object);
    ids.insert(c.id);
  }
  REQUIRE(conformal.size() == 3);
  REQUIRE(ids.size() == 1);
}

// FAMILY D — a coplanar pair adjacent to a crossing pair. The coplanar fact
// belongs to the PAIR, and @ref tf::intersect::distribute_coplanar_flags is
// its one producer: a neighbouring crossing must not inherit it and the
// coplanar contact must not lose it.
TEST_CASE("fan: a coplanar contact beside a crossing keeps its own flag",
          "[intersect][fan][coplanar]") {
  // A is a square in z = 0. B carries BOTH the coincident square (faces 0,1)
  // and a blade crossing A transversally (faces 2,3), so the fan rewrites
  // copies of the coplanar contact onto the crossing pair.
  fan_delivery_fixture a(
      make_fan_delivery_mesh({{0, 0, 0}, {10, 0, 0}, {10, 10, 0}, {0, 10, 0}},
                             {{0, 1, 2}, {0, 2, 3}}));
  fan_delivery_fixture b(
      make_fan_delivery_mesh({{0, 0, 0},
                              {10, 0, 0},
                              {10, 10, 0},
                              {0, 10, 0},
                              {6, 2, -8},
                              {6, 2, 8},
                              {6, 14, 0}},
                             {{0, 1, 2}, {0, 2, 3}, {0, 4, 5}, {0, 5, 6}}));
  auto ibp = build_pair(a, b);

  REQUIRE(ibp.flat_intersections().size() > 0);
  check_both_orientations(ibp);
  check_chord_partners_delivered(ibp);
  check_no_inert_pairs(ibp);

  // THE SCOPE: the flag describes the PAIR a record sits in. B's faces 0 and
  // 1 are the coincident square; 2 and 3 are the blade. A record must carry
  // the flag exactly when its own pair is the coincident one — which is why
  // the fan cannot carry a parent's flag onto a copy it rewrites, and does
  // not: the pair table is stamped onto the finished records.
  std::size_t coplanar_pair_records = 0, crossing_pair_records = 0;
  for (const auto &r : ibp.flat_intersections()) {
    const auto square_side = [](int tag, fan_delivery_index_t object) {
      return tag == 0 || object < fan_delivery_index_t(2);
    };
    const bool coincident_pair =
        square_side(int(r.tag), r.object) &&
        square_side(int(r.tag_other), r.object_other);
    const bool flagged = (r.flags & tf::intersect::coplanar_pair_flag) != 0;
    INFO("tag=" << int(r.tag) << ":" << r.object << " other="
                << int(r.tag_other) << ":" << r.object_other
                << " flagged=" << int(flagged));
    REQUIRE(flagged == coincident_pair);
    coplanar_pair_records += coincident_pair ? 1u : 0u;
    crossing_pair_records += coincident_pair ? 0u : 1u;
  }
  REQUIRE(coplanar_pair_records > 0);
  REQUIRE(crossing_pair_records > 0);
}

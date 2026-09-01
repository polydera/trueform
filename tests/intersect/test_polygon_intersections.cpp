/**
 * @file test_polygon_intersections.cpp
 * @brief tf::polygon_intersections — canonical point identity.
 *
 * Fixtures use integer coordinates so the converter is the identity and
 * every incidence below is exact on the lattice, not merely close.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/edge_edge_parameter.hpp>
#include <trueform/exact/edge_parameter.hpp>
#include <trueform/exact/edge_plane_parameter.hpp>
#include <trueform/exact/edge_point_parameter.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/exact/vertex.hpp>
#include <trueform/intersect/classify/face_plane_info.hpp>
#include <trueform/intersect/identity/materialize_intersection_points.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/topo_type.hpp>

#include "input_lattice_for.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using poly_intersections_index_t = int;
using poly_intersections_mesh_t =
    tf::polygons_buffer<poly_intersections_index_t, int, 3, 3>;
using poly_intersections_int_t = tf::exact::int32;
using poly_intersections_wide_t = tf::exact::meta<poly_intersections_int_t>::T2;
using poly_intersections_ibp_t =
    tf::polygon_intersections<poly_intersections_index_t, int,
                              poly_intersections_int_t>;

auto make_poly_intersections_mesh(
    const std::vector<std::array<int, 3>> &pts,
    const std::vector<std::array<poly_intersections_index_t, 3>> &faces)
    -> poly_intersections_mesh_t {
  poly_intersections_mesh_t mesh;
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

struct poly_intersections_fixture {
  poly_intersections_mesh_t mesh;
  tf::aabb_tree<poly_intersections_index_t, int, 3> tree;
  decltype(tf::make_face_membership(
      std::declval<poly_intersections_mesh_t &>().polygons())) fm;
  decltype(tf::make_manifold_edge_link(
      std::declval<poly_intersections_mesh_t &>().polygons())) mel;

  explicit poly_intersections_fixture(poly_intersections_mesh_t m)
      : mesh(std::move(m)), tree(mesh.polygons(), tf::config_tree(4, 4)),
        fm(tf::make_face_membership(mesh.polygons())),
        mel(tf::make_manifold_edge_link(mesh.polygons())) {}

  auto form() {
    return mesh.polygons() | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);
  }
};

auto poly_intersections_ratio(int num, int den)
    -> tf::exact::edge_parameter<poly_intersections_int_t> {
  return {poly_intersections_wide_t(num), poly_intersections_wide_t(den)};
}

auto face_ids(poly_intersections_mesh_t &mesh,
              poly_intersections_index_t object)
    -> std::array<poly_intersections_index_t, 3> {
  auto face = mesh.polygons().faces()[object];
  return {poly_intersections_index_t(face[0]),
          poly_intersections_index_t(face[1]),
          poly_intersections_index_t(face[2])};
}

/// One face's claim on a point, from whichever currency states it: a pair
/// record names the point at its own face as well as the pair it stands
/// on, a delivery names only the point. The loop reads both, so a fixture
/// about identity reads both.
struct poly_intersections_claim {
  int tag;
  poly_intersections_index_t object;
  tf::topo_id<poly_intersections_index_t> target;
  poly_intersections_index_t id;
};

auto poly_intersections_claims(const poly_intersections_ibp_t &ibp)
    -> std::vector<poly_intersections_claim> {
  std::vector<poly_intersections_claim> out;
  for (const auto &r : ibp.flat_intersections())
    out.push_back({int(r.tag), r.object, r.target, r.id});
  for (const auto &d : ibp.flat_deliveries())
    out.push_back({int(d.tag), d.object, d.target, d.id});
  return out;
}

/// Every record names a point in the published id space, no two
/// records state the same generators, the groups partition the records,
/// and every kind-E point is listed on the carrier it calls home. A
/// point an edge-edge record put on two carriers appears in both blocks
/// and names one of them as its home.
auto check_tables_closed(const poly_intersections_ibp_t &ibp,
                         poly_intersections_index_t n_tags) -> void {
  for (const auto &rec : ibp.flat_intersections()) {
    REQUIRE(rec.id >= 0);
    REQUIRE(rec.id < ibp.n_points());
  }
  std::set<std::tuple<int, poly_intersections_index_t, int,
                      poly_intersections_index_t, poly_intersections_index_t,
                      poly_intersections_index_t, poly_intersections_index_t,
                      poly_intersections_index_t>>
      generators;
  for (const auto &rec : ibp.flat_intersections())
    REQUIRE(generators
                .insert({int(rec.tag), rec.object, int(rec.tag_other),
                         rec.object_other,
                         poly_intersections_index_t(rec.target.label),
                         rec.target.id,
                         poly_intersections_index_t(rec.target_other.label),
                         rec.target_other.id})
                .second);

  std::size_t grouped = 0;
  for (auto group : ibp.intersections())
    grouped += group.size();
  REQUIRE(grouped == ibp.flat_intersections().size());
  std::size_t per_tag = 0;
  for (poly_intersections_index_t t = 0; t < n_tags; ++t)
    for (auto group : ibp.intersections(t)) {
      per_tag += group.size();
      for (const auto &rec : group)
        REQUIRE(rec.tag == t);
    }
  REQUIRE(per_tag == ibp.flat_intersections().size());

  // ONE CARRIER, TWO CURRENCIES at the same block position: a face is
  // there because one of them names it, and neither may name another
  // face's block.
  REQUIRE(ibp.intersections().size() == ibp.deliveries().size());
  std::size_t delivered = 0, delivered_per_tag = 0;
  for (std::size_t b = 0; b < std::size_t(ibp.deliveries().size()); ++b) {
    auto block = ibp.deliveries()[b];
    auto pairs = ibp.intersections()[b];
    delivered += block.size();
    REQUIRE((block.size() != 0 || pairs.size() != 0));
    const auto tag = pairs.size() != 0
                         ? poly_intersections_index_t(pairs[0].tag)
                         : poly_intersections_index_t(block[0].tag);
    const auto object = pairs.size() != 0 ? pairs[0].object : block[0].object;
    for (const auto &d : block) {
      REQUIRE(poly_intersections_index_t(d.tag) == tag);
      REQUIRE(d.object == object);
    }
    for (const auto &rec : pairs) {
      REQUIRE(poly_intersections_index_t(rec.tag) == tag);
      REQUIRE(rec.object == object);
    }
  }
  REQUIRE(delivered == ibp.flat_deliveries().size());
  for (poly_intersections_index_t t = 0; t < n_tags; ++t)
    for (auto block : ibp.deliveries(t)) {
      delivered_per_tag += block.size();
      for (const auto &d : block)
        REQUIRE(d.tag == t);
    }
  REQUIRE(delivered_per_tag == ibp.flat_deliveries().size());

  auto carriers = ibp.edge_carriers();
  auto splits = ibp.edge_splits();
  REQUIRE(std::size_t(splits.size()) == carriers.size());
  for (std::size_t g = 0; g < carriers.size(); ++g) {
    std::set<poly_intersections_index_t> seen;
    for (auto id : splits[g]) {
      REQUIRE(id >= 0);
      REQUIRE(id < ibp.n_points());
      REQUIRE(seen.insert(id).second);
    }
  }
  for (auto id = ibp.n_vertex_points(); id < ibp.n_points(); ++id) {
    const auto home = ibp.home_edge(id);
    bool listed = false;
    for (std::size_t g = 0; g < carriers.size() && !listed; ++g) {
      if (carriers[g].u != home.u || carriers[g].v != home.v)
        continue;
      for (auto other : splits[g])
        listed = listed || other == id;
    }
    REQUIRE(listed);
  }
}

/// The identity law, read back on BOTH of the fan's carriers: a vertex
/// target names the kind-V point anchored at that vertex's canonical
/// class, and a point is listed on the canonical carrier of every edge
/// target it has — a vertex lying on an edge subdivides it. A contact
/// without a vertex target is not required to name a kind-E point: a
/// pierce that lands exactly on an original vertex fuses with it and
/// takes its identity.
///
/// A delivery states the same law for one side alone, and it states it
/// for every face the feature reaches — including the pairs the fan's
/// gate proved inert, which hold no record.
template <typename FaceOf>
auto check_records_named(const poly_intersections_ibp_t &ibp,
                         const FaceOf &face_of) -> void {
  auto carriers = ibp.edge_carriers();
  auto splits = ibp.edge_splits();
  auto offsets = ibp.vertex_offsets();

  auto listed_on = [&](poly_intersections_index_t u,
                       poly_intersections_index_t v,
                       poly_intersections_index_t id) {
    for (std::size_t g = 0; g < carriers.size(); ++g) {
      if (carriers[g].u != u || carriers[g].v != v)
        continue;
      for (auto other : splits[g])
        if (other == id)
          return true;
    }
    return false;
  };
  auto check_edge = [&](std::int16_t tag, poly_intersections_index_t object,
                        poly_intersections_index_t local,
                        poly_intersections_index_t id) {
    auto face = face_of(int(tag), object);
    auto u = ibp.canonical_vertex(tag, face[std::size_t(local)]);
    auto v = ibp.canonical_vertex(tag, face[std::size_t((local + 1) % 3)]);
    REQUIRE(listed_on(std::min(u, v), std::max(u, v), id));
  };
  auto check_vertex = [&](std::int16_t tag, poly_intersections_index_t object,
                          poly_intersections_index_t local,
                          poly_intersections_index_t flat) {
    auto face = face_of(int(tag), object);
    REQUIRE(ibp.canonical_vertex(tag, face[std::size_t(local)]) == flat);
  };

  for (const auto &rec : ibp.flat_intersections()) {
    const bool at_target = rec.target.label == tf::topo_type::vertex;
    const bool at_other = rec.target_other.label == tf::topo_type::vertex;
    if (at_target || at_other) {
      REQUIRE(rec.id < ibp.n_vertex_points());
      const auto anchor = ibp.vertex_anchor(rec.id);
      const auto flat = offsets[std::size_t(anchor.tag)] + anchor.vid;
      if (at_target)
        check_vertex(rec.tag, rec.object, rec.target.id, flat);
      if (at_other)
        check_vertex(rec.tag_other, rec.object_other, rec.target_other.id,
                     flat);
    }
    if (rec.target.label == tf::topo_type::edge)
      check_edge(rec.tag, rec.object, rec.target.id, rec.id);
    if (rec.target_other.label == tf::topo_type::edge)
      check_edge(rec.tag_other, rec.object_other, rec.target_other.id, rec.id);
  }

  for (const auto &d : ibp.flat_deliveries()) {
    REQUIRE(d.id >= 0);
    REQUIRE(d.id < ibp.n_points());
    if (d.target.label == tf::topo_type::vertex) {
      REQUIRE(d.id < ibp.n_vertex_points());
      const auto anchor = ibp.vertex_anchor(d.id);
      check_vertex(d.tag, d.object, d.target.id,
                   offsets[std::size_t(anchor.tag)] + anchor.vid);
    }
    if (d.target.label == tf::topo_type::edge)
      check_edge(d.tag, d.object, d.target.id, d.id);
  }
}

/// An independent rederivation: a record's exact parameter rebuilt from
/// its own two targets, on the canonical carrier's coordinates. The
/// kernels state that parameter where they decide the incidence and it
/// travels on the record's id through duplication, transposition and
/// reversal; here the generators say it again, independently, and the
/// two must name one position.
///
/// Every published kind-E parameter is checked against its record, and
/// every carrier's split order is checked against the rederived
/// positions of the points listed on it — which also covers the
/// vertex-on-edge candidates, whose identity is a vertex and whose
/// parameter is therefore never published.
template <typename Converter>
auto check_parameters_match(
    const poly_intersections_ibp_t &ibp, const Converter &converter,
    const std::vector<poly_intersections_mesh_t *> &meshes) -> void {
  using vertex_t =
      tf::exact::vertex<poly_intersections_index_t, poly_intersections_int_t>;
  auto offsets = ibp.vertex_offsets();

  auto face_of = [&](int tag, poly_intersections_index_t object) {
    return face_ids(*meshes[std::size_t(tag)], object);
  };
  auto point_of = [&](int tag, poly_intersections_index_t vid) {
    return converter.convert(
        meshes[std::size_t(tag)]->polygons().points()[vid]);
  };
  auto lattice_of = [&](poly_intersections_index_t flat) {
    auto tag = int(std::upper_bound(offsets.begin(), offsets.end(), flat) -
                   offsets.begin()) -
               1;
    return point_of(tag, flat - offsets[std::size_t(tag)]);
  };
  auto carrier_of = [&](std::int16_t tag, poly_intersections_index_t object,
                        poly_intersections_index_t local) {
    auto face = face_of(int(tag), object);
    auto a = ibp.canonical_vertex(tag, face[std::size_t(local)]);
    auto b = ibp.canonical_vertex(tag, face[std::size_t((local + 1) % 3)]);
    return std::pair<poly_intersections_index_t, poly_intersections_index_t>{
        std::min(a, b), std::max(a, b)};
  };

  auto rederive =
      [&](const auto &rec,
          int side) -> tf::exact::edge_parameter<poly_intersections_int_t> {
    auto tag = side == 0 ? rec.tag : rec.tag_other;
    auto object = side == 0 ? rec.object : rec.object_other;
    auto local = side == 0 ? rec.target.id : rec.target_other.id;
    auto tag_o = side == 0 ? rec.tag_other : rec.tag;
    auto obj_o = side == 0 ? rec.object_other : rec.object;
    auto tgt_o = side == 0 ? rec.target_other : rec.target;
    auto carrier = carrier_of(tag, object, local);
    auto pu = lattice_of(carrier.first);
    auto pv = lattice_of(carrier.second);
    auto face = face_of(int(tag_o), obj_o);

    if (tgt_o.label == tf::topo_type::face) {
      std::vector<vertex_t> verts;
      for (std::size_t k = 0; k < 3; ++k)
        verts.push_back({face[k], point_of(int(tag_o), face[k])});
      const vertex_t *first = verts.data();
      auto plane = tf::exact::make_face_plane(
          tf::make_range(first, first + verts.size()));
      return tf::exact::make_edge_plane_parameter<poly_intersections_int_t>(
          verts[plane.info.i0].pt, verts[plane.info.i1].pt,
          verts[plane.info.i2].pt, pu, pv);
    }
    if (tgt_o.label == tf::topo_type::vertex)
      return tf::exact::make_edge_point_parameter<poly_intersections_int_t>(
          pu, pv,
          lattice_of(ibp.canonical_vertex(tag_o, face[std::size_t(tgt_o.id)])));

    auto q0 = point_of(int(tag_o), face[std::size_t(tgt_o.id)]);
    auto q1 = point_of(int(tag_o), face[std::size_t((tgt_o.id + 1) % 3)]);
    return tf::exact::make_edge_edge_parameter<poly_intersections_int_t>(
        pu, pv, q0, q1,
        tf::exact::projection_axes_edges<poly_intersections_int_t>(pu, pv, q0,
                                                                   q1));
  };

  /// The record's edge target on `carrier`, if it has one.
  auto side_on = [&](const auto &rec, poly_intersections_index_t u,
                     poly_intersections_index_t v) {
    for (int side = 0; side < 2; ++side) {
      auto target = side == 0 ? rec.target : rec.target_other;
      if (target.label != tf::topo_type::edge)
        continue;
      auto carrier = carrier_of(side == 0 ? rec.tag : rec.tag_other,
                                side == 0 ? rec.object : rec.object_other,
                                target.id);
      if (carrier.first == u && carrier.second == v)
        return side;
    }
    return -1;
  };

  int compared = 0;
  for (const auto &rec : ibp.flat_intersections()) {
    if (rec.id < ibp.n_vertex_points())
      continue;
    auto home = ibp.home_edge(rec.id);
    auto side = side_on(rec, home.u, home.v);
    if (side < 0)
      continue;
    REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(rec.id),
                                         rederive(rec, side)) == 0);
    ++compared;
  }

  auto carriers = ibp.edge_carriers();
  auto splits = ibp.edge_splits();
  for (std::size_t g = 0; g < carriers.size(); ++g) {
    std::vector<tf::exact::edge_parameter<poly_intersections_int_t>> along;
    for (auto id : splits[g]) {
      bool found = false;
      for (const auto &rec : ibp.flat_intersections()) {
        if (rec.id != id)
          continue;
        auto side = side_on(rec, carriers[g].u, carriers[g].v);
        if (side < 0)
          continue;
        along.push_back(rederive(rec, side));
        found = true;
        break;
      }
      REQUIRE(found);
      ++compared;
    }
    for (std::size_t k = 1; k < along.size(); ++k)
      REQUIRE(tf::exact::compare_parameter(along[k - 1], along[k]) < 0);
  }
  // A carrier exists exactly when some record put a point on an edge, so
  // this is the oracle refusing to pass on an empty set.
  REQUIRE((compared != 0) == (carriers.size() != 0));
}

auto poly_intersections_carrier_block(const poly_intersections_ibp_t &ibp,
                                      poly_intersections_index_t u,
                                      poly_intersections_index_t v)
    -> std::vector<poly_intersections_index_t> {
  auto carriers = ibp.edge_carriers();
  std::vector<poly_intersections_index_t> found;
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

/// The cutter used by the carrier-identity cases: a triangle in the
/// plane x = 4 whose first edge pierces z = 0 exactly at (4, 4, 0).
auto make_cutter() -> poly_intersections_mesh_t {
  return make_poly_intersections_mesh({{{4, 4, -5}}, {{4, 4, 5}}, {{4, 30, 0}}},
                                      {{{0, 1, 2}}});
}

} // namespace

TEST_CASE("polygon_intersections: pierced edges carry ordered splits",
          "[intersect][identity]") {
  // Two parallel triangles of A at z = 0 and z = 4; one edge of B runs
  // from z = -4 to z = 8 through both, at exact parameters 1/3 and 2/3.
  poly_intersections_fixture a(
      make_poly_intersections_mesh({{{0, 0, 0}},
                                    {{3, 0, 0}},
                                    {{0, 3, 0}},
                                    {{0, 0, 4}},
                                    {{3, 0, 4}},
                                    {{0, 3, 4}}},
                                   {{{0, 1, 2}}, {{3, 4, 5}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{1, 1, -4}}, {{1, 1, 8}}, {{50, 1, 2}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.n_vertex_points() == 0);
  REQUIRE(ibp.n_points() == 4);
  REQUIRE(ibp.edge_carriers().size() == 3);

  auto block = poly_intersections_carrier_block(ibp, 6, 7);
  REQUIRE(block.size() == 2);
  REQUIRE(block[0] != block[1]);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[0]),
                                       ibp.exact_parameter(block[1])) < 0);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[0]),
                                       poly_intersections_ratio(1, 3)) == 0);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[1]),
                                       poly_intersections_ratio(2, 3)) == 0);
}

TEST_CASE("polygon_intersections: a vertex on a face plane is a kind V point",
          "[intersect][identity]") {
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{-4, -4, 0}}, {{4, -4, 0}}, {{0, 4, 0}}}, {{{0, 1, 2}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{0, -8, 8}}, {{8, 8, 8}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.n_vertex_points() == 1);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.edge_carriers().size() == 0);
  REQUIRE(ibp.vertex_anchor(0).tag == 1);
  REQUIRE(ibp.vertex_anchor(0).vid == 0);
  REQUIRE(poly_intersections_claims(ibp).size() != 0);
  for (const auto &c : poly_intersections_claims(ibp))
    REQUIRE(c.id == 0);
}

TEST_CASE("polygon_intersections: two faces cutting one edge at one "
          "parameter are one point",
          "[intersect][identity]") {
  // A's two faces lie in different planes (z = 0 and z = x) and both
  // contain the origin; B's first edge is the z axis, so both cuts land
  // on the same exact parameter of the same carrier.
  poly_intersections_fixture a(
      make_poly_intersections_mesh({{{-4, -4, 0}},
                                    {{4, -4, 0}},
                                    {{0, 4, 0}},
                                    {{-5, -4, -5}},
                                    {{5, -4, 5}},
                                    {{0, 5, 0}}},
                                   {{{0, 1, 2}}, {{3, 4, 5}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, -4}}, {{0, 0, 4}}, {{30, 0, 1}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.n_vertex_points() == 0);
  auto block = poly_intersections_carrier_block(ibp, 6, 7);
  REQUIRE(block.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[0]),
                                       poly_intersections_ratio(1, 2)) == 0);

  std::set<poly_intersections_index_t> ids_on_edge;
  int faces_seen = 0;
  for (const auto &rec : ibp.flat_intersections()) {
    if (rec.tag != 1 || rec.target.label != tf::topo_type::edge ||
        rec.target.id != 0)
      continue;
    ++faces_seen;
    ids_on_edge.insert(rec.id);
  }
  REQUIRE(faces_seen == 2);
  REQUIRE(ids_on_edge.size() == 1);
  REQUIRE(*ids_on_edge.begin() == block[0]);
}

TEST_CASE("polygon_intersections: a shared corner is one kind V point",
          "[intersect][identity]") {
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{0, -6, 6}}, {{-6, 0, 6}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.n_vertex_points() == 1);
  REQUIRE(ibp.n_points() == 1);
  REQUIRE(ibp.vertex_anchor(0).tag == 0);
  REQUIRE(ibp.vertex_anchor(0).vid == 0);
  REQUIRE(ibp.canonical_vertex(0, 0) == 0);
  REQUIRE(ibp.canonical_vertex(1, 0) == 0);
  REQUIRE(poly_intersections_claims(ibp).size() != 0);
  for (const auto &c : poly_intersections_claims(ibp))
    REQUIRE(c.id == 0);
}

TEST_CASE("polygon_intersections: opposed instances of one edge share a "
          "carrier",
          "[intersect][identity]") {
  // The diagonal is edge 2 of face 0 running vertex 2 -> 0 and edge 0 of
  // face 1 running vertex 0 -> 2.
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{8, 8, 0}}, {{0, 8, 0}}},
      {{{0, 1, 2}}, {{0, 2, 3}}}));
  poly_intersections_fixture b(make_cutter());

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  auto diagonal = poly_intersections_carrier_block(ibp, 0, 2);
  REQUIRE(diagonal.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(diagonal[0]),
                                       poly_intersections_ratio(1, 2)) == 0);

  // The same point is a class on the cutter's edge too, and the
  // edge-edge equivalence collapsed the two classes.
  auto cutter = poly_intersections_carrier_block(ibp, 4, 5);
  REQUIRE(cutter.size() == 1);
  REQUIRE(cutter[0] == diagonal[0]);

  std::set<poly_intersections_index_t> instance_faces;
  for (const auto &c : poly_intersections_claims(ibp)) {
    if (c.tag != 0 || c.target.label != tf::topo_type::edge)
      continue;
    auto face = a.mesh.polygons().faces()[c.object];
    auto u = poly_intersections_index_t(face[c.target.id]);
    auto v = poly_intersections_index_t(face[(c.target.id + 1) % 3]);
    if (std::min(u, v) != 0 || std::max(u, v) != 2)
      continue;
    instance_faces.insert(c.object);
    REQUIRE(c.id == diagonal[0]);
  }
  const int instances = int(instance_faces.size());
  REQUIRE(instances == 2);
}

TEST_CASE("polygon_intersections: duplicated vertices name one carrier",
          "[intersect][identity]") {
  // One form, two faces meeting along a diagonal spelled by two pairs of
  // duplicated vertex ids. The self pass identifies them, so both edge
  // instances key on the same canonical carrier.
  poly_intersections_fixture a(
      make_poly_intersections_mesh({{{0, 0, 0}},
                                    {{8, 8, 0}},
                                    {{8, 0, 0}},
                                    {{0, 0, 0}},
                                    {{8, 8, 0}},
                                    {{0, 8, 0}}},
                                   {{{0, 1, 2}}, {{3, 4, 5}}}));
  poly_intersections_fixture b(make_cutter());

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_config(tf::intersect_mode::primitives |
                                 tf::intersect_mode::within));
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.canonical_vertex(0, 3) == 0);
  REQUIRE(ibp.canonical_vertex(0, 4) == 1);
  REQUIRE(ibp.n_vertex_points() == 2);

  auto diagonal = poly_intersections_carrier_block(ibp, 0, 1);
  REQUIRE(diagonal.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(diagonal[0]),
                                       poly_intersections_ratio(1, 2)) == 0);

  std::set<poly_intersections_index_t> faces_on_diagonal;
  for (const auto &c : poly_intersections_claims(ibp)) {
    if (c.tag != 0 || c.target.label != tf::topo_type::edge ||
        c.target.id != 0)
      continue;
    faces_on_diagonal.insert(c.object);
    REQUIRE(c.id == diagonal[0]);
  }
  REQUIRE(faces_on_diagonal.size() == 2);
}

TEST_CASE("polygon_intersections: an edge shared across forms is one carrier",
          "[intersect][identity]") {
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{8, 8, 0}}, {{8, 0, 0}}}, {{{0, 1, 2}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{8, 8, 0}}, {{0, 8, 0}}}, {{{0, 1, 2}}}));
  poly_intersections_fixture c(make_cutter());

  auto fa = a.form();
  decltype(fa) forms[] = {fa, b.form(), c.form()};

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(tf::make_range(forms, forms + 3), 0.0);
  ibp.build(tf::make_range(forms, forms + 3), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 3);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : tag == 1 ? b.mesh : c.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh, &c.mesh});

  REQUIRE(ibp.canonical_vertex(1, 0) == 0);
  REQUIRE(ibp.canonical_vertex(1, 1) == 1);
  REQUIRE(ibp.n_vertex_points() == 2);

  auto diagonal = poly_intersections_carrier_block(ibp, 0, 1);
  REQUIRE(diagonal.size() == 1);

  std::set<int> tags_on_diagonal;
  for (const auto &c : poly_intersections_claims(ibp)) {
    if (c.target.label != tf::topo_type::edge || c.target.id != 0 ||
        c.tag > 1)
      continue;
    tags_on_diagonal.insert(c.tag);
    REQUIRE(c.id == diagonal[0]);
  }
  REQUIRE(tags_on_diagonal.size() == 2);
}

TEST_CASE("polygon_intersections: records collapse on their generators",
          "[intersect][identity]") {
  // A coplanar fold over a shared edge with a duplicated apex. Every raw
  // record of the pair makes the self duplicator deliver the shared
  // vertices again, so the fan states the same facts several times; the
  // generator collapse is what makes each one appear once.
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{10, 0, 0}}, {{5, 10, 0}}, {{5, 10, 0}}},
      {{{0, 1, 2}}, {{0, 1, 3}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), 0.0);
  ibp.build(a.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 1);
  check_records_named(ibp, [&](int, poly_intersections_index_t object) {
    return face_ids(a.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh});

  // The two shared-edge endpoints and the identified apex, nothing else,
  // and one record per (face pair, targets) statement.
  REQUIRE(ibp.n_points() == 3);
  REQUIRE(ibp.n_vertex_points() == 3);
  REQUIRE(ibp.flat_intersections().size() == 6);
  REQUIRE(ibp.canonical_vertex(0, 3) == 2);
}

TEST_CASE("polygon_intersections: a vertex on an edge takes its place in "
          "the carrier order",
          "[intersect][identity]") {
  // B's first edge is the segment (0,0,0) -> (0,0,12). Face 1 and face 2
  // of A pierce it at z = 3 and z = 9; face 0, coplanar with B, puts its
  // vertex 0 exactly on it at z = 6; face 4 pierces it at z = 6 too, so
  // the vertex construction and the plane construction must produce the
  // same exact parameter and fuse. Face 3 puts a vertex on the edge's
  // own endpoint (0,0,0).
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 6}},
       {{-8, 0, 2}},
       {{-8, 0, 10}},
       {{-5, -5, 3}},
       {{5, -5, 3}},
       {{0, 5, 3}},
       {{-5, -5, 9}},
       {{5, -5, 9}},
       {{0, 5, 9}},
       {{0, 0, 0}},
       {{-6, 4, -4}},
       {{-6, -4, -4}},
       {{-5, -5, 6}},
       {{5, -5, 6}},
       {{0, 5, 6}}},
      {{{0, 1, 2}}, {{3, 4, 5}}, {{6, 7, 8}}, {{9, 10, 11}}, {{12, 13, 14}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{0, 0, 12}}, {{40, 0, 4}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  // The endpoint is shared with A's vertex 9, so the carrier is keyed on
  // that class's representative.
  REQUIRE(ibp.canonical_vertex(1, 0) == 9);
  auto block = poly_intersections_carrier_block(ibp, 9, 16);
  REQUIRE(block.size() == 3);

  REQUIRE(block[0] >= ibp.n_vertex_points());
  REQUIRE(block[2] >= ibp.n_vertex_points());
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[0]),
                                       poly_intersections_ratio(1, 4)) == 0);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[2]),
                                       poly_intersections_ratio(3, 4)) == 0);

  // The vertex incidence keeps its vertex identity and sits between the
  // two pierce points.
  REQUIRE(block[1] < ibp.n_vertex_points());
  REQUIRE(ibp.vertex_anchor(block[1]).tag == 0);
  REQUIRE(ibp.vertex_anchor(block[1]).vid == 0);

  // Face 4 pierces the carrier at the same exact position, so the plane
  // construction and the vertex construction land in one class and that
  // class is the vertex.
  int coincident_pierces = 0;
  for (const auto &rec : ibp.flat_intersections()) {
    if (rec.tag != 0 || rec.object != 4 ||
        rec.target.label != tf::topo_type::face ||
        rec.target_other.label != tf::topo_type::edge)
      continue;
    ++coincident_pierces;
    REQUIRE(rec.id == block[1]);
  }
  REQUIRE(coincident_pierces == 1);

  // The vertex coincident with the carrier's own endpoint is a point of
  // its own, and it does not enter the carrier's split list.
  poly_intersections_index_t endpoint_point = -1;
  for (const auto &c : poly_intersections_claims(ibp))
    if (c.tag == 0 && c.object == 3)
      endpoint_point = c.id;
  REQUIRE(endpoint_point >= 0);
  REQUIRE(endpoint_point < ibp.n_vertex_points());
  REQUIRE(ibp.vertex_anchor(endpoint_point).vid == 9);
  REQUIRE(std::find(block.begin(), block.end(), endpoint_point) == block.end());
}

TEST_CASE("polygon_intersections: a carrier that runs against its edge",
          "[intersect][identity]") {
  // B's first edge runs (0,0,-4) -> (0,0,4) and its far endpoint is A's
  // vertex 0, which is the lowest id of the class. The carrier therefore
  // runs the other way round than the edge does, and the parameter of
  // the pierce at z = 2 — 3/4 along the edge — is 1/4 along the carrier.
  poly_intersections_fixture a(
      make_poly_intersections_mesh({{{0, 0, 4}},
                                    {{-6, 4, 8}},
                                    {{-6, -4, 8}},
                                    {{-5, -5, 2}},
                                    {{5, -5, 2}},
                                    {{0, 5, 2}}},
                                   {{{0, 1, 2}}, {{3, 4, 5}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{0, 0, -4}}, {{0, 0, 4}}, {{30, 0, 1}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  REQUIRE(ibp.canonical_vertex(1, 1) == 0);
  auto block = poly_intersections_carrier_block(ibp, 0, 6);
  REQUIRE(block.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(block[0]),
                                       poly_intersections_ratio(1, 4)) == 0);
}

TEST_CASE("polygon_intersections: an edge-edge point is a different "
          "parameter on each carrier",
          "[intersect][identity]") {
  // The cutter's first edge crosses A's diagonal at (2, 2, 0): one
  // eighth along the cutter, one quarter along the diagonal. The point
  // is one point on two carriers, so each carrier must be handed the
  // fraction that belongs to it — including in the copies the
  // duplicator states the other way round.
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{8, 0, 0}}, {{8, 8, 0}}, {{0, 8, 0}}},
      {{{0, 1, 2}}, {{0, 2, 3}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{{2, 2, -1}}, {{2, 2, 7}}, {{2, 30, 3}}}, {{{0, 1, 2}}}));

  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  check_tables_closed(ibp, 2);
  check_records_named(ibp, [&](int tag, poly_intersections_index_t object) {
    return face_ids(tag == 0 ? a.mesh : b.mesh, object);
  });
  check_parameters_match(ibp, ibp_lattice.converter(), {&a.mesh, &b.mesh});

  auto diagonal = poly_intersections_carrier_block(ibp, 0, 2);
  REQUIRE(diagonal.size() == 1);
  REQUIRE(tf::exact::compare_parameter(ibp.exact_parameter(diagonal[0]),
                                       poly_intersections_ratio(1, 4)) == 0);

  auto cutter = poly_intersections_carrier_block(ibp, 4, 5);
  REQUIRE(cutter.size() == 1);
  REQUIRE(cutter[0] == diagonal[0]);
}

TEST_CASE("polygon_intersections: materialized positions sit on their "
          "carriers",
          "[intersect][identity]") {
  // The pierced-edge fixture again: two kind-E points at 1/3 and 2/3 of
  // a carrier, materialized by dyadic blend; a kind-V point copies its
  // original.
  poly_intersections_fixture a(make_poly_intersections_mesh(
      {{{0, 0, 0}}, {{12, 0, 0}}, {{0, 12, 0}}}, {{{0, 1, 2}}}));
  poly_intersections_fixture b(make_poly_intersections_mesh(
      {{4, -1, -6}, {4, 7, -6}, {4, 3, 6}, {8, -1, -6}, {8, 7, -6}, {8, 3, 6}},
      {{0, 1, 2}, {3, 4, 5}}));
  poly_intersections_ibp_t ibp;
  const auto ibp_lattice = tf::test::input_lattice_for(a.form(), b.form(), 0.0);
  ibp.build(a.form(), b.form(), ibp_lattice, tf::intersect_mode::primitives);
  tf::buffer<tf::exact::pt3<poly_intersections_int_t>> pts;
  auto get_flat_point = [&](poly_intersections_index_t flat)
      -> tf::exact::pt3<poly_intersections_int_t> {
    const auto off = ibp.vertex_offsets();
    const int tag = flat < off[1] ? 0 : 1;
    const auto vid = flat - off[std::size_t(tag)];
    tf::exact::pt3<poly_intersections_int_t> out;
    const auto &m = tag == 0 ? a.mesh : b.mesh;
    for (int d = 0; d < 3; ++d)
      out[d] = poly_intersections_int_t(m.points()[std::size_t(vid)][d]);
    return out;
  };
  tf::buffer<typename tf::exact::meta<poly_intersections_int_t>::param_type>
      params;
  tf::intersect::materialize_intersection_parameters<poly_intersections_index_t,
                                                     poly_intersections_int_t>(
      ibp, params);
  tf::intersect::materialize_intersection_points<poly_intersections_index_t,
                                                 poly_intersections_int_t>(
      ibp, get_flat_point, params, pts);
  REQUIRE(pts.size() == std::size_t(ibp.n_points()));
  for (poly_intersections_index_t id = ibp.n_vertex_points();
       id < ibp.n_points(); ++id) {
    const auto &e = ibp.home_edge(id);
    const auto u = get_flat_point(e.u);
    const auto v = get_flat_point(e.v);
    const auto &p = pts[std::size_t(id)];
    for (int d = 0; d < 3; ++d) {
      REQUIRE(p[d] >= std::min(u[d], v[d]));
      REQUIRE(p[d] <= std::max(u[d], v[d]));
    }
    REQUIRE((p[0] != u[0] || p[1] != u[1] || p[2] != u[2]));
  }
}

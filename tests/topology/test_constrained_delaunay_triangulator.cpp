#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/cdt_region_mode.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <array>
#include <initializer_list>

namespace {

using triangulator_index_t = int;
using Int = tf::exact::int32;
using Wide = tf::exact::int64;
using Cdt =
    tf::constrained_delaunay_triangulator<triangulator_index_t, Int, Int>;
using WideCdt =
    tf::constrained_delaunay_triangulator<triangulator_index_t, Wide, Wide>;

template <typename Coordinate>
auto make_triangulator_points(
    std::initializer_list<std::array<Coordinate, 2>> values)
    -> tf::points_buffer<Coordinate, 2> {
  tf::points_buffer<Coordinate, 2> points;
  for (const auto &value : values)
    points.push_back(tf::point<Coordinate, 2>{value[0], value[1]});
  return points;
}

/// True iff some triangle of the build carries the edge (a, b).
template <typename Triangulator>
auto has_edge(Triangulator &cdt, triangulator_index_t a, triangulator_index_t b)
    -> bool {
  for (auto face : cdt.make_faces())
    for (int c = 0; c < 3; ++c) {
      const triangulator_index_t u = face[std::size_t(c)];
      const triangulator_index_t v = face[std::size_t((c + 1) % 3)];
      if ((u == a && v == b) || (u == b && v == a))
        return true;
    }
  return false;
}

/// True iff the build kept (a, b) as a constrained edge of some triangle.
template <typename Triangulator>
auto has_constrained_edge(const Triangulator &cdt, triangulator_index_t a,
                          triangulator_index_t b) -> bool {
  bool found = false;
  cdt.for_each_face_adjacency([&](triangulator_index_t, triangulator_index_t v0,
                                  triangulator_index_t v1,
                                  triangulator_index_t v2, triangulator_index_t,
                                  const std::array<triangulator_index_t, 3> &,
                                  const std::array<bool, 3> &constrained) {
    const std::array<triangulator_index_t, 3> corners{v0, v1, v2};
    for (int c = 0; c < 3; ++c) {
      const triangulator_index_t u = corners[std::size_t(c)];
      const triangulator_index_t v = corners[std::size_t((c + 1) % 3)];
      if (((u == a && v == b) || (u == b && v == a)) &&
          constrained[std::size_t(c)])
        found = true;
    }
  });
  return found;
}

/// Sum of a triangle's corner coordinates: three times its centroid, which
/// lies strictly inside the region the triangle belongs to, so it classifies
/// the triangle without leaving the integer lattice.
template <typename Points, typename Face>
auto corner_sum(const Points &points, const Face &face) -> std::array<Int, 2> {
  std::array<Int, 2> sum{0, 0};
  for (int c = 0; c < 3; ++c) {
    const auto p = points[std::size_t(face[std::size_t(c)])];
    sum[0] += p[0];
    sum[1] += p[1];
  }
  return sum;
}

/// Is the centroid strictly inside the axis-aligned box [lo, hi]^2? Asked of
/// the corner sum, so the bounds are scaled by three.
auto centroid_inside(const std::array<Int, 2> &sum, Int lo, Int hi) -> bool {
  return sum[0] > 3 * lo && sum[0] < 3 * hi && sum[1] > 3 * lo &&
         sum[1] < 3 * hi;
}

/// One past the largest label. The components labelling is dense from 0, so
/// this counts the regions -- the hull exterior included even when it owns
/// no triangle.
template <typename Labels>
auto n_labels(const Labels &labels) -> triangulator_index_t {
  triangulator_index_t n = 0;
  for (auto label : labels)
    n = label + 1 > n ? label + 1 : n;
  return n;
}

/// Labels one input both ways on one triangulator, and returns the nesting
/// parities while leaving `cdt` on the components build. The mode chooses
/// what a label means, not what is triangulated, so the two builds must
/// produce the same faces -- which is what lets the returned parities be
/// read against `cdt`'s triangles.
template <typename Points, typename Edges, typename Boundary>
auto label_both_modes(Cdt &cdt, const Points &pts, const Edges &edges,
                      const Boundary &is_boundary)
    -> tf::buffer<triangulator_index_t> {
  REQUIRE(cdt.build(pts, edges, is_boundary));
  const auto nesting_faces = cdt.make_faces();
  tf::buffer<triangulator_index_t> nesting;
  for (auto label : cdt.region_labels())
    nesting.push_back(label);

  REQUIRE(cdt.build(pts, edges, is_boundary, true,
                    tf::cdt_region_mode::components));
  const auto faces = cdt.make_faces();
  REQUIRE(faces.size() == nesting_faces.size());
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t)
    for (std::size_t c = 0; c < 3; ++c)
      REQUIRE(faces[t][c] == nesting_faces[t][c]);

  return nesting;
}

} // namespace

TEST_CASE("CDT crossings use welded output point IDs",
          "[constrained_delaunay]") {
  auto points =
      make_triangulator_points<Int>({{0, 0}, {4, 0}, {5, 1}, {2, 0}, {5, -1}});
  std::array<std::array<triangulator_index_t, 2>, 3> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 4}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  const triangulator_index_t junction = cdt.index_map().f()[3];
  CHECK(cdt.index_map().kept_ids()[junction] == 3);

  triangulator_index_t callback_count = 0;
  cdt.for_each_constraint_crossing(
      [&](triangulator_index_t point, const auto &constraints) {
        ++callback_count;
        CHECK(point == junction);
        REQUIRE(constraints.size() == 1);
        CHECK(constraints[0] == 0);
      });
  CHECK(callback_count == 1);
}

TEST_CASE("CDT crossings retain synthetic point constraint incidences",
          "[constrained_delaunay]") {
  auto points =
      make_triangulator_points<Int>({{-2, 0}, {2, 0}, {0, -2}, {0, 2}});
  std::array<std::array<triangulator_index_t, 2>, 3> edge_data{
      std::array<triangulator_index_t, 2>{0, 0},
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  triangulator_index_t crossing_point = triangulator_index_t(-1);
  triangulator_index_t reported = 0;
  cdt.for_each_constraint_crossing(
      [&](triangulator_index_t point, const auto &constraints) {
        ++reported;
        crossing_point = point;
        // Both constraints the point is interior to; the report states
        // which, not in what order.
        REQUIRE(constraints.size() == 2);
        const triangulator_index_t lo =
            constraints[0] < constraints[1] ? constraints[0] : constraints[1];
        const triangulator_index_t hi =
            constraints[0] < constraints[1] ? constraints[1] : constraints[0];
        CHECK(lo == 1);
        CHECK(hi == 2);
      });
  CHECK(reported == 1);
  REQUIRE(crossing_point != triangulator_index_t(-1));
  CHECK(cdt.index_map().kept_ids()[crossing_point] ==
        static_cast<triangulator_index_t>(points.size()));
}

TEST_CASE("CDT refuses a crossing when constraints must be preserved",
          "[constrained_delaunay]") {
  // Two constraints crossing transversally. Without permission to split,
  // the build must refuse so the caller's recovery path can run.
  auto points =
      make_triangulator_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  CHECK_FALSE(cdt.build(points.points(), tf::make_edges(edge_data),
                        tf::make_constant_range(true, 2), false));
}

TEST_CASE("CDT resolves a crossing when splitting is allowed",
          "[constrained_delaunay]") {
  auto points =
      make_triangulator_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  // The crossing became a vertex, and is reported so the caller can
  // broadcast it to the other carriers of those edges.
  CHECK(cdt.points().size() > points.size());
  triangulator_index_t reported = 0;
  cdt.for_each_constraint_crossing(
      [&](triangulator_index_t point, const auto &parents) {
        ++reported;
        CHECK(parents.size() >= 1);
        CHECK(point < static_cast<triangulator_index_t>(cdt.points().size()));
      });
  CHECK(reported >= 1);

  for (const auto &record : cdt.parameterized_crossings()) {
    CHECK(record.id_a != record.id_b);
    CHECK(record.t_a > Cdt::param_t(0));
  }
}

TEST_CASE("CDT splits a constraint at a vertex lying on it",
          "[constrained_delaunay]") {
  // Point 2 sits exactly on the segment (0, 1). This is a T-junction, not
  // a crossing: it must be resolved on the existing vertex, and reported
  // with a SINGLE parent.
  auto points =
      make_triangulator_points<Int>({{0, 0}, {10, 0}, {5, 0}, {5, 6}, {2, 6}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  const triangulator_index_t a = cdt.index_map().f()[0];
  const triangulator_index_t b = cdt.index_map().f()[1];
  const triangulator_index_t mid = cdt.index_map().f()[2];

  // The whole constraint must NOT survive; its two halves must.
  CHECK_FALSE(has_edge(cdt, a, b));
  CHECK(has_edge(cdt, a, mid));
  CHECK(has_edge(cdt, mid, b));

  cdt.for_each_constraint_crossing(
      [](triangulator_index_t, const auto &parents) {
        CHECK(parents.size() == 1);
      });
}

TEST_CASE("CDT keeps points() and kept_ids() parallel through resolution",
          "[constrained_delaunay]") {
  // Consumers index kept_ids() BY OUTPUT POINT ID. A resolution point has
  // no input of its own and must carry the "not an input vertex"
  // sentinel; letting the two buffers diverge is an out-of-bounds read.
  auto points =
      make_triangulator_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  const auto &kept = cdt.index_map().kept_ids();
  REQUIRE(kept.size() == cdt.points().size());
  const triangulator_index_t n_input =
      static_cast<triangulator_index_t>(cdt.index_map().f().size());
  for (auto id : kept)
    CHECK(id <= n_input);

  for (const auto &record : cdt.parameterized_crossings())
    CHECK(kept[std::size_t(record.point)] == n_input);
}

TEST_CASE("CDT places a crossing near the end of a long constraint",
          "[constrained_delaunay]") {
  // The parameter is the geometry, so it must be carried at the exact
  // substrate's width. Too few bits quantise this crossing to the far
  // endpoint and put the derived point millions of lattice units away.
  // The placement must stay within a unit of the true crossing, (1, 0).
  // The long constraint must be the one inserted SECOND: the resolution
  // point is placed by blending along the constraint currently being
  // recovered, so that is where a coarse parameter shows up.
  const Wide remote = Wide(1) << 50;
  auto points =
      make_triangulator_points<Wide>({{0, 0}, {remote, 0}, {1, -4}, {1, 4}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{0, 1}};

  WideCdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  bool found = false;
  for (std::size_t q = 0; q < cdt.points().size(); ++q) {
    const auto p = cdt.points()[q];
    if (p[1] == Wide(0) && p[0] > Wide(0) && p[0] <= Wide(2))
      found = true;
  }
  CHECK(found);
  for (const auto &record : cdt.parameterized_crossings()) {
    const auto p = cdt.points()[std::size_t(record.point)];
    CHECK(p[0] <= Wide(2));
  }
}

TEST_CASE("CDT leaves a non-crossing constraint set untouched",
          "[constrained_delaunay]") {
  // The common path: nothing crosses, so resolution must never fire and
  // both build modes must agree.
  auto points = make_triangulator_points<Int>({{0, 0}, {8, 0}, {8, 8}, {0, 8}});
  std::array<std::array<triangulator_index_t, 2>, 4> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{1, 2},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 0}};

  Cdt preserved;
  REQUIRE(preserved.build(points.points(), tf::make_edges(edge_data),
                          tf::make_constant_range(true, 4), false));
  CHECK(preserved.parameterized_crossings().size() == 0);

  Cdt split;
  REQUIRE(split.build(points.points(), tf::make_edges(edge_data),
                      tf::make_constant_range(true, 4), true));
  CHECK(split.parameterized_crossings().size() == 0);
  CHECK(split.make_faces().size() == preserved.make_faces().size());
}

TEST_CASE("CDT resolved crossings reach the emitted triangulation",
          "[constrained_delaunay]") {
  // The two diagonals of a square cross, so recovering them creates a vertex
  // that no input carried. Everything the build reports about itself has to
  // account for that vertex: a triangle touching it is a triangle of the
  // result, and a label is owed for each.
  auto points = make_triangulator_points<Int>({{0, 0}, {4, 0}, {4, 4}, {0, 4}});
  std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 2},
      std::array<triangulator_index_t, 2>{1, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  CHECK(cdt.points().size() > points.size());
  const auto faces = cdt.make_faces();
  CHECK(faces.size() == cdt.n_triangles());
  CHECK(faces.size() != 0);
  CHECK(cdt.region_labels().size() ==
        static_cast<decltype(cdt.region_labels().size())>(faces.size()));
}

TEST_CASE("CDT ids the interior of a single loop", "[constrained_delaunay]") {
  auto points = make_triangulator_points<Int>({{0, 0}, {8, 0}, {8, 8}, {0, 8}});
  std::array<std::array<triangulator_index_t, 2>, 4> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{1, 2},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 0}};

  Cdt cdt;
  const auto parity =
      label_both_modes(cdt, points.points(), tf::make_edges(edge_data),
                       tf::make_constant_range(true, 4));

  const auto faces = cdt.make_faces();
  const auto ids = cdt.region_labels();
  REQUIRE(faces.size() != 0);
  REQUIRE(ids.size() == static_cast<decltype(ids.size())>(faces.size()));
  REQUIRE(parity.size() == std::size_t(faces.size()));

  // The hull IS the loop, so the exterior owns no triangle at all -- and is
  // still region 0, because ids address regions, not triangles.
  CHECK(n_labels(ids) == 2);
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    CHECK(ids[t] == 1);
    CHECK(parity[t] == 1);
  }
}

TEST_CASE("CDT ids a loop apart from the hole inside it",
          "[constrained_delaunay]") {
  // Parity cannot express nesting: the hole and the outside are both 0. The
  // ids are what tells them apart, so they must differ here.
  auto points = make_triangulator_points<Int>(
      {{0, 0}, {12, 0}, {12, 12}, {0, 12}, {4, 4}, {8, 4}, {8, 8}, {4, 8}});
  std::array<std::array<triangulator_index_t, 2>, 8> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{1, 2},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 0},
      std::array<triangulator_index_t, 2>{4, 5},
      std::array<triangulator_index_t, 2>{5, 6},
      std::array<triangulator_index_t, 2>{6, 7},
      std::array<triangulator_index_t, 2>{7, 4}};

  Cdt cdt;
  const auto parity =
      label_both_modes(cdt, points.points(), tf::make_edges(edge_data),
                       tf::make_constant_range(true, 8));

  const auto faces = cdt.make_faces();
  const auto out_points = cdt.points();
  const auto ids = cdt.region_labels();
  CHECK(n_labels(ids) == 3);

  triangulator_index_t ring_id = triangulator_index_t(-1);
  triangulator_index_t hole_id = triangulator_index_t(-1);
  triangulator_index_t n_ring = 0;
  triangulator_index_t n_hole = 0;
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    if (centroid_inside(corner_sum(out_points, faces[t]), 4, 8)) {
      ++n_hole;
      if (hole_id == triangulator_index_t(-1))
        hole_id = ids[t];
      CHECK(ids[t] == hole_id);
      CHECK(parity[t] == 0);
    } else {
      ++n_ring;
      if (ring_id == triangulator_index_t(-1))
        ring_id = ids[t];
      CHECK(ids[t] == ring_id);
      CHECK(parity[t] == 1);
    }
  }
  REQUIRE(n_hole != 0);
  REQUIRE(n_ring != 0);
  // The hull is the outer loop, so neither region is the exterior.
  CHECK(ring_id != 0);
  CHECK(hole_id != 0);
  CHECK(ring_id != hole_id);
}

TEST_CASE("CDT ids the three regions of two overlapping loops",
          "[constrained_delaunay]") {
  // Two quads meeting at a corner. Their overlap carries the outside's
  // parity, so only the ids separate A-only, the overlap and B-only. The
  // hull adds a triangle outside both loops at each end of the overlap, and
  // they are the exterior itself -- reached through two DIFFERENT hull
  // edges, which is why the walk enters at every one of them.
  auto points = make_triangulator_points<Int>(
      {{0, 0}, {12, 0}, {12, 12}, {0, 12}, {6, 6}, {18, 6}, {18, 18}, {6, 18}});
  std::array<std::array<triangulator_index_t, 2>, 8> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{1, 2},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 0},
      std::array<triangulator_index_t, 2>{4, 5},
      std::array<triangulator_index_t, 2>{5, 6},
      std::array<triangulator_index_t, 2>{6, 7},
      std::array<triangulator_index_t, 2>{7, 4}};

  Cdt cdt;
  const auto parity =
      label_both_modes(cdt, points.points(), tf::make_edges(edge_data),
                       tf::make_constant_range(true, 8));

  const auto faces = cdt.make_faces();
  const auto out_points = cdt.points();
  const auto ids = cdt.region_labels();
  CHECK(n_labels(ids) == 4);

  // 0: outside both loops, 1: A only, 2: the overlap, 3: B only.
  std::array<triangulator_index_t, 4> id_of{
      triangulator_index_t(-1), triangulator_index_t(-1),
      triangulator_index_t(-1), triangulator_index_t(-1)};
  std::array<triangulator_index_t, 4> parity_of{
      triangulator_index_t(-1), triangulator_index_t(-1),
      triangulator_index_t(-1), triangulator_index_t(-1)};
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    const auto sum = corner_sum(out_points, faces[t]);
    const bool in_a = centroid_inside(sum, 0, 12);
    const bool in_b = centroid_inside(sum, 6, 18);
    const std::size_t region = in_a ? (in_b ? 2u : 1u) : (in_b ? 3u : 0u);
    if (id_of[region] == triangulator_index_t(-1)) {
      id_of[region] = ids[t];
      parity_of[region] = parity[t];
    }
    CHECK(ids[t] == id_of[region]);
    CHECK(parity[t] == parity_of[region]);
  }
  for (std::size_t region = 0; region < 4; ++region)
    REQUIRE(id_of[region] != triangulator_index_t(-1));

  CHECK(id_of[0] == 0);
  CHECK(id_of[1] != 0);
  CHECK(id_of[2] != 0);
  CHECK(id_of[3] != 0);
  CHECK(id_of[1] != id_of[2]);
  CHECK(id_of[2] != id_of[3]);
  CHECK(id_of[1] != id_of[3]);

  CHECK(parity_of[0] == 0);
  CHECK(parity_of[1] == 1);
  CHECK(parity_of[2] == 0);
  CHECK(parity_of[3] == 1);
}

TEST_CASE("CDT keeps a slit inside one region", "[constrained_delaunay]") {
  // The slit crosses the loop from side to side and is preserved, but it is
  // not a wall: it must re-enter the region it left, so one id and one
  // parity cover both of its sides.
  auto points = make_triangulator_points<Int>(
      {{0, 0}, {10, 0}, {10, 10}, {0, 10}, {0, 5}, {10, 5}});
  std::array<std::array<triangulator_index_t, 2>, 5> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{1, 2},
      std::array<triangulator_index_t, 2>{2, 3},
      std::array<triangulator_index_t, 2>{3, 0},
      std::array<triangulator_index_t, 2>{4, 5}};
  std::array<bool, 5> is_boundary{true, true, true, true, false};

  Cdt cdt;
  const auto parity =
      label_both_modes(cdt, points.points(), tf::make_edges(edge_data),
                       tf::make_range(is_boundary));

  const auto faces = cdt.make_faces();
  const auto out_points = cdt.points();
  const auto ids = cdt.region_labels();
  CHECK(n_labels(ids) == 2);

  triangulator_index_t n_below = 0;
  triangulator_index_t n_above = 0;
  for (std::size_t t = 0; t < std::size_t(faces.size()); ++t) {
    CHECK(ids[t] == 1);
    CHECK(parity[t] == 1);
    if (corner_sum(out_points, faces[t])[1] < 15)
      ++n_below;
    else
      ++n_above;
  }
  // Both sides of the slit are triangulated, so the single id is a fact
  // about the slit rather than about an empty half.
  REQUIRE(n_below != 0);
  REQUIRE(n_above != 0);

  // The slit is still there: preserved, and still constrained.
  CHECK(has_constrained_edge(cdt, cdt.index_map().f()[4],
                             cdt.index_map().f()[5]));
}

TEST_CASE("a crossing whose point already exists is still stated on both "
          "constraints",
          "[topology][constrained_delaunay_triangulator]") {
  // These two constraints cross, but the lattice point the crossing rounds
  // onto is already a vertex, so recovery has no point left to create. The
  // meeting is still a meeting: each constraint carries it at its own exact
  // parameter, and the existing vertex is only the coordinate it borrows.
  //
  // It cannot be stated as a landing instead, because a landing asserts that
  // the vertex lies ON the constraint and this one does not lie on both.
  // Naming a single side leaves the other's incidence unstated, and a caller
  // that re-feeds the same constraints then rediscovers the same crossing
  // without end — the meeting is what terminates the round.
  const auto points = make_triangulator_points<triangulator_index_t>(
      {{-3, 2}, {2, -2}, {-4, 1}, {3, 4}, {-2, 1}, {2, -3}});
  const std::array<std::array<triangulator_index_t, 2>, 2> edge_data{
      std::array<triangulator_index_t, 2>{0, 1},
      std::array<triangulator_index_t, 2>{2, 3}};

  Cdt cdt;
  cdt.always_track_constraint_owners();
  REQUIRE(cdt.build_triangulation(points.points(), tf::make_edges(edge_data),
                                  true));

  // The crossing created no point: it landed on one that was already there.
  CHECK(cdt.points().size() == points.size());
  REQUIRE(cdt.parameterized_crossings().size() == 1);

  const auto &record = cdt.parameterized_crossings()[0];
  const auto whole = decltype(record.t_a)(1) << Cdt::crossing_param_bits();
  CHECK(record.point < static_cast<triangulator_index_t>(cdt.points().size()));

  // Both constraints are named, each placed by its own parameter, and both
  // parameters are strictly interior: the meeting is on each of them.
  CHECK(record.id_a != triangulator_index_t(-1));
  CHECK(record.id_b != triangulator_index_t(-1));
  CHECK(record.id_a != record.id_b);
  CHECK(record.t_a > decltype(record.t_a)(0));
  CHECK(record.t_a < whole);
  CHECK(record.t_b > decltype(record.t_b)(0));
  CHECK(record.t_b < whole);
}

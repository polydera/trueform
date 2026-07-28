#include <catch2/catch_test_macros.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <array>
#include <initializer_list>

namespace {

using Index = int;
using Int = tf::exact::int32;
using Wide = tf::exact::int64;
using Cdt = tf::constrained_delaunay_triangulator<Index, Int, Int>;
using WideCdt = tf::constrained_delaunay_triangulator<Index, Wide, Wide>;

template <typename Coordinate>
auto make_points(std::initializer_list<std::array<Coordinate, 2>> values)
    -> tf::points_buffer<Coordinate, 2> {
  tf::points_buffer<Coordinate, 2> points;
  for (const auto &value : values)
    points.push_back(tf::point<Coordinate, 2>{value[0], value[1]});
  return points;
}

/// True iff some triangle of the build carries the edge (a, b).
template <typename Triangulator>
auto has_edge(Triangulator &cdt, Index a, Index b) -> bool {
  for (auto face : cdt.make_faces())
    for (int c = 0; c < 3; ++c) {
      const Index u = face[std::size_t(c)];
      const Index v = face[std::size_t((c + 1) % 3)];
      if ((u == a && v == b) || (u == b && v == a))
        return true;
    }
  return false;
}

} // namespace

TEST_CASE("CDT crossings use welded output point IDs",
          "[constrained_delaunay]") {
  auto points = make_points<Int>({{0, 0}, {4, 0}, {5, 1}, {2, 0}, {5, -1}});
  std::array<std::array<Index, 2>, 3> edge_data{std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3},
                                                std::array<Index, 2>{3, 4}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  const Index junction = cdt.index_map().f()[3];
  CHECK(cdt.index_map().kept_ids()[junction] == 3);

  Index callback_count = 0;
  cdt.for_each_constraint_crossing([&](Index point, const auto &constraints) {
    ++callback_count;
    CHECK(point == junction);
    REQUIRE(constraints.size() == 1);
    CHECK(constraints[0] == 0);
  });
  CHECK(callback_count == 1);
}

TEST_CASE("CDT crossings retain synthetic point constraint incidences",
          "[constrained_delaunay]") {
  auto points = make_points<Int>({{-2, 0}, {2, 0}, {0, -2}, {0, 2}});
  std::array<std::array<Index, 2>, 3> edge_data{std::array<Index, 2>{0, 0},
                                                std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  Index crossing_point = Index(-1);
  Index reported = 0;
  cdt.for_each_constraint_crossing([&](Index point, const auto &constraints) {
    ++reported;
    crossing_point = point;
    // Both constraints the point is interior to; the report states
    // which, not in what order.
    REQUIRE(constraints.size() == 2);
    const Index lo =
        constraints[0] < constraints[1] ? constraints[0] : constraints[1];
    const Index hi =
        constraints[0] < constraints[1] ? constraints[1] : constraints[0];
    CHECK(lo == 1);
    CHECK(hi == 2);
  });
  CHECK(reported == 1);
  REQUIRE(crossing_point != Index(-1));
  CHECK(cdt.index_map().kept_ids()[crossing_point] ==
        static_cast<Index>(points.size()));
}

TEST_CASE("CDT refuses a crossing when constraints must be preserved",
          "[constrained_delaunay]") {
  // Two constraints crossing transversally. Without permission to split,
  // the build must refuse so the caller's recovery path can run.
  auto points = make_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3}};

  Cdt cdt;
  CHECK_FALSE(cdt.build(points.points(), tf::make_edges(edge_data),
                        tf::make_constant_range(true, 2), false));
}

TEST_CASE("CDT resolves a crossing when splitting is allowed",
          "[constrained_delaunay]") {
  auto points = make_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  // The crossing became a vertex, and is reported so the caller can
  // broadcast it to the other carriers of those edges.
  CHECK(cdt.points().size() > points.size());
  Index reported = 0;
  cdt.for_each_constraint_crossing([&](Index point, const auto &parents) {
    ++reported;
    CHECK(parents.size() >= 1);
    CHECK(point < static_cast<Index>(cdt.points().size()));
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
  auto points = make_points<Int>({{0, 0}, {10, 0}, {5, 0}, {5, 6}, {2, 6}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  const Index a = cdt.index_map().f()[0];
  const Index b = cdt.index_map().f()[1];
  const Index mid = cdt.index_map().f()[2];

  // The whole constraint must NOT survive; its two halves must.
  CHECK_FALSE(has_edge(cdt, a, b));
  CHECK(has_edge(cdt, a, mid));
  CHECK(has_edge(cdt, mid, b));

  cdt.for_each_constraint_crossing(
      [](Index, const auto &parents) { CHECK(parents.size() == 1); });
}

TEST_CASE("CDT keeps points() and kept_ids() parallel through resolution",
          "[constrained_delaunay]") {
  // Consumers index kept_ids() BY OUTPUT POINT ID. A resolution point has
  // no input of its own and must carry the "not an input vertex"
  // sentinel; letting the two buffers diverge is an out-of-bounds read.
  auto points = make_points<Int>({{0, 0}, {10, 10}, {0, 10}, {10, 0}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{0, 1},
                                                std::array<Index, 2>{2, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data),
                    tf::make_constant_range(true, 2), true));

  const auto &kept = cdt.index_map().kept_ids();
  REQUIRE(kept.size() == cdt.points().size());
  const Index n_input = static_cast<Index>(cdt.index_map().f().size());
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
  const Wide far = Wide(1) << 50;
  auto points = make_points<Wide>({{0, 0}, {far, 0}, {1, -4}, {1, 4}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{2, 3},
                                                std::array<Index, 2>{0, 1}};

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
  auto points = make_points<Int>({{0, 0}, {8, 0}, {8, 8}, {0, 8}});
  std::array<std::array<Index, 2>, 4> edge_data{
      std::array<Index, 2>{0, 1}, std::array<Index, 2>{1, 2},
      std::array<Index, 2>{2, 3}, std::array<Index, 2>{3, 0}};

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
  auto points = make_points<Int>({{0, 0}, {4, 0}, {4, 4}, {0, 4}});
  std::array<std::array<Index, 2>, 2> edge_data{std::array<Index, 2>{0, 2},
                                                std::array<Index, 2>{1, 3}};

  Cdt cdt;
  REQUIRE(cdt.build(points.points(), tf::make_edges(edge_data)));

  CHECK(cdt.points().size() > points.size());
  const auto faces = cdt.make_faces();
  CHECK(faces.size() == cdt.n_triangles());
  CHECK(faces.size() != 0);
  CHECK(cdt.region_labels().size() ==
        static_cast<decltype(cdt.region_labels().size())>(faces.size()));
}

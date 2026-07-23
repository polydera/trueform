#include <catch2/catch_test_macros.hpp>
#include <trueform/core/point.hpp>
#include <trueform/intersect/intersections_within_segments.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_edge_membership.hpp>

#include <array>

TEST_CASE("Duplicated segment vertex hits retain local endpoint IDs",
          "[intersect][segments]") {
  using Index = int;
  using Point = tf::point<int, 2>;
  using Edge = std::array<Index, 2>;

  std::array<Point, 5> points{
      Point{0, 0}, Point{4, 0}, Point{5, 1}, Point{2, 0}, Point{5, -1}};
  std::array<Edge, 3> edge_data{Edge{0, 1}, Edge{2, 3}, Edge{3, 4}};

  auto edges = tf::make_edges(tf::make_range(edge_data));
  auto segments = tf::make_segments(edges, tf::make_range(points));
  auto membership = tf::make_edge_membership(segments);
  tf::aabb_tree<Index, int, 2> tree;
  tree.build(segments, tf::config_tree(4, 4));
  auto tagged = segments | tf::tag(membership) | tf::tag(tree);

  tf::intersections_within_segments<Index, int, 2> intersections;
  intersections.build(tagged);

  auto records = intersections.intersections();
  REQUIRE(records.size() == 3);
  REQUIRE(records[1].size() == 1);
  REQUIRE(records[2].size() == 1);

  const auto &incoming = records[1][0];
  CHECK(incoming.object == 1);
  CHECK(incoming.target.label == tf::topo_type::vertex);
  CHECK(incoming.target.id == 1);

  const auto &outgoing = records[2][0];
  CHECK(outgoing.object == 2);
  CHECK(outgoing.target.label == tf::topo_type::vertex);
  CHECK(outgoing.target.id == 0);
}

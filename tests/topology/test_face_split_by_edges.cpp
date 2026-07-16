/**
 * @file test_face_split_by_edges.cpp
 * @brief Tests for tf::face_split_by_edges with exact arithmetic
 *
 * Verifies face subdivision: crossings, nested crossings, loops,
 * cuts, non-crossings, and combinations. All geometry uses int32
 * exact arithmetic.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/blocked_buffer.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/topology/face_split_by_edges.hpp>

using Index = int;

namespace {

auto make_pts(const std::vector<std::array<int32_t, 2>> &data)
    -> tf::points_buffer<int32_t, 2> {
  tf::points_buffer<int32_t, 2> pts;
  pts.allocate(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    pts[i][0] = data[i][0];
    pts[i][1] = data[i][1];
  }
  return pts;
}

auto make_undirected_edges(const std::vector<std::array<int, 2>> &edges)
    -> tf::blocked_buffer<int, 2> {
  tf::blocked_buffer<int, 2> buf;
  buf.allocate(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    buf[i][0] = edges[i][0];
    buf[i][1] = edges[i][1];
  }
  return buf;
}

auto face_verts(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> std::vector<int> {
  std::vector<int> v;
  auto it = fsbe.faces().begin();
  std::advance(it, idx);
  for (auto vid : *it)
    v.push_back(vid);
  return v;
}

auto hole_verts(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> std::vector<int> {
  std::vector<int> v;
  auto it = fsbe.holes().begin();
  std::advance(it, idx);
  for (auto vid : *it)
    v.push_back(vid);
  return v;
}

auto face_area(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> long long {
  auto it = fsbe.face_areas().begin();
  std::advance(it, idx);
  return static_cast<long long>(*it);
}

auto hole_area(const tf::face_split_by_edges<Index> &fsbe, std::size_t idx)
    -> long long {
  auto it = fsbe.hole_areas().begin();
  std::advance(it, idx);
  return static_cast<long long>(*it);
}

auto total_area(const tf::face_split_by_edges<Index> &fsbe) -> long long {
  long long sum = 0;
  for (auto a : fsbe.face_areas())
    sum += static_cast<long long>(a);
  for (auto a : fsbe.hole_areas())
    sum += static_cast<long long>(a);
  return sum;
}

auto edge_in_output(const tf::face_split_by_edges<Index> &fsbe, int a, int b)
    -> bool {
  auto scan = [&](auto &&loops) {
    for (auto &&loop : loops) {
      auto n = loop.size();
      for (std::size_t i = 0, p = n - 1; i < n; p = i++)
        if ((loop[p] == a && loop[i] == b) || (loop[p] == b && loop[i] == a))
          return true;
    }
    return false;
  };
  return scan(fsbe.faces()) || scan(fsbe.holes());
}

auto hole_parent(const tf::face_split_by_edges<Index> &fsbe, int hole_id)
    -> int {
  int face_id = 0;
  for (auto &&holes : fsbe.holes_for_faces()) {
    for (auto h : holes)
      if (h == hole_id)
        return face_id;
    ++face_id;
  }
  return -1;
}

} // namespace

TEST_CASE("Simple crossing", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_area(fsbe, 0) == 40000);
  CHECK(face_area(fsbe, 1) == 40000);
}

TEST_CASE("Nested crossings (different start/end)",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 50},   {50, 0},   {150, 0},
                       {200, 50}, {150, 100}, {50, 100},
                       {75, 50},  {125, 50}});
  auto edges = make_undirected_edges({{0, 6}, {6, 5}, {1, 7}, {7, 4}});
  std::vector<int> face = {0, 1, 2, 3, 4, 5};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 6, 5});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 7, 4, 5, 6});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{1, 2, 3, 4, 7});
  CHECK(face_area(fsbe, 0) == 3750);
  CHECK(face_area(fsbe, 1) == 13750);
  CHECK(face_area(fsbe, 2) == 12500);
}

TEST_CASE("Same end crossings", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {60, 100}, {140, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {1, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 5, 2, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{1, 2, 5});
  CHECK(face_area(fsbe, 0) == 32000);
  CHECK(face_area(fsbe, 1) == 36000);
  CHECK(face_area(fsbe, 2) == 12000);
}

TEST_CASE("Same start crossings", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {60, 100}, {140, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 3}, {0, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 5, 2, 3, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{0, 1, 2, 5});
  CHECK(face_area(fsbe, 0) == 12000);
  CHECK(face_area(fsbe, 1) == 36000);
  CHECK(face_area(fsbe, 2) == 32000);
}

TEST_CASE("Same start AND end (area ordering)", "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 40}, {100, 160}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {0, 5}, {5, 2}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 5, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 4, 2, 5});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_area(fsbe, 0) == 28000);
  CHECK(face_area(fsbe, 1) == 24000);
  CHECK(face_area(fsbe, 2) == 28000);
}

TEST_CASE("Interior loop (not touching base)", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {70, 70},
                       {130, 70},
                       {130, 130},
                       {70, 130}});
  auto edges = make_undirected_edges({{4, 5}, {5, 6}, {6, 7}, {7, 4}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{5, 6, 7, 4});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{4, 7, 6, 5});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(face_area(fsbe, 1) == 7200);
  CHECK(hole_area(fsbe, 0) == -7200);

  // hole 0 assigned to face 0
  auto hff = fsbe.holes_for_faces();
  auto it = hff.begin();
  CHECK((*it).size() == 1);
  CHECK((*it)[0] == 0);
}

TEST_CASE("Loop sharing vertex with base (start==end)",
          "[face_split_by_edges]") {
  auto pts = make_pts(
      {{0, 0}, {200, 0}, {200, 200}, {0, 200}, {40, 40}, {80, 40}, {60, 80}});
  auto edges = make_undirected_edges({{0, 4}, {4, 5}, {5, 6}, {6, 0}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 2);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{4, 5, 6, 0});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 6, 5, 4});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(face_area(fsbe, 1) == 2400);
  CHECK(hole_area(fsbe, 0) == -2400);
}

TEST_CASE("Crossing + loop sharing start vertex",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {30, 150},
                       {50, 10},
                       {80, 10},
                       {65, 30}});
  auto edges = make_undirected_edges(
      {{0, 4}, {4, 2}, {0, 5}, {5, 6}, {6, 7}, {7, 0}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 2, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{0, 1, 2, 4});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{5, 6, 7, 0});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 7, 6, 5});
  CHECK(face_area(fsbe, 0) == 16000);
  CHECK(face_area(fsbe, 1) == 64000);
  CHECK(face_area(fsbe, 2) == 1450);
  CHECK(hole_area(fsbe, 0) == -1450);

  // hole assigned to face 1 (the larger triangle containing the loop)
  auto hff = fsbe.holes_for_faces();
  auto f0 = hff.begin();
  auto f1 = f0;
  ++f1;
  CHECK((*f0).size() == 0);
  CHECK((*f1).size() == 1);
  CHECK((*f1)[0] == 0);
}

TEST_CASE("Simple cut", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 1);
  CHECK(fsbe.holes().size() == 1);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 1, 2, 3});
  CHECK(hole_verts(fsbe, 0) == std::vector<int>{0, 4});
  CHECK(face_area(fsbe, 0) == 80000);
  CHECK(hole_area(fsbe, 0) == 0);
}

// A disconnected triangle with one antenna: the cycle is plucked as a
// loop path (hole + reversed face pair) and the antenna alone reaches
// planar_graph_regions as an out-and-back zero-area walk — it must come
// out as a slit hole (the cut-path shape) so its edge stays a constraint.
TEST_CASE("Island loop with antenna slit", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {70, 70},
                       {130, 70},
                       {100, 130},
                       {100, 160}});
  auto edges = make_undirected_edges({{4, 5}, {5, 6}, {6, 4}, {6, 7}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  REQUIRE(fsbe.faces().size() == 2);
  REQUIRE(fsbe.holes().size() == 2);
  CHECK(total_area(fsbe) == 80000);
  int base_face = face_area(fsbe, 0) == 80000 ? 0 : 1;
  int loop_hole = -1, slit_hole = -1;
  for (std::size_t i = 0; i < fsbe.holes().size(); ++i) {
    if (hole_area(fsbe, i) == -3600)
      loop_hole = int(i);
    if (hole_area(fsbe, i) == 0)
      slit_hole = int(i);
  }
  REQUIRE(loop_hole != -1);
  REQUIRE(slit_hole != -1);
  CHECK(hole_parent(fsbe, loop_hole) == base_face);
  CHECK(hole_parent(fsbe, slit_hole) == base_face);
  for (auto &&[a, b] :
       std::vector<std::array<int, 2>>{{4, 5}, {5, 6}, {6, 4}, {6, 7}})
    CHECK(edge_in_output(fsbe, a, b));
}

// A cycle whose corners have ODD degree cannot be plucked as a loop path:
// it reaches planar_graph_regions as open paths that reform the cycle
// there (the face-25 shape). Its outer walk is a negative region and must
// come out as a hole in the surrounding face — emitting it as a face
// would duplicate the cycle's cell with inverted winding (the fold-back).
TEST_CASE("Odd-degree cycle: negative outer walk becomes hole",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {70, 70},
                       {130, 70},
                       {100, 130},
                       {50, 50},
                       {160, 50}});
  auto edges = make_undirected_edges(
      {{4, 5}, {5, 6}, {6, 4}, {4, 7}, {5, 8}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  REQUIRE(fsbe.faces().size() == 2);
  REQUIRE(fsbe.holes().size() == 1);
  CHECK(total_area(fsbe) == 80000);
  // triangle interior emitted exactly once, positive
  bool has_triangle = false;
  for (std::size_t i = 0; i < fsbe.faces().size(); ++i)
    has_triangle |= face_area(fsbe, i) == 3600;
  CHECK(has_triangle);
  // the cycle's outer walk (antenna slits included) is the hole
  CHECK(hole_area(fsbe, 0) == -3600);
  int base_face = face_area(fsbe, 0) == 80000 ? 0 : 1;
  CHECK(hole_parent(fsbe, 0) == base_face);
  for (auto &&[a, b] : std::vector<std::array<int, 2>>{
           {4, 5}, {5, 6}, {6, 4}, {4, 7}, {5, 8}})
    CHECK(edge_in_output(fsbe, a, b));
}

// Island with a lake: square cycle containing a triangle cycle, joined by
// a chord, all disconnected from the base loop. The cycles come out as
// loop paths (hole + reversed face pairs); the chord alone reaches
// planar_graph_regions, where its out-and-back walk is a zero-area
// region — it must be emitted as a slit hole (cut-path shape) nested in
// the ring, or the constraint edge vanishes from the triangulation.
TEST_CASE("Nested cycles (island with lake)", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {40, 40},
                       {160, 40},
                       {160, 160},
                       {40, 160},
                       {80, 80},
                       {120, 80},
                       {100, 120}});
  auto edges = make_undirected_edges(
      {{4, 5}, {5, 6}, {6, 7}, {7, 4}, {8, 9}, {9, 10}, {10, 8}, {4, 8}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  REQUIRE(fsbe.faces().size() == 3);
  REQUIRE(fsbe.holes().size() == 3);
  CHECK(total_area(fsbe) == 80000);
  int base_face = -1, square_face = -1, triangle_face = -1;
  for (std::size_t i = 0; i < fsbe.faces().size(); ++i) {
    if (face_area(fsbe, i) == 80000)
      base_face = int(i);
    if (face_area(fsbe, i) == 28800)
      square_face = int(i);
    if (face_area(fsbe, i) == 1600)
      triangle_face = int(i);
  }
  REQUIRE(base_face != -1);
  REQUIRE(square_face != -1);
  REQUIRE(triangle_face != -1);
  int square_hole = -1, triangle_hole = -1, slit_hole = -1;
  for (std::size_t i = 0; i < fsbe.holes().size(); ++i) {
    if (hole_area(fsbe, i) == -28800)
      square_hole = int(i);
    if (hole_area(fsbe, i) == -1600)
      triangle_hole = int(i);
    if (hole_area(fsbe, i) == 0)
      slit_hole = int(i);
  }
  REQUIRE(square_hole != -1);
  REQUIRE(triangle_hole != -1);
  REQUIRE(slit_hole != -1);
  // square punches the base; triangle and the chord slit live in the ring
  CHECK(hole_parent(fsbe, square_hole) == base_face);
  CHECK(hole_parent(fsbe, triangle_hole) == square_face);
  CHECK(hole_parent(fsbe, slit_hole) == square_face);
  // the chord survives as a constraint via the slit hole
  CHECK(edge_in_output(fsbe, 4, 8));
}

// The same cycle bridged to the base loop through crossing chords: the
// outer side of the cycle is absorbed into the surrounding walks, so no
// negative region exists and nothing may be emitted as a hole.
TEST_CASE("Cycle bridged to base loop", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {80, 80},
                       {120, 80},
                       {100, 120},
                       {100, 160}});
  auto edges = make_undirected_edges(
      {{0, 4}, {4, 5}, {5, 6}, {6, 4}, {6, 7}, {5, 1}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.holes().size() == 0);
  CHECK(total_area(fsbe) == 80000);
  bool has_triangle = false;
  for (std::size_t i = 0; i < fsbe.faces().size(); ++i)
    has_triangle |= face_area(fsbe, i) == 1600;
  CHECK(has_triangle);
  for (auto &&[a, b] : std::vector<std::array<int, 2>>{
           {0, 4}, {4, 5}, {5, 6}, {6, 4}, {6, 7}, {5, 1}})
    CHECK(edge_in_output(fsbe, a, b));
}

// Two chains between the same base vertices through coincident interior
// points (ids 4 and 5 share coordinates — the sub-ulp twin shape; the
// real pipeline dedups these upstream, so this is a robustness contract,
// not a shape the arrangement feeds us). The polar sort ties on the
// coincident directions and the tie-break decides the wiring; whatever it
// picks, the invariants must hold: area conserved, no negative or
// inverted faces, every input edge present as a constraint.
TEST_CASE("Zero-area sandwich between twin chains",
          "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0},
                       {200, 0},
                       {200, 200},
                       {0, 200},
                       {100, 100},
                       {100, 100},
                       {60, 120}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {0, 5}, {5, 2}, {4, 6}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(total_area(fsbe) == 80000);
  for (auto a : fsbe.face_areas())
    CHECK(static_cast<long long>(a) > 0);
  for (auto a : fsbe.hole_areas())
    CHECK(static_cast<long long>(a) <= 0);
  // both chains survive as constraints in the emitted loops
  CHECK(edge_in_output(fsbe, 0, 4));
  CHECK(edge_in_output(fsbe, 4, 2));
  CHECK(edge_in_output(fsbe, 0, 5));
  CHECK(edge_in_output(fsbe, 5, 2));
  CHECK(edge_in_output(fsbe, 4, 6));
}

TEST_CASE("Non-crossings (T-junction)", "[face_split_by_edges]") {
  auto pts = make_pts({{0, 0}, {200, 0}, {200, 200}, {0, 200}, {100, 100}});
  auto edges = make_undirected_edges({{0, 4}, {4, 2}, {4, 3}});
  std::vector<int> face = {0, 1, 2, 3};

  tf::face_split_by_edges<Index> fsbe;
  fsbe.build(tf::make_range(face), tf::make_edges(edges), pts.points());

  CHECK(fsbe.faces().size() == 3);
  CHECK(fsbe.holes().size() == 0);
  CHECK(face_verts(fsbe, 0) == std::vector<int>{0, 4, 3});
  CHECK(face_verts(fsbe, 1) == std::vector<int>{4, 0, 1, 2});
  CHECK(face_verts(fsbe, 2) == std::vector<int>{4, 2, 3});
  CHECK(face_area(fsbe, 0) == 20000);
  CHECK(face_area(fsbe, 1) == 40000);
  CHECK(face_area(fsbe, 2) == 20000);
}

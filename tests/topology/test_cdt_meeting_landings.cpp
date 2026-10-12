/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/edges.hpp>
#include <trueform/core/points_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/orient2d.hpp>
#include <trueform/topology/constrained_delaunay_triangulator.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

namespace {

template <typename Int> struct landing_scene {
  std::vector<std::array<Int, 2>> points;
  std::vector<std::array<int, 2>> edges;
  std::vector<char> walls;
};

template <typename Int>
using landing_cdt = tf::constrained_delaunay_triangulator<int, Int, Int>;

template <typename Int>
auto build_landing_scene(landing_cdt<Int> &cdt, const landing_scene<Int> &scene,
                         bool split) -> bool {
  tf::points_buffer<Int, 2> points;
  for (const auto &p : scene.points)
    points.push_back(tf::point<Int, 2>{p[0], p[1]});
  std::vector<int> flat;
  for (const auto &edge : scene.edges) {
    flat.push_back(edge[0]);
    flat.push_back(edge[1]);
  }
  cdt.always_track_constraint_owners();
  return cdt.build(
      points.points(),
      tf::make_edges(tf::make_blocked_range<2>(tf::make_range(flat))),
      tf::make_range(scene.walls), split);
}

/// The build is a triangulation that holds its constraints: every face
/// bounds area with one winding, no edge carries three faces, and every
/// input constraint is a chain of constrained edges whose vertices stand
/// within two lattice units of it.
template <typename Int>
auto landing_triangulation_holds(const landing_cdt<Int> &cdt,
                                 const landing_scene<Int> &scene) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto points = cdt.points();
  const auto at = [&](int i) {
    return tf::point<Int, 2>{Int(points[std::size_t(i)][0]),
                             Int(points[std::size_t(i)][1])};
  };
  int winding = 0;
  std::map<std::pair<int, int>, int> faces_of_edge;
  for (const auto face : cdt.make_faces()) {
    const T2 area = tf::exact::orient2d<Int>(at(face[0]), at(face[1]),
                                             at(face[2]));
    const int sign = area > T2(0) ? 1 : (area < T2(0) ? -1 : 0);
    if (sign == 0 || (winding != 0 && sign != winding))
      return false;
    winding = sign;
    for (int c = 0; c < 3; ++c) {
      const int u = face[c], v = face[(c + 1) % 3];
      if (++faces_of_edge[{u < v ? u : v, u < v ? v : u}] > 2)
        return false;
    }
  }
  std::vector<std::vector<int>> chain(std::size_t(points.size()));
  cdt.for_each_face_adjacency([&](int, int v0, int v1, int v2, int,
                                  const std::array<int, 3> &,
                                  const std::array<bool, 3> &constrained) {
    const std::array<int, 3> corners{v0, v1, v2};
    for (int c = 0; c < 3; ++c)
      if (constrained[std::size_t(c)]) {
        chain[std::size_t(corners[std::size_t(c)])].push_back(
            corners[std::size_t((c + 1) % 3)]);
        chain[std::size_t(corners[std::size_t((c + 1) % 3)])].push_back(
            corners[std::size_t(c)]);
      }
  });
  const auto input = cdt.index_map().f();
  for (const auto &edge : scene.edges) {
    const int from = input[std::size_t(edge[0])];
    const int to = input[std::size_t(edge[1])];
    const double length =
        std::sqrt(double(tf::exact::distance2<Int>(at(from), at(to))));
    std::vector<char> seen(std::size_t(points.size()), 0);
    std::vector<int> stack{from};
    seen[std::size_t(from)] = 1;
    bool reached = false;
    while (!stack.empty() && !reached) {
      const int vertex = stack.back();
      stack.pop_back();
      reached = vertex == to;
      for (const int next : chain[std::size_t(vertex)])
        if (!seen[std::size_t(next)] &&
            std::fabs(double(tf::exact::orient2d<Int>(at(from), at(to),
                                                      at(next)))) <=
                2.0 * length) {
          seen[std::size_t(next)] = 1;
          stack.push_back(next);
        }
    }
    if (!reached)
      return false;
  }
  return true;
}

/// Landings on created points, each checked to name a point some meeting
/// of two constraints minted: one identity the landing references.
template <typename Int>
auto created_landings_of(const landing_cdt<Int> &cdt) -> int {
  const auto &records = cdt.parameterized_crossings();
  const int n_input = int(cdt.index_map().f().size());
  int landings = 0;
  for (const auto &landing : records) {
    if (landing.id_b != -1 ||
        int(cdt.index_map().kept_ids()[std::size_t(landing.point)]) != n_input)
      continue;
    bool named = false;
    for (const auto &meeting : records)
      named = named || (meeting.id_b != -1 && meeting.point == landing.point);
    if (!named)
      return -1;
    ++landings;
  }
  return landings;
}

// The client plane's two refusing carriers: vertices a few units apart at
// 1e17 make two constraints nearly overlap, and a third enters between them
// at a shallow angle, so each rounded crossing lands across the other.
auto near_overlap_five() -> landing_scene<tf::exact::int64> {
  return {{{-545020530454653381LL, 353835787474720801LL},
           {-539031207207925604LL, 361370513388240941LL},
           {-536192548368805975LL, 364696470682410215LL},
           {-545020530454653374LL, 353835787474720810LL},
           {-536335934011831235LL, 364528470738659695LL}},
          {{0, 1}, {0, 3}, {1, 3}, {1, 4}, {2, 3}, {2, 4}},
          {1, 1, 0, 1, 1, 1}};
}

auto near_overlap_six() -> landing_scene<tf::exact::int64> {
  return {{{-617216197846625193LL, 206370718622390587LL},
           {-622023084454420931LL, 181079011068896586LL},
           {-545020530454653381LL, 353835787474720801LL},
           {-619818111922605406LL, 192680599578956832LL},
           {-545020530454653381LL, 353835787474720800LL},
           {-622023084454420931LL, 181079011068896587LL}},
          {{0, 3}, {0, 4}, {1, 3}, {1, 5}, {2, 4}, {2, 5}, {3, 5}},
          {1, 1, 1, 1, 1, 1, 0}};
}

// Three constraints from a cluster to one far point, and a crosser entering
// the bundle toward its middle point.
auto near_overlap_bundle32() -> landing_scene<tf::exact::int32> {
  return {{{-500000000, -500000000},
           {-499999999, -499999975},
           {-499999994, -499999935},
           {-463321687, 306922824},
           {-466161986, 145318260}},
          {{0, 3}, {1, 3}, {2, 3}, {4, 1}, {0, 1}, {1, 2}},
          {1, 1, 0, 1, 1, 1}};
}

auto near_overlap_bundle64() -> landing_scene<tf::exact::int64> {
  return {{{-200000000000000000LL, -200000000000000000LL},
           {-199999999999999999LL, -199999999999999975LL},
           {-199999999999999994LL, -199999999999999935LL},
           {-185328675927897405LL, 122769129586257120LL},
           {-186464794826619424LL, 58127293849003272LL}},
          {{0, 3}, {1, 3}, {2, 3}, {4, 1}, {0, 1}, {1, 2}},
          {1, 1, 0, 1, 1, 1}};
}

// Near-parallel constraints leaving a cluster of near-coincident vertices,
// and a crosser through the cluster.
auto near_overlap_cluster32() -> landing_scene<tf::exact::int32> {
  return {{{-3, 1},
           {124073372, 380270696},
           {-4, 0},
           {125524401, 379794183},
           {0, 1},
           {124229505, 380219710},
           {-112317063, -383907381},
           {101085356, 345516643}},
          {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
          {1, 1, 1, 1}};
}

auto near_overlap_cluster64() -> landing_scene<tf::exact::int64> {
  return {{{-3, 1},
           {49629347397425748LL, 152108276819197540LL},
           {-4, 0},
           {50209761564643045LL, 151917674559682781LL},
           {0, 1},
           {49691802894021126LL, 152087884872996222LL},
           {-44926825049995952LL, -153562952533894240LL},
           {40434142544996360LL, 138206657280504816LL}},
          {{0, 1}, {2, 3}, {4, 5}, {6, 7}},
          {1, 1, 1, 1}};
}

/// splitmix64: the battery's own generator, so its scenes are the same bytes
/// on every standard library.
struct landing_battery_rng {
  std::uint64_t state;
  auto next() -> std::uint64_t {
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
};

struct landing_battery_digest {
  std::uint64_t value = 1469598103934665603ull;
  template <typename T> auto add(const T &x) -> void {
    unsigned char bytes[sizeof(T)];
    std::memcpy(bytes, &x, sizeof(T));
    for (const auto byte : bytes) {
      value ^= byte;
      value *= 1099511628211ull;
    }
  }
};

/// Ordinary builds across the whole lattice: distinct points, distinct
/// constraints with distinct ends, crossings resolved. Returns the digest of
/// every product and counts the meetings and the landings on created points.
template <typename Int>
auto landing_battery(int n_builds, std::uint64_t seed, int &meetings,
                     int &created_landings) -> std::uint64_t {
  const std::uint64_t span =
      sizeof(Int) == 4 ? 2000000000ull : 800000000000000000ull;
  landing_battery_rng rng{seed};
  landing_battery_digest digest;
  meetings = 0;
  created_landings = 0;
  for (int build = 0; build < n_builds; ++build) {
    const int n_points = 8 + int(rng.next() % 24);
    tf::points_buffer<Int, 2> points;
    for (int i = 0; i < n_points; ++i)
      points.push_back(tf::point<Int, 2>{
          Int(std::int64_t(rng.next() % span) - std::int64_t(span / 2)),
          Int(std::int64_t(rng.next() % span) - std::int64_t(span / 2))});
    std::vector<int> flat;
    std::vector<char> walls;
    const int n_edges = 3 + int(rng.next() % 12);
    for (int e = 0; e < n_edges; ++e) {
      const int a = int(rng.next() % std::uint64_t(n_points));
      const int b = int(rng.next() % std::uint64_t(n_points));
      bool fresh = a != b;
      for (std::size_t k = 0; fresh && k < flat.size(); k += 2)
        fresh = !((flat[k] == a && flat[k + 1] == b) ||
                  (flat[k] == b && flat[k + 1] == a));
      if (!fresh)
        continue;
      flat.push_back(a);
      flat.push_back(b);
      walls.push_back(char(rng.next() & 1));
    }
    landing_cdt<Int> cdt;
    cdt.always_track_constraint_owners();
    const bool built = cdt.build(
        points.points(),
        tf::make_edges(tf::make_blocked_range<2>(tf::make_range(flat))),
        tf::make_range(walls), true);
    digest.add(built);
    if (!built)
      continue;
    for (std::size_t i = 0; i < std::size_t(cdt.points().size()); ++i) {
      digest.add(Int(cdt.points()[i][0]));
      digest.add(Int(cdt.points()[i][1]));
    }
    for (const auto face : cdt.make_faces())
      for (int c = 0; c < 3; ++c)
        digest.add(int(face[c]));
    for (const auto label : cdt.region_labels())
      digest.add(int(label));
    const int n_input = int(cdt.index_map().f().size());
    for (const auto &record : cdt.parameterized_crossings()) {
      digest.add(int(record.point));
      digest.add(int(record.id_a));
      digest.add(int(record.id_b));
      digest.add(record.t_a);
      digest.add(record.t_b);
      if (record.id_b != -1)
        ++meetings;
      if (record.id_b == -1 &&
          int(cdt.index_map().kept_ids()[std::size_t(record.point)]) == n_input)
        ++created_landings;
    }
    for (const auto &collision : cdt.constraint_collisions()) {
      digest.add(int(collision.prior_input));
      digest.add(int(collision.input));
    }
  }
  return digest.value;
}

} // namespace

TEST_CASE("a crossing rounding across a near-coincident constraint lands on "
          "it",
          "[topology][constrained_delaunay_triangulator]") {
  // Each rounded crossing of these near-overlapping constraints lands across
  // the other one; re-derived as a fresh crossing it alternates between them
  // until the resolution depth refuses the build. Landed, the meeting is a
  // point of both, and the build closes.
  for (const auto &scene : {near_overlap_five(), near_overlap_six()}) {
    landing_cdt<tf::exact::int64> preserved;
    CHECK_FALSE(build_landing_scene(preserved, scene, false));

    landing_cdt<tf::exact::int64> cdt;
    REQUIRE(build_landing_scene(cdt, scene, true));
    CHECK(landing_triangulation_holds(cdt, scene));
    CHECK(created_landings_of(cdt) > 0);
  }
}

TEST_CASE("a meeting rounding across two constraints of a bundle lands on "
          "both",
          "[topology][constrained_delaunay_triangulator]") {
  // The crosser's rounded meeting with one constraint of the bundle lands
  // across the other two at once. Left whole, they keep a face the meeting
  // flattens.
  {
    const auto scene = near_overlap_bundle32();
    landing_cdt<tf::exact::int32> cdt;
    REQUIRE(build_landing_scene(cdt, scene, true));
    CHECK(landing_triangulation_holds(cdt, scene));
    CHECK(created_landings_of(cdt) > 0);
  }
  {
    const auto scene = near_overlap_bundle64();
    landing_cdt<tf::exact::int64> cdt;
    REQUIRE(build_landing_scene(cdt, scene, true));
    CHECK(landing_triangulation_holds(cdt, scene));
    CHECK(created_landings_of(cdt) > 0);
  }
}

TEST_CASE("a crosser through a cluster of near-coincident vertices converges",
          "[topology][constrained_delaunay_triangulator]") {
  // A meeting rounds exactly onto a constraint leaving the cluster. Released
  // before the meeting enters, that constraint's face is flipped away instead
  // of flattened, and the constraint is recovered through the meeting.
  {
    const auto scene = near_overlap_cluster32();
    landing_cdt<tf::exact::int32> cdt;
    REQUIRE(build_landing_scene(cdt, scene, true));
    CHECK(landing_triangulation_holds(cdt, scene));
    CHECK(created_landings_of(cdt) > 0);
  }
  {
    const auto scene = near_overlap_cluster64();
    landing_cdt<tf::exact::int64> cdt;
    REQUIRE(build_landing_scene(cdt, scene, true));
    CHECK(landing_triangulation_holds(cdt, scene));
    CHECK(created_landings_of(cdt) > 0);
  }
}

TEST_CASE("a build whose roundings reach no other constraint lands nothing",
          "[topology][constrained_delaunay_triangulator]") {
  // Ordinary builds across the lattice resolve many crossings, none of whose
  // roundings reaches a third constraint. Their products are pinned: the
  // digests are the ones the kernel produced before meetings could land.
  int meetings = 0, landings = 0;
  CHECK(landing_battery<tf::exact::int32>(256, 32, meetings, landings) ==
        0xb00b04ce31e836b9ull);
  CHECK(meetings == 1487);
  CHECK(landings == 0);
  CHECK(landing_battery<tf::exact::int64>(256, 64, meetings, landings) ==
        0xada2e2ab7fcca014ull);
  CHECK(meetings == 1331);
  CHECK(landings == 0);
}

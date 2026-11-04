/*
 * Copyright (c) 2026 XLAB
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
#include <trueform/core/index_map.hpp>
#include <trueform/core/none.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/tetras.hpp>
#include <trueform/core/views/blocked_range.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/take.hpp>
#include <trueform/exact/int128.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/topology/cdt/delaunay_execution_policy.hpp>
#include <trueform/topology/cdt/dt3/append_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/build_parallel_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/compact_tet_dead_rows.hpp>
#include <trueform/topology/cdt/dt3/compact_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/protect_tet_facets.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_segments.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_point.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_recovery.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_status.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_site_on_segment.hpp>
#include <trueform/topology/cdt/dt3/tet_aligned_position.hpp>
#include <trueform/topology/cdt/dt3/tet_claim_workspace.hpp>
#include <trueform/topology/cdt/dt3/tet_facets.hpp>
#include <trueform/topology/cdt/dt3/tet_owner_complex.hpp>
#include <trueform/topology/cdt/dt3/tet_position.hpp>
#include <trueform/topology/cdt/dt3/tet_site_position.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <tbb/task_arena.h>
#include <vector>

namespace {

using seg_i32 = tf::exact::int32;
using seg_coord = std::int64_t;
using seg_owner_t =
    tf::topology::cdt::dt3::tetrahedralization_owner<int, seg_coord, seg_i32>;
using seg_dt =
    tf::topology::cdt::dt3::tet_owner_complex<false, seg_owner_t, tf::none_t>;
tf::none_t seg_serial{};
/// The complex of `flat`, built serially into `owner`.
auto seg_build(seg_owner_t &owner, const std::vector<seg_coord> &flat) -> bool {
  return tf::topology::cdt::dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat)));
}
/// Recover `segments` on `dt` and publish its complex, as the segment phase
/// does after its recovery.
template <typename DT, typename Segments, typename Recovery>
auto seg_recover(DT &dt, const Segments &segments, int depth,
                 Recovery &recovery) -> bool {
  tf::topology::cdt::dt3::compact_tet_dead_rows<true>(dt.owner());
  const bool appended = tf::topology::cdt::dt3::recover_tet_segments(
      dt, segments, depth, recovery);
  tf::topology::cdt::dt3::compact_tetrahedralization<true, true>(dt.owner());
  return appended;
}
using seg_recovery =
    tf::topology::cdt::dt3::tet_segment_recovery<int, seg_coord>;
using seg_status = tf::topology::cdt::dt3::tet_segment_status;

/// A cloud on the int32 lattice, stored scaled by 2^scale.
auto seg_cloud(std::uint32_t seed, std::size_t n, int span, int scale)
    -> std::vector<seg_coord> {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> axis(-span, span);
  std::vector<seg_coord> flat;
  for (std::size_t i = 0; i < 3 * n; ++i)
    flat.push_back(seg_coord(axis(rng)) * (seg_coord(1) << scale));
  return flat;
}

template <typename DT> auto seg_edges(const DT &dt) {
  std::vector<std::array<int, 2>> edges;
  const auto cells = dt.tets();
  for (std::size_t row = 0; row < dt.n_tets(); ++row) {
    if (!dt.finite(int(row)))
      continue;
    const auto cell = cells[row];
    for (std::size_t i = 0; i < 4; ++i)
      for (std::size_t j = i + 1; j < 4; ++j)
        edges.push_back({std::min(int(cell[i]), int(cell[j])),
                         std::max(int(cell[i]), int(cell[j]))});
  }
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  return edges;
}

/// A recovered segment's chain read on its own terms: it runs from the
/// segment's first end to its last through sites strictly ordered on the
/// segment, every link an edge of the complex, every numerator on the grid
/// naming its own site's point.
template <typename DT, typename Segment>
auto seg_chain_holds(const DT &dt, const seg_recovery &recovery, std::size_t s,
                     const Segment &segment, int depth) -> bool {
  namespace dt3 = tf::topology::cdt::dt3;
  const auto chain = recovery.chains[s];
  const auto &owner = dt.owner();
  const auto edges = seg_edges(dt);
  if (chain.size() < 2 || chain[0].site != segment[0] ||
      chain[chain.size() - 1].site != segment[1])
    return false;
  const auto &a = dt3::tet_shallow_point(owner, segment[0]);
  const auto &b = dt3::tet_shallow_point(owner, segment[1]);
  std::size_t axis = 0;
  for (std::size_t k = 1; k < 3; ++k)
    if (std::abs(b[k] - a[k]) > std::abs(b[axis] - a[axis]))
      axis = k;
  const seg_coord direction = b[axis] > a[axis] ? 1 : -1;
  for (std::size_t i = 0; i < chain.size(); ++i) {
    const auto &p = dt3::tet_shallow_point(owner, chain[i].site);
    if (i && i + 1 < chain.size() &&
        !dt3::tet_site_on_segment<seg_i32>(a, b, p))
      return false;
    if (chain[i].numerator >= 0 &&
        !(dt3::tet_minted_position<seg_coord>(
              dt3::tet_segment_point<tf::exact::meta<seg_i32>::T2>(
                  a, b, chain[i].numerator, depth),
              0, depth) == dt3::tet_minted_position<seg_coord>(p)))
      return false;
    if (i + 1 == chain.size())
      continue;
    const auto &q = dt3::tet_shallow_point(owner, chain[i + 1].site);
    if ((q[axis] - p[axis]) * direction <= 0)
      return false;
    const std::array<int, 2> link{
        std::min(int(chain[i].site), int(chain[i + 1].site)),
        std::max(int(chain[i].site), int(chain[i + 1].site))};
    if (!std::binary_search(edges.begin(), edges.end(), link))
      return false;
  }
  return true;
}

} // namespace

TEST_CASE("tet segments: segments through a cloud recover as chains of edges",
          "[topology][delaunay3][segments]") {
  const int depth = 12;
  const auto flat = seg_cloud(9001, 400, 1 << 16, depth);
  seg_owner_t owner;
  REQUIRE(seg_build(owner, flat));
  seg_dt dt{owner, seg_serial};
  std::mt19937 rng(9002);
  std::vector<std::array<int, 2>> segments;
  for (int i = 0; i < 24; ++i) {
    const int a = int(rng() % dt.n_sites());
    int b = int(rng() % dt.n_sites());
    if (b == a)
      b = (b + 1) % int(dt.n_sites());
    segments.push_back({a, b});
  }
  const auto n_sites = dt.n_sites();
  seg_recovery recovery;
  REQUIRE(seg_recover(dt, segments, depth, recovery));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  CHECK(dt.n_sites() > n_sites);
  std::size_t recovered = 0;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    CAPTURE(s, int(recovery.status[s]));
    if (recovery.status[s] != seg_status::recovered)
      continue;
    ++recovered;
    CHECK(seg_chain_holds(dt, recovery, s, segments[s], depth));
  }
  CHECK(recovered == segments.size());
}

TEST_CASE("tet segments: a segment through a standing site splits there",
          "[topology][delaunay3][segments]") {
  const int depth = 8;
  for (int numerator : {1, 2}) {
    CAPTURE(numerator);
    std::vector<seg_coord> flat{0, 0, 0, 4 << depth, 0, 0};
    const seg_coord on = seg_coord(numerator) << depth;
    flat.insert(flat.end(), {on, 0, 0});
    const auto cloud =
        seg_cloud(9100 + std::uint32_t(numerator), 40, 32, depth);
    flat.insert(flat.end(), cloud.begin(), cloud.end());
    seg_owner_t owner;
    REQUIRE(seg_build(owner, flat));
    seg_dt dt{owner, seg_serial};
    const auto f = dt.index_map().f();
    const std::vector<std::array<int, 2>> segments{{f[0], f[1]}};
    const auto n_sites = dt.n_sites();
    seg_recovery recovery;
    REQUIRE(seg_recover(dt, segments, depth, recovery));
    REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
    REQUIRE(recovery.status[0] == seg_status::recovered);
    CHECK(seg_chain_holds(dt, recovery, 0, segments[0], depth));
    const auto chain = recovery.chains[0];
    bool shared = false;
    for (const auto &vertex : chain)
      shared = shared || vertex.site == f[2];
    CHECK(shared);
    if (numerator == 2)
      for (const auto &vertex : chain)
        if (vertex.site == f[2])
          CHECK(vertex.numerator == std::int64_t(1) << (depth - 1));
    CHECK(dt.n_sites() >= n_sites);
  }
}

namespace {

/// Long segments between far-apart sites of a 65k cloud, which a fixed
/// shallow depth cannot recover.
template <typename DT>
auto seg_long_segments(const DT &dt, std::uint32_t seed, std::size_t count)
    -> std::vector<std::array<int, 2>> {
  std::mt19937 rng(seed);
  std::vector<std::array<int, 2>> segments;
  const auto &owner = dt.owner();
  while (segments.size() < count) {
    const int a = int(rng() % dt.n_sites()), b = int(rng() % dt.n_sites());
    double d2 = 0;
    for (std::size_t k = 0; k < 3; ++k) {
      const double d =
          double(tf::topology::cdt::dt3::tet_shallow_point(owner, a)[k] -
                 tf::topology::cdt::dt3::tet_shallow_point(owner, b)[k]);
      d2 += d * d;
    }
    if (d2 > double(seg_coord(1) << 30) * double(seg_coord(1) << 30) / 4)
      segments.push_back({a, b});
  }
  return segments;
}

template <typename DT>
auto seg_stream(const DT &dt, const seg_recovery &recovery) {
  std::vector<std::int64_t> stream;
  const auto cells = dt.tets();
  for (std::size_t row = 0; row < dt.n_tets(); ++row) {
    if (!dt.finite(int(row)))
      continue;
    for (std::size_t k = 0; k < 4; ++k)
      stream.push_back(cells[row][k]);
  }
  const auto &owner = dt.owner();
  for (int site = 0; site < int(dt.n_sites()); ++site) {
    stream.push_back(tf::topology::cdt::dt3::tet_site_name(owner, site));
    for (std::size_t k = 0; k < 3; ++k)
      stream.push_back(
          tf::topology::cdt::dt3::tet_shallow_point(owner, site)[k]);
  }
  for (std::size_t s = 0; s < recovery.status.size(); ++s) {
    stream.push_back(std::int64_t(recovery.status[s]));
    stream.push_back(recovery.rounds[s]);
    for (const auto &vertex : recovery.chains[s]) {
      stream.push_back(vertex.site);
      stream.push_back(vertex.numerator);
    }
  }
  return stream;
}

} // namespace

TEST_CASE("tet segments: a 65k cloud refuses at a shallow depth and keeps "
          "its complex",
          "[topology][delaunay3][segments]") {
  const int scale = 12;
  const auto flat = seg_cloud(9200, 65536, 1 << 18, scale);
  seg_owner_t shallow_owner;
  REQUIRE(seg_build(shallow_owner, flat));
  seg_dt shallow{shallow_owner, seg_serial};
  const auto segments = seg_long_segments(shallow, 9201, 32);
  seg_recovery recovery;
  REQUIRE(seg_recover(shallow, segments, 4, recovery));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(
      shallow_owner));
  std::size_t refused = 0;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    CHECK((recovery.status[s] == seg_status::recovered ||
           recovery.status[s] == seg_status::depth));
    refused += recovery.status[s] == seg_status::depth;
    if (recovery.status[s] == seg_status::recovered)
      CHECK(seg_chain_holds(shallow, recovery, s, segments[s], 4));
  }
  CHECK(refused > 0);

  seg_owner_t deep_owner;
  REQUIRE(seg_build(deep_owner, flat));
  seg_dt deep{deep_owner, seg_serial};
  REQUIRE(seg_recover(deep, segments, scale, recovery));
  REQUIRE(
      tf::topology::cdt::dt3::check_tetrahedralization_validity(deep_owner));
  for (std::size_t s = 0; s < segments.size(); ++s) {
    CAPTURE(s);
    REQUIRE(recovery.status[s] == seg_status::recovered);
    CHECK(seg_chain_holds(deep, recovery, s, segments[s], scale));
  }
}

TEST_CASE("tet segments: a segment whose frame runs out deepens it and "
          "recovers",
          "[topology][delaunay3][segments][depth]") {
  // A segment 2^21 lattice units long passes through 512 sites within 16 of
  // its middle, stored at 2^10: its pieces there are finer than a 10-bit
  // frame splits, so on that frame it refuses for depth. Deepening, its
  // frame reaches past the sites' own, its splits stand exactly on it below
  // the frame, and it recovers as a chain of edges.
  namespace dt3 = tf::topology::cdt::dt3;
  const int depth = 10;
  const seg_coord reach = seg_coord(1) << 20;
  std::mt19937 rng(1);
  std::uniform_int_distribution<int> spread(-int(reach), int(reach)),
      cluster(-16, 16);
  std::vector<seg_coord> flat{-reach, 0, 0, reach, 0, 0};
  for (int i = 0; i < 32; ++i)
    flat.insert(flat.end(), {spread(rng), spread(rng), spread(rng)});
  for (int i = 0; i < 512; ++i)
    flat.insert(flat.end(), {cluster(rng), cluster(rng), cluster(rng)});
  for (auto &coordinate : flat)
    coordinate *= seg_coord(1) << depth;
  for (const bool deepens : {false, true}) {
    CAPTURE(deepens);
    seg_owner_t owner;
    REQUIRE(seg_build(owner, flat));
    seg_dt dt{owner, seg_serial};
    const auto f = dt.index_map().f();
    const std::vector<std::array<int, 2>> segments{{f[0], f[1]}};
    seg_recovery recovery;
    recovery.deepens = deepens;
    REQUIRE(seg_recover(dt, segments, depth, recovery));
    REQUIRE(dt3::check_tetrahedralization_validity(owner));
    int deepest = 0;
    for (std::size_t site = 0; site < dt3::tet_site_count(owner); ++site)
      deepest = std::max(deepest, dt3::tet_site_depth(owner, int(site)));
    if (!deepens) {
      CHECK(recovery.status[0] == seg_status::depth);
      CHECK(deepest == 0);
      continue;
    }
    REQUIRE(recovery.status[0] == seg_status::recovered);
    CHECK(recovery.bits[0] > depth);
    CHECK(deepest > 0);
    using wide_t = tf::exact::meta<tf::exact::int64>::T2;
    const auto at = [&owner, deepest](int site) {
      return dt3::tet_aligned_position<seg_coord>(
          dt3::tet_site_position(owner, site), deepest);
    };
    const auto chain = recovery.chains[0];
    REQUIRE(chain.size() > 2);
    CHECK(chain[0].site == segments[0][0]);
    CHECK(chain[chain.size() - 1].site == segments[0][1]);
    const auto a = at(segments[0][0]), b = at(segments[0][1]);
    const auto edges = seg_edges(dt);
    for (std::size_t i = 0; i < chain.size(); ++i) {
      const auto p = at(chain[i].site);
      for (std::size_t k = 0; k < 3; ++k) {
        const std::size_t u = (k + 1) % 3, v = (k + 2) % 3;
        CHECK(wide_t(p[u] - a[u]) * wide_t(b[v] - a[v]) ==
              wide_t(p[v] - a[v]) * wide_t(b[u] - a[u]));
      }
      if (i + 1 == chain.size())
        continue;
      CHECK(at(chain[i + 1].site)[0] > p[0]);
      const std::array<int, 2> link{
          std::min(int(chain[i].site), int(chain[i + 1].site)),
          std::max(int(chain[i].site), int(chain[i + 1].site))};
      CHECK(std::binary_search(edges.begin(), edges.end(), link));
    }
  }
}

TEST_CASE("tet segments: recovery on a parallel-built source is one stream "
          "at any worker count",
          "[topology][delaunay3][segments][determinism]") {
  const int scale = 12;
  const auto flat = seg_cloud(9300, 16384, 1 << 18, scale);
  const auto points = tf::make_points<3>(tf::make_range(flat));
  std::vector<std::int64_t> expected;
  std::size_t minted = 0;
  for (int workers : {1, 4, 8}) {
    CAPTURE(workers);
    tbb::task_arena arena(workers);
    seg_owner_t owner;
    seg_dt dt{owner, seg_serial};
    seg_recovery recovery;
    std::vector<std::array<int, 2>> segments;
    REQUIRE(arena.execute([&] {
      tf::topology::cdt::dt3::tet_claim_workspace<int> work{};
      if (!tf::topology::cdt::dt3::build_parallel_tetrahedralization(
              owner, points, work))
        return false;
      segments = seg_long_segments(dt, 9301, 48);
      const auto n = dt.n_sites();
      const bool done = seg_recover(dt, segments, scale, recovery);
      minted = dt.n_sites() - n;
      return done;
    }));
    CHECK((owner._stats.parallel_bands > 0) == (workers >= 4));
    CHECK(minted > 0);
    for (std::size_t s = 0; s < segments.size(); ++s)
      CHECK(recovery.status[s] == seg_status::recovered);
    const auto stream = seg_stream(dt, recovery);
    if (expected.empty())
      expected = stream;
    CHECK(stream == expected);
  }
}



TEST_CASE("tet segments: wide lattice sites recover, their splits at exact "
          "midpoints",
          "[topology][delaunay3][segments]") {
  using wide_owner_t = tf::topology::cdt::dt3::tetrahedralization_owner<
      int, std::int64_t, std::int64_t>;
  using wide_dt =
      tf::topology::cdt::dt3::tet_owner_complex<false, wide_owner_t, tf::none_t>;
  const std::int64_t reach = std::int64_t(1) << 50;
  const std::vector<std::int64_t> flat{
      -reach, 0, 0,     reach,        0, 0,      0,          0, 1024, 0, 0,
      -1024,  0, reach, reach / 1024, 0, -reach, reach / 512};
  wide_owner_t owner;
  REQUIRE(tf::topology::cdt::dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat))));
  wide_dt dt{owner, seg_serial};
  const auto f = dt.index_map().f();
  const std::vector<std::array<int, 2>> segments{{f[0], f[1]}};
  tf::topology::cdt::dt3::tet_segment_recovery<int, std::int64_t> recovery;
  REQUIRE(
      seg_recover(dt, segments, 4, recovery));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  REQUIRE(recovery.status[0] == seg_status::recovered);
  const auto chain = recovery.chains[0];
  CHECK(chain.size() >= 2);
  namespace dt3 = tf::topology::cdt::dt3;
  const auto split = [&owner, &f](std::int64_t numerator) {
    return dt3::tet_minted_position<std::int64_t>(
        dt3::tet_segment_point<tf::exact::int128>(
            dt3::tet_shallow_point(owner, f[0]),
            dt3::tet_shallow_point(owner, f[1]), numerator, 4),
        0, 4);
  };
  for (const auto &vertex : chain)
    if (vertex.numerator > 0 && vertex.numerator < 16)
      CHECK(split(vertex.numerator) ==
            dt3::tet_site_position(owner, vertex.site));
  // the frame's own arithmetic on sites past 32 bits: the exact midpoint
  CHECK(split(8) == dt3::tet_minted_position<std::int64_t>(
                        tf::exact::pt3<std::int64_t>{0, 0, 0}));
}

TEST_CASE("tet segments: two segments whose splits land on one point share "
          "it",
          "[topology][delaunay3][segments]") {
  const int depth = 8;
  const seg_coord reach = seg_coord(64) << depth;
  std::vector<seg_coord> flat{-reach, 0,     0, reach, 0, 0, 0, -reach, 0,
                              0,      reach, 0, 0,     0, 1, 0, 0,      -1};
  const auto cloud = seg_cloud(9400, 24, 256, depth);
  flat.insert(flat.end(), cloud.begin(), cloud.end());
  seg_owner_t owner;
  REQUIRE(seg_build(owner, flat));
  seg_dt dt{owner, seg_serial};
  const auto f = dt.index_map().f();
  const std::vector<std::array<int, 2>> segments{{f[0], f[1]}, {f[2], f[3]}};
  const auto n_sites = dt.n_sites();
  seg_recovery recovery;
  REQUIRE(seg_recover(dt, segments, depth, recovery));
  REQUIRE(tf::topology::cdt::dt3::check_tetrahedralization_validity(owner));
  CHECK(recovery.status[0] == seg_status::recovered);
  CHECK(recovery.status[1] == seg_status::recovered);
  const auto shared = [&recovery](std::size_t s, int site) {
    for (const auto &vertex : recovery.chains[s])
      if (vertex.site == site)
        return true;
    return false;
  };
  int common = -1;
  for (const auto &vertex : recovery.chains[0])
    if (std::size_t(vertex.site) >= n_sites && shared(1, vertex.site) &&
        tf::topology::cdt::dt3::tet_shallow_point(owner, vertex.site) ==
            tf::exact::pt3<seg_coord>{0, 0, 0})
      common = vertex.site;
  CHECK(common >= 0);
  std::size_t at_origin = 0;
  for (int site = 0; site < int(dt.n_sites()); ++site)
    at_origin += tf::topology::cdt::dt3::tet_shallow_point(owner, site) ==
                 tf::exact::pt3<seg_coord>{0, 0, 0};
  CHECK(at_origin == 1);
}

namespace {

/// The on-segment verdict read without the filter: the cross product in the
/// lattice's own wide type, then betweenness along the widest axis.
template <typename Int, typename Coord>
auto seg_on_segment_exact(const tf::exact::pt3<Coord> &a,
                          const tf::exact::pt3<Coord> &b,
                          const tf::exact::pt3<Coord> &p) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  T2 u[3], v[3];
  for (std::size_t k = 0; k < 3; ++k) {
    u[k] = T2(b[k]) - T2(a[k]);
    v[k] = T2(p[k]) - T2(a[k]);
  }
  if (u[1] * v[2] != u[2] * v[1] || u[2] * v[0] != u[0] * v[2] ||
      u[0] * v[1] != u[1] * v[0])
    return false;
  T2 along = 0, length = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    along += u[k] * v[k];
    length += u[k] * u[k];
  }
  return T2(0) < along && along < length;
}

template <typename Int, typename Coord>
auto seg_check_on_segment(std::uint32_t seed, Coord span) -> void {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<long long> axis(-(long long)(span),
                                                (long long)(span));
  const auto point = [&] {
    return tf::exact::pt3<Coord>{Coord(axis(rng)), Coord(axis(rng)),
                                 Coord(axis(rng))};
  };
  std::size_t on = 0;
  for (int trial = 0; trial < 4000; ++trial) {
    auto a = point(), b = point(), p = point();
    if (b == a)
      b[0] += 1;
    if (trial % 3 == 1) {
      const Coord k = Coord(trial % 5) - 1;
      for (std::size_t c = 0; c < 3; ++c)
        p[c] = a[c] + (b[c] - a[c]) * k;
    } else if (trial % 3 == 2) {
      for (std::size_t c = 0; c < 3; ++c) {
        p[c] = a[c] + b[c];
        a[c] *= 2;
        b[c] *= 2;
      }
    }
    const bool stated =
        tf::topology::cdt::dt3::tet_site_on_segment<Int>(a, b, p);
    CHECK(stated == seg_on_segment_exact<Int>(a, b, p));
    on += stated;
  }
  CHECK(on > 1000);
}

} // namespace

TEST_CASE("tet segments: the filtered on-segment test agrees with its exact "
          "reading",
          "[topology][delaunay3][segments][filter]") {
  seg_check_on_segment<seg_i32, seg_coord>(9500, seg_coord(1) << 40);
  seg_check_on_segment<seg_i32, seg_coord>(9501, seg_coord(1) << 12);
  using wide = tf::exact::meta<tf::exact::int64>::T1;
  seg_check_on_segment<tf::exact::int64, wide>(9502, wide(1) << 60);
}


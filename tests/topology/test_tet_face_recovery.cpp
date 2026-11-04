/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://www.boost.org/LICENSE_1_0.txt
 *
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <trueform/core/none.hpp>
#include <trueform/core/offset_block_buffer.hpp>
#include <trueform/core/points.hpp>
#include <trueform/core/range.hpp>
#include <trueform/core/views/offset_block_range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/orient3d.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/topology/cdt/dt3/build_tetrahedralization.hpp>
#include <trueform/topology/cdt/dt3/check_tetrahedralization_validity.hpp>
#include <trueform/topology/cdt/dt3/load_tetrahedralization_sites.hpp>
#include <trueform/topology/cdt/dt3/recovery/check_tet_face_fixpoint.hpp>
#include <trueform/topology/cdt/dt3/recovery/check_tet_pieces_stand.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_face_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/recover_tet_segment_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_face_recovery.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_face_status.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_pierce_in_facet.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_plane_lattice_point.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_plane_normal.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_point_in_facet.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_rim_piece.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_phase.hpp>
#include <trueform/topology/cdt/dt3/recovery/tet_segment_status.hpp>
#include <trueform/topology/cdt/dt3/tet_aligned_position.hpp>
#include <trueform/topology/cdt/dt3/tet_decision_sites.hpp>
#include <trueform/topology/cdt/dt3/tet_facet_is_protected.hpp>
#include <trueform/topology/cdt/dt3/tet_facet_key.hpp>
#include <trueform/topology/cdt/dt3/tet_facets.hpp>
#include <trueform/topology/cdt/dt3/tet_flip_constraints.hpp>
#include <trueform/topology/cdt/dt3/tet_flip_eligible.hpp>
#include <trueform/topology/cdt/dt3/tet_shared_owners.hpp>
#include <trueform/topology/cdt/dt3/tet_site_position.hpp>
#include <trueform/topology/cdt/dt3/tet_state.hpp>
#include <trueform/topology/cdt/dt3/tetrahedralization_owner.hpp>
#include <trueform/topology/cdt/dt3/weld_tetrahedralization_sites.hpp>
#include <utility>
#include <vector>

namespace {
namespace dt3 = tf::topology::cdt::dt3;
using face_i32 = tf::exact::int32;
using face_coord = std::int64_t;
using face_t = std::array<int, 3>;
using face_t2 = tf::exact::meta<face_i32>::T2;
using face_owner = dt3::tetrahedralization_owner<int, face_coord, face_i32>;
using face_phase_t = dt3::tet_segment_phase<int, face_coord>;
using face_recovery_t = dt3::tet_face_recovery<int, face_coord, face_i32>;
using face_facets_t = tf::offset_block_buffer<int, face_t>;

/// The sites standing at `points`, loaded and welded into `owner` as a
/// build welds them; per point, the slot it took.
template <typename Int>
auto face_sites_of(const std::vector<tf::exact::pt3<face_coord>> &points,
                   dt3::tetrahedralization_owner<int, face_coord, Int> &owner)
    -> std::vector<int> {
  std::vector<face_coord> flat;
  for (const auto &p : points)
    flat.insert(flat.end(), {p[0], p[1], p[2]});
  dt3::load_tetrahedralization_sites(owner,
                                     tf::make_points<3>(tf::make_range(flat)));
  dt3::weld_tetrahedralization_sites(owner);
  return {owner._index_map.f().begin(), owner._index_map.f().end()};
}

/// The triangles `facet` names by point, named by the slots `slot` took.
auto face_slotted(const std::vector<face_t> &facet,
                  const std::vector<int> &slot) -> std::vector<face_t> {
  std::vector<face_t> out;
  for (const auto &t : facet)
    out.push_back({slot[std::size_t(t[0])], slot[std::size_t(t[1])],
                   slot[std::size_t(t[2])]});
  return out;
}

/// A cloud on the int32 lattice, stored scaled by 2^scale.
auto face_cloud(std::uint32_t seed, std::size_t n, int span, int scale)
    -> std::vector<face_coord> {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> axis(-span, span);
  std::vector<face_coord> flat;
  for (std::size_t i = 0; i < 3 * n; ++i)
    flat.push_back(face_coord(axis(rng)) * (face_coord(1) << scale));
  return flat;
}

/// A cube of side 2^11 cut `cuts` by `cuts` on each side, wound outward,
/// among a cloud of `cloud` sites, every coordinate a multiple of 2^scale.
auto face_cube_case(std::uint32_t seed, int cuts, int scale, std::size_t cloud,
                    std::vector<face_coord> &flat, std::vector<face_t> &faces)
    -> void {
  const face_coord unit = face_coord(1) << scale;
  const int side = 1 << 10;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> axis(-2 * side, 2 * side);
  flat.clear();
  faces.clear();
  std::vector<std::array<int, 3>> corners;
  const auto site = [&](int x, int y, int z) {
    const std::array<int, 3> key{x, y, z};
    const auto at = std::find(corners.begin(), corners.end(), key);
    if (at != corners.end())
      return int(at - corners.begin());
    corners.push_back(key);
    flat.insert(flat.end(), {face_coord(x) * unit, face_coord(y) * unit,
                             face_coord(z) * unit});
    return int(corners.size() - 1);
  };
  const int step = 2 * side / cuts;
  for (int axis_index = 0; axis_index < 3; ++axis_index)
    for (int sign : {-1, 1})
      for (int i = 0; i < cuts; ++i)
        for (int j = 0; j < cuts; ++j) {
          const auto at = [&](int a, int b) {
            std::array<int, 3> p{};
            p[std::size_t(axis_index)] = sign * side;
            p[std::size_t((axis_index + 1) % 3)] = -side + a * step;
            p[std::size_t((axis_index + 2) % 3)] = -side + b * step;
            return site(p[0], p[1], p[2]);
          };
          const int a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1),
                    d = at(i, j + 1);
          if (sign > 0) {
            faces.push_back({a, b, c});
            faces.push_back({a, c, d});
          } else {
            faces.push_back({a, c, b});
            faces.push_back({a, d, c});
          }
        }
  for (std::size_t i = 0; i < cloud; ++i)
    flat.insert(flat.end(),
                {face_coord(axis(rng)) * unit, face_coord(axis(rng)) * unit,
                 face_coord(axis(rng)) * unit});
}

/// Pages of a book about one spine, each a triangle on the spine's two
/// ends, among a cloud.
auto face_book_case(std::uint32_t seed, std::vector<face_coord> &flat,
                    std::vector<face_t> &faces) -> void {
  const int depth = 12;
  const face_coord unit = face_coord(1) << depth;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> axis(-1500, 1500);
  flat = {0, 0, -1000 * unit, 0, 0, 1000 * unit};
  faces.clear();
  const int pages = 6 + int(seed % 5);
  const double spread = 0.05 + 0.02 * double(seed % 4);
  for (int p = 0; p < pages; ++p) {
    const double angle = spread * p;
    const double reach = 800 + 300 * (p % 3);
    flat.insert(flat.end(),
                {face_coord(std::llround(reach * std::cos(angle))) * unit,
                 face_coord(std::llround(reach * std::sin(angle))) * unit,
                 face_coord(axis(rng) / 4) * unit});
    faces.push_back({0, 1, 2 + p});
  }
  for (int i = 0; i < 200; ++i)
    flat.insert(flat.end(),
                {face_coord(axis(rng)) * unit, face_coord(axis(rng)) * unit,
                 face_coord(axis(rng)) * unit});
}

/// One block per face of `faces`, each of one triangle, or, given
/// `offsets`, block `f` the triangles from `offsets[f]` to `offsets[f + 1]`.
auto face_facets_of(const std::vector<face_t> &faces,
                    std::vector<int> offsets = {}) -> face_facets_t {
  if (offsets.empty())
    for (std::size_t f = 0; f <= faces.size(); ++f)
      offsets.push_back(int(f));
  face_facets_t facets;
  facets.offsets_buffer().allocate(offsets.size());
  std::copy(offsets.begin(), offsets.end(), facets.offsets_buffer().begin());
  for (const auto &face : faces)
    facets.data_buffer().push_back(face);
  return facets;
}

/// Build the complex of `flat`, rename `faces` and `segments` into its
/// slots, then run phase S over the faces' rims and the segments given and
/// phase F over the faces, serially, blocked into facets at `offsets`
/// (@ref face_facets_of). Returns whether every append was published.
auto face_case(face_owner &owner, const std::vector<face_coord> &flat,
               std::vector<std::array<int, 2>> segments,
               std::vector<face_t> &faces, face_phase_t &phase,
               face_recovery_t &recovery, int depth = 12,
               std::size_t budget_ratio = 32,
               const std::vector<int> &offsets = {}) -> bool {
  REQUIRE(dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat))));
  for (auto &face : faces)
    for (auto &corner : face)
      corner = owner._index_map.f()[std::size_t(corner)];
  for (auto &segment : segments)
    for (auto &end : segment)
      end = owner._index_map.f()[std::size_t(end)];
  tf::none_t serial{};
  const auto facets = face_facets_of(faces, offsets);
  const auto blocks =
      tf::make_offset_block_range(facets.offsets_buffer(), facets.data_buffer());
  phase.segment.site_budget = dt3::tet_site_count(owner) * budget_ratio;
  REQUIRE(dt3::recover_tet_segment_phase<false>(owner, serial, segments, blocks,
                                                depth, phase));
  return dt3::recover_tet_face_phase<false>(owner, serial, blocks, phase,
                                            recovery);
}

/// The test's own reading of a face's cover: every facet of the published
/// complex whose corners lie on the face's plane inside its closed triangle,
/// each once; `tiled` when their projected areas sum to the face's. Every
/// site is read at the deepest depth any stands, on the int64 lattice.
auto face_cover_of(const face_owner &owner, const face_t &face, bool &tiled)
    -> std::vector<face_t> {
  using wide_t2 = tf::exact::meta<tf::exact::int64>::T2;
  int depth = 0;
  for (std::size_t s = 0; s < dt3::tet_site_count(owner); ++s)
    depth = std::max(depth, dt3::tet_site_depth(owner, int(s)));
  const auto pt = [&owner, depth](int s) {
    return dt3::tet_aligned_position<face_coord>(
        dt3::tet_site_position(owner, s), depth);
  };
  std::array<wide_t2, 3> normal{};
  for (std::size_t k = 0; k < 3; ++k) {
    const std::size_t i = (k + 1) % 3, j = (k + 2) % 3;
    normal[k] = wide_t2(pt(face[1])[i] - pt(face[0])[i]) *
                    wide_t2(pt(face[2])[j] - pt(face[0])[j]) -
                wide_t2(pt(face[1])[j] - pt(face[0])[j]) *
                    wide_t2(pt(face[2])[i] - pt(face[0])[i]);
  }
  const auto axes = tf::exact::projection_axes<tf::exact::int64>(normal);
  const auto area = [&axes](const auto &x, const auto &y, const auto &z) {
    const auto i = std::size_t(axes.first), j = std::size_t(axes.second);
    return wide_t2(y[i] - x[i]) * wide_t2(z[j] - x[j]) -
           wide_t2(y[j] - x[j]) * wide_t2(z[i] - x[i]);
  };
  const auto a = pt(face[0]), b = pt(face[1]), c = pt(face[2]);
  const wide_t2 whole = area(a, b, c);
  const auto inside = [&](int s) {
    const auto p = pt(s);
    const bool on = tf::exact::orient3d_value_scaled<tf::exact::int64>(
                        a, b, c, p) == wide_t2(0);
    if (!on)
      return false;
    const auto u = area(a, b, p), v = area(b, c, p), w = area(c, a, p);
    return whole > wide_t2(0)
               ? (u >= wide_t2(0) && v >= wide_t2(0) && w >= wide_t2(0))
               : (u <= wide_t2(0) && v <= wide_t2(0) && w <= wide_t2(0));
  };
  std::vector<face_t> facets;
  for (std::size_t cell = 0; cell < owner._n_finite; ++cell) {
    const auto corners = owner._corners[cell];
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const auto f = dt3::tet_facet(corners, slot);
      if (!inside(int(f[0])) || !inside(int(f[1])) || !inside(int(f[2])))
        continue;
      face_t key{int(f[0]), int(f[1]), int(f[2])};
      std::sort(key.begin(), key.end());
      facets.push_back(key);
    }
  }
  std::sort(facets.begin(), facets.end());
  facets.erase(std::unique(facets.begin(), facets.end()), facets.end());
  wide_t2 sum(0);
  for (const auto &f : facets) {
    const auto s = area(pt(f[0]), pt(f[1]), pt(f[2]));
    sum = sum + (s < wide_t2(0) ? -s : s);
  }
  tiled = sum == (whole < wide_t2(0) ? -whole : whole);
  return facets;
}

/// Every face recovered, each tiled by the test's own reading, and the
/// phase law's oracles silent.
auto face_recovered_whole(const face_owner &owner,
                          const std::vector<face_t> &faces,
                          const face_phase_t &phase,
                          const face_recovery_t &recovery) -> bool {
  for (auto status : phase.segment.status)
    if (status != dt3::tet_segment_status::recovered)
      return false;
  for (std::size_t f = 0; f < faces.size(); ++f) {
    if (recovery.status[f] != dt3::tet_face_status::recovered)
      return false;
    bool tiled = false;
    face_cover_of(owner, faces[f], tiled);
    if (!tiled)
      return false;
  }
  const auto facets = face_facets_of(faces);
  const auto blocks =
      tf::make_offset_block_range(facets.offsets_buffer(), facets.data_buffer());
  return dt3::check_tetrahedralization_validity(owner) &&
         dt3::check_tet_face_fixpoint(owner, blocks, recovery) &&
         dt3::check_tet_pieces_stand(owner, phase.pieces);
}

/// The owners of the lane facet `key`, slots ascending: its stack, or its
/// one owner, or none.
auto face_owners_of(const face_owner &owner, face_t key) -> std::vector<int> {
  std::sort(key.begin(), key.end());
  for (std::size_t cell = 0; cell < owner._n_finite; ++cell)
    for (std::size_t slot = 0; slot < 4; ++slot) {
      if (dt3::tet_facet_key(owner._corners[cell], slot) != key)
        continue;
      if (!dt3::tet_facet_is_protected(owner, int(cell), slot))
        return {};
      const int who = owner._owners[4 * cell + slot];
      if (who != face_owner::shared_owner)
        return {who};
      std::vector<int> stack;
      const auto range = dt3::tet_shared_owners(owner, key);
      for (auto at = range.first; at != range.second; ++at)
        stack.push_back(at->second);
      return stack;
    }
  return {};
}

/// The third corners of the lane facets face `f` holds on the edge `p q`,
/// each once.
auto face_wings_of(const face_owner &owner, int f, int p, int q)
    -> std::vector<int> {
  std::vector<face_t> keys;
  for (std::size_t cell = 0; cell < owner._n_finite; ++cell)
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const auto key = dt3::tet_facet_key(owner._corners[cell], slot);
      if (std::find(key.begin(), key.end(), p) == key.end() ||
          std::find(key.begin(), key.end(), q) == key.end())
        continue;
      const auto owners = face_owners_of(owner, key);
      if (std::find(owners.begin(), owners.end(), f) != owners.end())
        keys.push_back(key);
    }
  std::sort(keys.begin(), keys.end());
  keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
  std::vector<int> wings;
  for (const auto &key : keys)
    for (auto corner : key)
      if (corner != p && corner != q)
        wings.push_back(corner);
  return wings;
}
} // namespace

TEST_CASE("tet faces: facets the complex holds are covered without a flip",
          "[topology][delaunay3][faces]") {
  for (std::uint32_t seed = 9800; seed < 9810; ++seed) {
    CAPTURE(seed);
    const auto flat = face_cloud(seed, 400, 1 << 12, 12);
    face_owner probe;
    REQUIRE(dt3::build_tetrahedralization(
        probe, tf::make_points<3>(tf::make_range(flat))));
    std::vector<face_t> faces;
    std::mt19937 rng(seed);
    for (int k = 0; k < 40; ++k) {
      const std::size_t cell = std::size_t(rng()) % probe._n_finite;
      const auto f = dt3::tet_facet(probe._corners[cell], std::size_t(rng()) % 4);
      face_t face{int(f[0]), int(f[1]), int(f[2])};
      for (auto &corner : face)
        corner = probe._index_map.kept_ids()[std::size_t(corner)];
      faces.push_back(face);
    }
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
    CHECK(recovery.census.covered == faces.size());
    CHECK(recovery.census.points == 0);
    CHECK(recovery.census.refused == 0);
  }
}

TEST_CASE("tet faces: a cube's surface among a cloud is tiled by the complex",
          "[topology][delaunay3][faces][surface]") {
  for (std::uint32_t seed = 9900; seed < 9910; ++seed) {
    CAPTURE(seed);
    std::vector<face_coord> flat;
    std::vector<face_t> faces;
    face_cube_case(seed, 3, 12, 300, flat, faces);
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
    CHECK(recovery.census.refused == 0);
  }
}

TEST_CASE("tet faces: a cube alone is tiled through its hull",
          "[topology][delaunay3][faces][hull]") {
  // Every face lies on the convex hull, where the Delaunay hull's
  // diagonals need not be the input's: the hull's own 4-4 flips recover
  // them, no point placed.
  for (const int cuts : {1, 2, 3}) {
    CAPTURE(cuts);
    std::vector<face_coord> flat;
    std::vector<face_t> faces;
    face_cube_case(1, cuts, 12, 0, flat, faces);
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
    CHECK(recovery.census.points == 0);
    CHECK(phase.segment.split_rows == 0);
  }
}

TEST_CASE("tet faces: a book of pages about one spine recovers whole",
          "[topology][delaunay3][faces][surface]") {
  for (std::uint32_t seed = 9950; seed < 9970; ++seed) {
    CAPTURE(seed);
    std::vector<face_coord> flat;
    std::vector<face_t> faces;
    face_book_case(seed, flat, faces);
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
  }
}

TEST_CASE("tet faces: faces lying on one another stack on their facet",
          "[topology][delaunay3][faces][stack]") {
  // A fold is two faces on one facet wound against each other; a surface
  // lying on itself can put three or more there. The lane holds the stack
  // whole, every face of it recovered.
  const auto flat = face_cloud(31, 200, 1 << 12, 12);
  auto sites = flat;
  const face_coord unit = face_coord(1) << 12;
  const int base = int(flat.size() / 3);
  sites.insert(sites.end(), {-900 * unit, -900 * unit, 0, 900 * unit,
                             -900 * unit, 0, 0, 900 * unit, 0});
  for (const std::size_t depth : {2, 3, 4}) {
    CAPTURE(depth);
    std::vector<face_t> faces;
    for (std::size_t k = 0; k < depth; ++k)
      faces.push_back(k % 2 ? face_t{base, base + 2, base + 1}
                            : face_t{base, base + 1, base + 2});
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    auto named = faces;
    REQUIRE(face_case(owner, sites, {}, named, phase, recovery));
    CHECK(face_recovered_whole(owner, named, phase, recovery));
    bool tiled = false;
    const auto cover = face_cover_of(owner, named[0], tiled);
    CHECK(tiled);
    REQUIRE(!cover.empty());
    for (const auto &facet : cover) {
      auto owners = face_owners_of(owner, facet);
      std::sort(owners.begin(), owners.end());
      std::vector<int> expected;
      for (std::size_t k = 0; k < depth; ++k)
        expected.push_back(int(k));
      CHECK(owners == expected);
    }
  }
}

TEST_CASE("tet faces: a fin on a plate's interior edge holds the plate from "
          "both sides",
          "[topology][delaunay3][faces][slit]") {
  // The fin O B F hangs off the plate's interior edge O B, so three
  // triangles hold O B and it bounds the plate, which reaches around O as
  // one facet on both sides of it: the plate's block states O B twice, each
  // with the plate's own triangle holding it, and the plate is seeded from
  // both sides. Sites standing on O B and on C D split their chains, the
  // last piece of each running from the higher slot to the lower, and sites
  // on the plate's plane outside it beside D A and beside that piece of C D
  // stand where a side read the wrong way would seed the plate. Pierced
  // beside the fin, the plate's first read finds no facet on the C side of
  // O B, and the serial pass recovers it there.
  const face_coord unit = face_coord(1) << 12;
  for (const bool pierced : {false, true}) {
    CAPTURE(pierced);
    // O A B C D on the plate, the fin's apex F, sites far about them, one
    // on the plane outside the plate beside D A, one standing on O B, one
    // standing on C D with one outside the plate beside it, and where
    // pierced two sites straddling the plate beside the fin
    std::vector<std::array<face_coord, 3>> points{{0, 0, 0},
                                                  {-1000, -900, 0},
                                                  {1100, -1000, 0},
                                                  {900, 1000, 0},
                                                  {-1000, 1100, 0},
                                                  {550, -500, 1000},
                                                  {-2600, -2900, 2700},
                                                  {2800, -2500, 2900},
                                                  {2500, 2700, 2600},
                                                  {-2700, 2800, 3000},
                                                  {-2900, -2600, -2800},
                                                  {2700, -2800, -2500},
                                                  {2600, 2900, -2700},
                                                  {-2800, 2500, -2600},
                                                  {-2100, 100, 0},
                                                  {550, -500, 0},
                                                  {824, 1004, 0},
                                                  {865, 1062, 0}};
    if (pierced)
      points.insert(points.end(), {{600, -400, 10}, {600, -400, -10}});
    std::vector<face_coord> flat;
    for (const auto &p : points)
      flat.insert(flat.end(), {p[0] * unit, p[1] * unit, p[2] * unit});
    // the fin, then the plate's four triangles about O, wound alike
    std::vector<face_t> faces{
        {0, 2, 5}, {1, 2, 0}, {2, 3, 0}, {3, 4, 0}, {4, 1, 0}};
    const std::vector<int> offsets{0, 1, 5};
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(
        face_case(owner, flat, {}, faces, phase, recovery, 12, 32, offsets));
    const int o = faces[1][2], a = faces[1][0], b = faces[1][1],
              c = faces[2][1], d = faces[3][1];
    const auto segment_of = [&phase](int p, int q) {
      const std::array<int, 2> key{std::min(p, q), std::max(p, q)};
      const auto at =
          std::lower_bound(phase.segments.begin(), phase.segments.end(), key);
      return at != phase.segments.end() && *at == key
                 ? std::size_t(at - phase.segments.begin())
                 : phase.segments.size();
    };
    const auto descends = [](const auto &chain) {
      for (std::size_t i = 1; i < chain.size(); ++i)
        if (chain[i - 1].site > chain[i].site)
          return true;
      return false;
    };
    const auto segment = segment_of(o, b);
    REQUIRE(segment < phase.segments.size());
    std::vector<std::array<int, 3>> holders;
    const auto &plate = phase.edges.offsets_buffer();
    for (auto e = std::size_t(plate[1]); e < std::size_t(plate[2]); ++e)
      if (std::size_t(phase.edges.data_buffer()[e]) == segment)
        holders.push_back(phase.holders[e]);
    std::sort(holders.begin(), holders.end());
    std::vector<std::array<int, 3>> expected{{b, o, a}, {o, b, c}};
    std::sort(expected.begin(), expected.end());
    CHECK(holders == expected);
    CHECK(recovery.status[0] == dt3::tet_face_status::recovered);
    CHECK(recovery.status[1] == dt3::tet_face_status::recovered);
    const bool serial =
        std::find(recovery.uncovered.begin(), recovery.uncovered.end(), 1) !=
        recovery.uncovered.end();
    CHECK(serial == pierced);
    const auto chain = phase.segment.chains[segment];
    REQUIRE(chain.size() > 2);
    CHECK(descends(chain));
    const auto cd = segment_of(c, d);
    REQUIRE(cd < phase.segments.size());
    REQUIRE(phase.segment.chains[cd].size() > 2);
    CHECK(descends(phase.segment.chains[cd]));
    if (pierced) {
      // the plate is the serial pass's last face, so the rim it read stands
      CHECK(recovery.census.flipped + recovery.census.points >= 1);
      for (std::size_t i = 1; i < chain.size(); ++i) {
        const int p = chain[i - 1].site, q = chain[i].site;
        const auto pieces = std::equal_range(
            recovery.rim.begin(), recovery.rim.end(),
            dt3::tet_rim_piece<int>{std::min(p, q), std::max(p, q), 0});
        REQUIRE(pieces.second - pieces.first == 2);
        CHECK(pieces.first->side != 0);
        CHECK(pieces.first->side == -(pieces.first + 1)->side);
      }
    }
    int depth = 0;
    for (std::size_t s = 0; s < dt3::tet_site_count(owner); ++s)
      depth = std::max(depth, dt3::tet_site_depth(owner, int(s)));
    const auto side = [&owner, depth](int p, int q, int w) {
      using wide_t2 = tf::exact::meta<tf::exact::int64>::T2;
      const auto at = [&owner, depth](int s) {
        return dt3::tet_aligned_position<face_coord>(
            dt3::tet_site_position(owner, s), depth);
      };
      const auto x = at(p), y = at(q), z = at(w);
      const wide_t2 value = wide_t2(y[0] - x[0]) * wide_t2(z[1] - x[1]) -
                            wide_t2(y[1] - x[1]) * wide_t2(z[0] - x[0]);
      return value > wide_t2(0) ? 1 : value < wide_t2(0) ? -1 : 0;
    };
    for (std::size_t i = 1; i < chain.size(); ++i) {
      const int p = chain[i - 1].site, q = chain[i].site;
      const auto wings = face_wings_of(owner, 1, p, q);
      REQUIRE(wings.size() == 2);
      CHECK(side(p, q, wings[0]) != 0);
      CHECK(side(p, q, wings[0]) == -side(p, q, wings[1]));
    }
    const auto facets = face_facets_of(faces, offsets);
    const auto blocks = tf::make_offset_block_range(facets.offsets_buffer(),
                                                    facets.data_buffer());
    CHECK(dt3::check_tetrahedralization_validity(owner));
    CHECK(dt3::check_tet_face_fixpoint(owner, blocks, recovery));
    CHECK(dt3::check_tet_pieces_stand(owner, phase.pieces));
  }
}

TEST_CASE("tet faces: a plate among sites on its own plane is tiled as found",
          "[topology][delaunay3][faces][coverage]") {
  // The complex's own edges on the plane need not be the input's two
  // diagonals; the face is recovered when its plane inside its rim is
  // tiled by whatever facets stand there.
  for (std::uint32_t seed = 9700; seed < 9710; ++seed) {
    CAPTURE(seed);
    const face_coord unit = face_coord(1) << 12;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> span(-1000, 1000);
    std::uniform_int_distribution<int> off(-2000, 2000);
    std::vector<face_coord> flat{-1500 * unit, -1500 * unit, 0,
                                 1500 * unit,  -1500 * unit, 0,
                                 1500 * unit,  1500 * unit,  0,
                                 -1500 * unit, 1500 * unit,  0};
    for (int i = 0; i < 60; ++i)
      flat.insert(flat.end(), {face_coord(span(rng)) * unit,
                               face_coord(span(rng)) * unit, 0});
    for (int i = 0; i < 300; ++i)
      flat.insert(flat.end(),
                  {face_coord(off(rng)) * unit, face_coord(off(rng)) * unit,
                   face_coord(off(rng)) * unit});
    std::vector<face_t> faces{{0, 1, 2}, {0, 2, 3}};
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
    CHECK(recovery.census.refused == 0);
    bool tiled = false;
    const auto cover = face_cover_of(owner, faces[0], tiled);
    CHECK(tiled);
    CHECK(cover.size() > 1);
  }
}

TEST_CASE("tet faces: Schoenhardt's prism resolves with a point",
          "[topology][delaunay3][faces][steiner]") {
  // The twisted prism has no tetrahedralization of its own vertices: a
  // point is placed, on a reflex edge by phase S or on a face by phase F.
  const face_coord unit = face_coord(1) << 12;
  const double twist = 0.52;
  std::vector<face_coord> flat;
  for (int level = 0; level < 2; ++level)
    for (int k = 0; k < 3; ++k) {
      const double angle = 2.0943951 * k + (level ? twist : 0.0);
      flat.insert(flat.end(),
                  {face_coord(std::llround(1000.0 * std::cos(angle))) * unit,
                   face_coord(std::llround(1000.0 * std::sin(angle))) * unit,
                   face_coord(level ? 1000 : -1000) * unit});
    }
  std::vector<face_t> faces{{0, 2, 1}, {3, 4, 5}};
  for (int k = 0; k < 3; ++k) {
    const int a = k, b = (k + 1) % 3, c = 3 + k, d = 3 + (k + 1) % 3;
    faces.push_back({a, b, d});
    faces.push_back({a, d, c});
  }
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
  CHECK(face_recovered_whole(owner, faces, phase, recovery));
  CHECK(phase.segment.split_rows + recovery.census.points >= 1);
  CHECK(recovery.census.refused == 0);
}

TEST_CASE("tet faces: a segment piercing a face refuses it as crossed",
          "[topology][delaunay3][faces][refusal]") {
  const face_coord unit = face_coord(1) << 12;
  const auto cloud = face_cloud(77, 150, 1 << 11, 12);
  auto flat = cloud;
  const int base = int(cloud.size() / 3);
  // the segment crosses the plane at 15/34 of its length, no dyadic
  // split's position, so no split lands on the plane and the pierce is
  // no site's
  flat.insert(flat.end(), {-900 * unit, -900 * unit, 0, 900 * unit, -900 * unit,
                           0, 0, 900 * unit, 0, 0, 0, -1500 * unit, 17 * unit, 0,
                           1900 * unit});
  std::vector<face_t> faces{{base, base + 1, base + 2}};
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, flat, {{base + 3, base + 4}}, faces, phase,
                    recovery));
  CHECK(recovery.status[0] == dt3::tet_face_status::refused);
  CHECK(recovery.census.crossed == 1);
  CHECK(dt3::check_tetrahedralization_validity(owner));
}

TEST_CASE("tet faces: a face whose lattice point stands places one below the "
          "frame",
          "[topology][delaunay3][faces][depth]") {
  // A B C lies on 9 x + 7 z = 0, whose lattice runs in lines along the rim
  // A B seven units apart in x: the rim's own, one between, and C's. Six
  // sites within two units of z = 0 about the rim leave Delaunay edges
  // passing over A B that cross the plane within a fifth of a line off the
  // rim, so every point the face can place on the frame lies on the line
  // between and stands already: the frame's depth witness. The face reads
  // its lattice deeper and places the shallowest point in the piercing
  // edge's kernel below the frame, and recovers. Stored at 2^10 with frames
  // of 10 bits, the same scene has lines between and recovers on the frame.
  const std::vector<face_coord> flat{0,   -169, 0, 0,   169, 0, -14, -126, 18,
                                     -50, -6,   0, 40,  30,  1, -7,  120,  1,
                                     29,  42,   1, -48, 147, 0, 7,   -72,  2};
  const auto deep_sites = [](const face_owner &owner) {
    std::size_t count = 0;
    for (std::size_t site = 0; site < dt3::tet_site_count(owner); ++site)
      count += dt3::tet_site_depth(owner, int(site)) != 0;
    return count;
  };
  {
    std::vector<face_t> faces{{0, 1, 2}};
    face_owner owner;
    face_phase_t phase;
    face_recovery_t recovery;
    REQUIRE(face_case(owner, flat, {}, faces, phase, recovery, 0));
    CHECK(face_recovered_whole(owner, faces, phase, recovery));
    CHECK(recovery.census.standing == 0);
    CHECK(recovery.census.points >= 1);
    CHECK(deep_sites(owner) >= 1);
  }
  auto framed = flat;
  for (auto &coordinate : framed)
    coordinate *= 1024;
  std::vector<face_t> faces{{0, 1, 2}};
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, framed, {}, faces, phase, recovery, 10));
  CHECK(face_recovered_whole(owner, faces, phase, recovery));
  CHECK(recovery.census.standing == 0);
  CHECK(deep_sites(owner) == 0);
}

TEST_CASE("tet faces: a face whose lattice names no point on the frame "
          "places one below it",
          "[topology][delaunay3][faces][depth]") {
  // Nine sites on the bare lattice, the face A B C among them: an edge of
  // the complex pierces the face, and no point of the face's own lattice
  // lies in the face off its rim, so on the frame the face has no point to
  // place. Its lattice read deeper names one in the kernel of the piercing
  // edge's ring, placed below the frame, and the face recovers.
  const std::vector<face_coord> flat{-13, 16,  -32, 15,  36, 6,  9,   -29, -32,
                                     21,  34,  -17, -35, 2,  6,  -19, 11,  33,
                                     2,   -27, -22, -17, 5,  12, -7,  14,  -26};
  std::vector<face_t> faces{{0, 1, 2}};
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, flat, {}, faces, phase, recovery, 0));
  CHECK(face_recovered_whole(owner, faces, phase, recovery));
  CHECK(recovery.census.off_lattice == 0);
  CHECK(recovery.census.points == 1);
  int deepest = 0;
  for (std::size_t site = 0; site < dt3::tet_site_count(owner); ++site)
    deepest = std::max(deepest, dt3::tet_site_depth(owner, int(site)));
  CHECK(deepest > 0);
}

TEST_CASE("tet faces: the site budget refuses typed through both phases",
          "[topology][delaunay3][faces][refusal]") {
  // With no site to spare, the prism's reflex edge cannot split: its
  // segment refuses for the budget and the faces on it for their rim.
  const face_coord unit = face_coord(1) << 12;
  const double twist = 0.52;
  std::vector<face_coord> flat;
  for (int level = 0; level < 2; ++level)
    for (int k = 0; k < 3; ++k) {
      const double angle = 2.0943951 * k + (level ? twist : 0.0);
      flat.insert(flat.end(),
                  {face_coord(std::llround(1000.0 * std::cos(angle))) * unit,
                   face_coord(std::llround(1000.0 * std::sin(angle))) * unit,
                   face_coord(level ? 1000 : -1000) * unit});
    }
  std::vector<face_t> faces{{0, 2, 1}, {3, 4, 5}};
  for (int k = 0; k < 3; ++k) {
    const int a = k, b = (k + 1) % 3, c = 3 + k, d = 3 + (k + 1) % 3;
    faces.push_back({a, b, d});
    faces.push_back({a, d, c});
  }
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, flat, {}, faces, phase, recovery, 12, 1));
  std::size_t budget = 0;
  for (auto status : phase.segment.status)
    budget += status == dt3::tet_segment_status::budget;
  CHECK(budget + recovery.census.budget >= 1);
  CHECK(recovery.census.open_rim + recovery.census.budget >= 1);
  CHECK(recovery.census.points == 0);
  CHECK(dt3::check_tetrahedralization_validity(owner));
}

TEST_CASE("tet faces: the fixpoint oracles catch a dropped piece and an "
          "untiled face",
          "[topology][delaunay3][faces][oracle]") {
  std::vector<face_coord> flat;
  std::vector<face_t> faces;
  face_cube_case(9901, 3, 12, 300, flat, faces);
  face_owner owner;
  face_phase_t phase;
  face_recovery_t recovery;
  REQUIRE(face_case(owner, flat, {}, faces, phase, recovery));
  REQUIRE(face_recovered_whole(owner, faces, phase, recovery));
  const auto facets = face_facets_of(faces);
  const auto blocks =
      tf::make_offset_block_range(facets.offsets_buffer(), facets.data_buffer());
  {
    auto pieces = phase.pieces;
    bool dropped = false;
    for (const auto &chain : phase.segment.chains)
      if (chain.size() > 2) {
        pieces.push_back({std::min(chain[0].site, chain[2].site),
                          std::max(chain[0].site, chain[2].site)});
        std::sort(pieces.begin(), pieces.end());
        dropped = true;
        break;
      }
    if (dropped)
      CHECK_FALSE(dt3::check_tet_pieces_stand(owner, pieces));
  }
  {
    face_owner untiled = owner;
    bool cleared = false;
    for (std::size_t cell = 0; cell < untiled._n_finite && !cleared; ++cell)
      for (std::size_t slot = 0; slot < 4 && !cleared; ++slot)
        if (dt3::tet_facet_is_protected(untiled, int(cell), slot) &&
            untiled._owners[4 * cell + slot] >= 0) {
          dt3::state_tet_facet_protection(untiled, int(cell), slot, false,
                                          face_owner::none);
          --untiled._n_protected;
          cleared = true;
        }
    REQUIRE(cleared);
    CHECK_FALSE(dt3::check_tet_face_fixpoint(untiled, blocks, recovery));
  }
}

TEST_CASE("tet faces: the flip constraint refuses an edge piercing the polygon",
          "[topology][delaunay3][faces][eligibility]") {
  const face_coord unit = face_coord(1) << 12;
  const std::vector<face_coord> flat{
      -1000 * unit, -1000 * unit, 0,          1000 * unit, -1000 * unit, 0,
      0,            1000 * unit,  0,          100 * unit,  100 * unit,   500 * unit,
      100 * unit,   100 * unit,   -500 * unit, 3000 * unit, 3000 * unit,  500 * unit,
      3000 * unit,  3000 * unit,  -500 * unit, 0,           0,            900 * unit};
  face_owner owner;
  REQUIRE(dt3::build_tetrahedralization(
      owner, tf::make_points<3>(tf::make_range(flat))));
  const auto &slot = owner._index_map.f();
  const std::vector<face_t> polygon{{slot[0], slot[1], slot[2]}};
  dt3::tet_flip_constraints<int, face_coord, face_i32> fc;
  fc.polygon = {polygon.data(), polygon.data() + polygon.size()};
  fc.axes = {0, 1};
  // a 2-3 whose new edge e d crosses the triangle's interior is refused,
  // one crossing its plane outside the triangle is not
  CHECK_FALSE(dt3::tet_flip_eligible(owner, 1, slot[0], slot[1], slot[7],
                                     slot[3], slot[4], fc));
  CHECK(dt3::tet_flip_eligible(owner, 1, slot[0], slot[1], slot[7], slot[5],
                               slot[6], fc));
}

TEST_CASE(
    "tet faces: a plane's lattice point lies in the facet near the pierce",
    "[topology][delaunay3][faces][lattice]") {
  using pt = tf::exact::pt3<face_coord>;
  struct case_t {
    pt a, b, c, d, e;
  };
  const face_coord u = face_coord(1) << 12;
  const std::vector<case_t> cases{
      {{0, 0, 0},
       {10 * u, 0, 0},
       {0, 10 * u, 0},
       {3 * u, 4 * u, -5 * u},
       {3 * u, 4 * u, 5 * u}},
      {{0, 0, 0},
       {20 * u, 10 * u, 0},
       {0, 10 * u, 30 * u},
       {5 * u, 5 * u, -20 * u},
       {5 * u, 5 * u, 20 * u}},
      {{1000 * u, 2000 * u, 3000 * u},
       {1307 * u, 2011 * u, 3005 * u},
       {1010 * u, 2401 * u, 3017 * u},
       {1100 * u, 2100 * u, 2500 * u},
       {1100 * u, 2100 * u, 3500 * u}},
  };
  const std::vector<face_t> facet{{0, 1, 2}};
  for (const auto &k : cases) {
    face_owner owner;
    const auto slot = face_sites_of({k.a, k.b, k.c, k.d, k.e}, owner);
    const auto slotted = face_slotted(facet, slot);
    const auto axes = tf::exact::projection_axes<face_i32>(
        dt3::tet_plane_normal<face_i32>(k.a, k.b, k.c));
    const auto od =
        tf::exact::orient3d_value_scaled<face_i32>(k.a, k.b, k.c, k.d);
    const auto oe =
        tf::exact::orient3d_value_scaled<face_i32>(k.a, k.b, k.c, k.e);
    const auto sites = dt3::tet_shallow_decision_of(owner);
    pt out{};
    REQUIRE(dt3::tet_plane_lattice_point(sites, slotted, axes, slot[3], slot[4],
                                         od, oe, out, 0));
    CHECK(tf::exact::orient3d_value_scaled<face_i32>(k.a, k.b, k.c, out) ==
          face_t2(0));
    CHECK(dt3::tet_point_in_facet(sites, slotted, axes, out));
    // the crossing X = d + od / (od - oe) (e - d); |out - X| is within the
    // plane lattice's reduced cell, well under one input unit here
    const double lambda = double(od) / double(od - oe);
    double distance2 = 0;
    for (std::size_t i = 0; i < 3; ++i) {
      const double x = double(k.d[i]) + lambda * double(k.e[i] - k.d[i]);
      distance2 += (double(out[i]) - x) * (double(out[i]) - x);
    }
    CHECK(distance2 < double(u) * double(u));
  }
  face_owner owner;
  const auto slot = face_sites_of(
      {{0, 0, 0}, {10 * u, 0, 0}, {10 * u, 10 * u, 0}, {0, 10 * u, 0}}, owner);
  const auto triangle = face_slotted({{0, 1, 3}}, slot);
  const auto square = face_slotted({{0, 1, 2}, {0, 2, 3}}, slot);
  const std::pair<int, int> axes{0, 1};
  const auto sites = dt3::tet_shallow_decision_of(owner);
  CHECK(dt3::tet_point_in_facet(sites, triangle, axes, pt{3 * u, 4 * u, 0}));
  CHECK_FALSE(
      dt3::tet_point_in_facet(sites, triangle, axes, pt{30 * u, 4 * u, 0}));
  // on the rim the point is a segment's and at a corner a site's; on an
  // edge two triangles hold it is the face's
  CHECK_FALSE(dt3::tet_point_in_facet(sites, triangle, axes, pt{5 * u, 0, 0}));
  CHECK_FALSE(dt3::tet_point_in_facet(sites, triangle, axes, pt{0, 0, 0}));
  CHECK(dt3::tet_point_in_facet(sites, square, axes, pt{5 * u, 5 * u, 0}));
  CHECK_FALSE(
      dt3::tet_point_in_facet(sites, square, axes, pt{10 * u, 5 * u, 0}));
}

TEST_CASE("tet faces: a crossing rounding off the facet's rim walks inward",
          "[topology][delaunay3][faces][lattice]") {
  using pt = tf::exact::pt3<face_coord>;
  const face_coord u = face_coord(1) << 12;
  const auto point_of = [](const std::vector<pt> &points,
                           const std::vector<face_t> &facet, int d, int e,
                           pt &out) {
    face_owner owner;
    const auto slot = face_sites_of(points, owner);
    const auto &t = facet[0];
    const auto &a = points[std::size_t(t[0])];
    const auto &b = points[std::size_t(t[1])];
    const auto &c = points[std::size_t(t[2])];
    const auto axes = tf::exact::projection_axes<face_i32>(
        dt3::tet_plane_normal<face_i32>(a, b, c));
    return dt3::tet_plane_lattice_point(
        dt3::tet_shallow_decision_of(owner), face_slotted(facet, slot), axes,
        slot[std::size_t(d)], slot[std::size_t(e)],
        tf::exact::orient3d_value_scaled<face_i32>(a, b, c,
                                                   points[std::size_t(d)]),
        tf::exact::orient3d_value_scaled<face_i32>(a, b, c,
                                                   points[std::size_t(e)]),
        out, 0);
  };
  const std::vector<face_t> triangle{{0, 1, 2}};
  pt out{};
  // the crossing (3u, 1/5, 0) rounds onto the rim, (1/5, 1/5, 0) onto a
  // corner; the line one in holds the point nearest it
  REQUIRE(point_of({{0, 0, 0},
                    {10 * u, 0, 0},
                    {0, 10 * u, 0},
                    {3 * u, 0, -1},
                    {3 * u, 1, 4}},
                   triangle, 3, 4, out));
  CHECK(out == pt{3 * u, 1, 0});
  REQUIRE(point_of(
      {{0, 0, 0}, {10 * u, 0, 0}, {0, 10 * u, 0}, {1, 1, -4}, {0, 0, 1}},
      triangle, 3, 4, out));
  CHECK(out == pt{1, 1, 0});
  // the crossing on the edge two triangles hold is kept
  REQUIRE(point_of({{0, 0, 0},
                    {10 * u, 0, 0},
                    {10 * u, 10 * u, 0},
                    {0, 10 * u, 0},
                    {5 * u, 5 * u, -1},
                    {5 * u, 5 * u, 1}},
                   {{0, 1, 2}, {0, 2, 3}}, 4, 5, out));
  CHECK(out == pt{5 * u, 5 * u, 0});
}

TEST_CASE("tet faces: a sliver on the int64 rung holds its lattice point",
          "[topology][delaunay3][faces][lattice]") {
  using wide = tf::exact::int64;
  using pt = tf::exact::pt3<std::int64_t>;
  // two faces of one model on the int64 lattice framed by 2^10: the nearest
  // lattice point of the first's plane to its crossing lies past the sliver,
  // of the second's on its rim
  const std::vector<std::vector<pt>> cases{
      {{-231772921416236032, -1204417789520040960, -313717695811369984},
       {-231701001808845824, -1356737022893912064, -367780090951892992},
       {-231772921416236032, -1204417628966608896, -313719002577469440},
       {116480996129210368, -1211460900392992768, -312132090453543936},
       {-231772921416236032, -1204423600322278400, -313719804509433856}},
      {{-231701001808845824, -1511733366100836352, 62112170262446080},
       {-231701001808845824, -1390098131441991680, 86397024357878784},
       {-231772921416236032, -1213559468501450752, 85246310639635456},
       {116409076521820160, -1237695688741609472, 30913444576565248},
       {-231772921416236032, -1216405087633857536, 85264290541483008}},
  };
  const std::vector<face_t> facet{{0, 1, 2}};
  for (const auto &points : cases) {
    dt3::tetrahedralization_owner<int, std::int64_t, wide> owner;
    const auto slot = face_sites_of(points, owner);
    const auto slotted = face_slotted(facet, slot);
    const auto &a = points[0], &b = points[1], &c = points[2];
    const auto axes =
        tf::exact::projection_axes<wide>(dt3::tet_plane_normal<wide>(a, b, c));
    const auto od = tf::exact::orient3d_value_scaled<wide>(a, b, c, points[3]);
    const auto oe = tf::exact::orient3d_value_scaled<wide>(a, b, c, points[4]);
    const auto sites = dt3::tet_shallow_decision_of(owner);
    REQUIRE(dt3::tet_pierce_in_facet(sites, slotted, axes, slot[3], slot[4], od,
                                     oe));
    pt out{};
    REQUIRE(dt3::tet_plane_lattice_point(sites, slotted, axes, slot[3], slot[4],
                                         od, oe, out, 0));
    CHECK(tf::exact::orient3d_value_scaled<wide>(a, b, c, out) ==
          tf::exact::meta<wide>::T2(0));
    CHECK(dt3::tet_point_in_facet(sites, slotted, axes, out));
  }
}

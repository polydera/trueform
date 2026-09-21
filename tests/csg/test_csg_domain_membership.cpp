/**
 * @file test_csg_domain_membership.cpp
 * @brief The per-cell operand rows a domain read publishes: the winding
 *        depth of every form at every kept cell, plain and within.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <trueform/core/signed_volume.hpp>
#include <trueform/csg.hpp>
#include <trueform/csg/expression/operators.hpp>
#include <trueform/topology/domain_config.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

namespace {

using membership_index_t = int;
using membership_real_t = double;
using membership_mesh_t =
    tf::polygons_buffer<membership_index_t, membership_real_t, 3, 3>;
using membership_operand_t =
    tf::test::tagged_operand<membership_index_t, membership_real_t>;

constexpr auto membership_within_config = tf::arrangement_config{
    tf::intersect_config{tf::intersect_mode::primitives |
                             tf::intersect_mode::within,
                         0.0}};

constexpr auto membership_keep_universe =
    tf::domain_config::ignore_open_fragments;

/// One cell as the read states it: how much space it holds and which
/// operands hold it.
struct membership_row {
  membership_real_t volume;
  std::vector<char> inside;

  auto operator<(const membership_row &o) const -> bool {
    if (std::abs(volume - o.volume) > membership_real_t(1e-6))
      return volume < o.volume;
    return inside < o.inside;
  }
};

auto membership_moved(membership_mesh_t mesh, membership_real_t x,
                      membership_real_t y, membership_real_t z)
    -> membership_mesh_t {
  for (auto &&p : mesh.points()) {
    p[0] += x;
    p[1] += y;
    p[2] += z;
  }
  return mesh;
}

auto membership_reversed(membership_mesh_t mesh) -> membership_mesh_t {
  for (auto face : mesh.faces_buffer())
    std::swap(face[1], face[2]);
  return mesh;
}

/// Two triangles standing on their own, sharing one edge and touching
/// nothing else in the scene.
auto membership_flap() -> membership_mesh_t {
  membership_mesh_t mesh;
  auto &points = mesh.points_buffer();
  points.allocate(4);
  points[0] = tf::point<membership_real_t, 3>{5, 0, 0};
  points[1] = tf::point<membership_real_t, 3>{6, 0, 0};
  points[2] = tf::point<membership_real_t, 3>{6, 1, 0};
  points[3] = tf::point<membership_real_t, 3>{5, 1, 0};
  auto &faces = mesh.faces_buffer();
  faces.allocate(2);
  faces[0] = std::array<membership_index_t, 3>{0, 1, 2};
  faces[1] = std::array<membership_index_t, 3>{0, 2, 3};
  return mesh;
}

template <typename Cells, typename Map>
auto membership_rows_of(const Cells &cells, const Map &imap)
    -> std::vector<membership_row> {
  std::vector<membership_row> rows;
  rows.reserve(cells.size());
  for (std::size_t k = 0; k < cells.size(); ++k) {
    membership_row row;
    row.volume =
        std::abs(membership_real_t(tf::signed_volume(cells[k].polygons())));
    for (std::size_t i = 0; i < std::size_t(imap.inclusion.block_size()); ++i)
      row.inside.push_back(char(imap.inclusion[k][i] ? 1 : 0));
    rows.push_back(std::move(row));
  }
  std::sort(rows.begin(), rows.end());
  return rows;
}

auto membership_require_rows(std::vector<membership_row> got,
                             std::vector<membership_row> want) -> void {
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  REQUIRE(got.size() == want.size());
  for (std::size_t i = 0; i < got.size(); ++i) {
    REQUIRE_THAT(got[i].volume,
                 Catch::Matchers::WithinRel(want[i].volume, 1e-6));
    REQUIRE(got[i].inside == want[i].inside);
  }
}

auto membership_volume_of(const membership_mesh_t &mesh) -> membership_real_t {
  return std::abs(membership_real_t(tf::signed_volume(mesh.polygons())));
}

} // namespace

TEST_CASE("csg domains: two overlapping boxes state both operands, plain and "
          "within",
          "[domains][membership]") {
  auto a = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  auto b = membership_moved(
      tf::make_box_mesh<membership_index_t>(
          membership_real_t(2), membership_real_t(2), membership_real_t(2)),
      membership_real_t(1), membership_real_t(0.5), membership_real_t(0.25));
  const membership_real_t both = membership_real_t(1) *
                                 membership_real_t(1.5) *
                                 membership_real_t(1.75);
  const membership_real_t only = membership_real_t(8) - both;

  std::vector<membership_operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(a));
  operands.push_back(tf::test::make_tagged_operand(b));
  auto forms = tf::test::tagged_forms(operands);

  for (int within = 0; within < 2; ++within) {
    DYNAMIC_SECTION((within ? "within" : "plain")) {
      auto graph = tf::test::build_range_csg_graph(
          tf::test::forms_range(forms), tf::test::no_sheets(),
          within ? membership_within_config : tf::arrangement_config{});

      auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(graph);
      REQUIRE(cells.size() == 3);
      membership_require_rows(membership_rows_of(cells, imap),
                              {{both, {1, 1}}, {only, {1, 0}},
                               {only, {0, 1}}});

      auto count = [&](const tf::csg::selection_t &selection) {
        return tf::test::csg_domains_of(graph, selection).first.size();
      };
      REQUIRE(count(tf::csg::op(0)) == 2);
      REQUIRE(count(tf::csg::op(1)) == 2);
      REQUIRE(count(tf::csg::op(0) & tf::csg::op(1)) == 1);
      REQUIRE(count(tf::csg::op(0) | tf::csg::op(1)) == 3);
      REQUIRE(count(~tf::csg::op(0)) == 1);
    }
  }
}

TEST_CASE("csg domains: a sheet operand states the side it cut",
          "[domains][membership][sheets]") {
  auto box = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  auto knife = tf::make_plane_mesh<membership_index_t>(membership_real_t(4),
                                                       membership_real_t(4));

  std::vector<membership_operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(box));
  operands.push_back(tf::test::make_tagged_operand(knife));
  auto forms = tf::test::tagged_forms(operands);
  const std::vector<int> sheets{1};

  for (int within = 0; within < 2; ++within) {
    DYNAMIC_SECTION((within ? "within" : "plain")) {
      auto graph = tf::test::build_range_csg_graph(
          tf::test::forms_range(forms), tf::test::sheets_of(sheets),
          within ? membership_within_config : tf::arrangement_config{});

      auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(graph);
      REQUIRE(cells.size() == 2);
      membership_require_rows(membership_rows_of(cells, imap),
                              {{membership_real_t(4), {1, 1}},
                               {membership_real_t(4), {1, 0}}});

      auto behind = tf::test::csg_domains_of(graph, tf::csg::op(1));
      REQUIRE(behind.first.size() == 1);
      REQUIRE_THAT(membership_volume_of(behind.first[0]),
                   Catch::Matchers::WithinRel(membership_real_t(4), 1e-6));
    }
  }
}

TEST_CASE("csg domains: separate nested spheres state universe, shell and core",
          "[domains][membership][nesting]") {
  auto big = tf::make_sphere_mesh<membership_index_t>(membership_real_t(2), 32,
                                                      32);
  auto small = tf::make_sphere_mesh<membership_index_t>(membership_real_t(1),
                                                        32, 32);
  const membership_real_t v_big = membership_volume_of(big);
  const membership_real_t v_small = membership_volume_of(small);

  std::vector<membership_operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(big));
  operands.push_back(tf::test::make_tagged_operand(small));
  auto forms = tf::test::tagged_forms(operands);

  for (int within = 0; within < 2; ++within) {
    DYNAMIC_SECTION((within ? "within" : "plain")) {
      auto graph = tf::test::build_range_csg_graph(
          tf::test::forms_range(forms), tf::test::no_sheets(),
          within ? membership_within_config : tf::arrangement_config{});

      auto [cells, ids, imap] =
          tf::test::csg_domains_with_index_map_of(graph,
                                                  membership_keep_universe);
      REQUIRE(cells.size() == 3);
      membership_require_rows(membership_rows_of(cells, imap),
                              {{v_big, {0, 0}},
                               {v_big - v_small, {1, 0}},
                               {v_small, {1, 1}}});
    }
  }
}

TEST_CASE("csg domains: a doubly enclosed interior of one operand reads inside",
          "[domains][membership][within]") {
  // Both shells wound outward in ONE operand: the core lies inside the
  // operand twice over, and twice is inside.
  auto big = tf::make_sphere_mesh<membership_index_t>(membership_real_t(2), 32,
                                                      32);
  auto small = tf::make_sphere_mesh<membership_index_t>(membership_real_t(1),
                                                        32, 32);
  const membership_real_t v_big = membership_volume_of(big);
  const membership_real_t v_small = membership_volume_of(small);

  auto soup = tf::concatenated(big.polygons(), small.polygons());
  auto operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto [cells, ids, imap] =
      tf::test::csg_domains_with_index_map_of(graph, membership_keep_universe);
  REQUIRE(cells.size() == 3);
  membership_require_rows(
      membership_rows_of(cells, imap),
      {{v_big, {0}}, {v_big - v_small, {1}}, {v_small, {1}}});
}

TEST_CASE("csg domains: a reversed inner shell hollows its operand",
          "[domains][membership][within]") {
  // The true hollow: the cavity is enclosed once and left once, so no
  // form covers it — and the blank read still extracts it.
  auto big = tf::make_sphere_mesh<membership_index_t>(membership_real_t(2), 32,
                                                      32);
  auto small = tf::make_sphere_mesh<membership_index_t>(membership_real_t(1),
                                                        32, 32);
  const membership_real_t v_big = membership_volume_of(big);
  const membership_real_t v_small = membership_volume_of(small);

  auto soup = tf::concatenated(big.polygons(),
                               membership_reversed(small).polygons());
  auto operand = tf::test::make_tagged_operand(soup);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto [cells, ids, imap] =
      tf::test::csg_domains_with_index_map_of(graph, membership_keep_universe);
  REQUIRE(cells.size() == 3);
  membership_require_rows(
      membership_rows_of(cells, imap),
      {{v_big, {0}}, {v_big - v_small, {1}}, {v_small, {0}}});

  auto blank = tf::test::csg_domains_with_index_map_of(graph);
  auto &blank_cells = std::get<0>(blank);
  REQUIRE(blank_cells.size() == 2);
  membership_require_rows(
      membership_rows_of(blank_cells, std::get<2>(blank)),
      {{v_big - v_small, {1}}, {v_small, {0}}});
}

TEST_CASE("csg domains: a disjoint open flap changes no row",
          "[domains][membership][within]") {
  auto box = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  auto flapped = tf::concatenated(box.polygons(), membership_flap().polygons());

  auto plain_operand = tf::test::make_tagged_operand(box);
  auto plain_graph = tf::test::build_self_csg_graph(plain_operand.form(), {});
  auto plain = tf::test::csg_domains_with_index_map_of(plain_graph);

  auto flapped_operand = tf::test::make_tagged_operand(flapped);
  auto flapped_graph =
      tf::test::build_self_csg_graph(flapped_operand.form(), {});
  auto with_flap = tf::test::csg_domains_with_index_map_of(flapped_graph);

  membership_require_rows(
      membership_rows_of(std::get<0>(plain), std::get<2>(plain)),
      {{membership_real_t(8), {1}}});
  membership_require_rows(
      membership_rows_of(std::get<0>(with_flap), std::get<2>(with_flap)),
      {{membership_real_t(8), {1}}});
}

TEST_CASE("csg domains: an opposing coincident pair encloses nothing",
          "[domains][membership][within][coincident]") {
  // A solid beside its own reversed copy: every wall carries both layers,
  // they cancel, and the interior is inside no form.
  auto box = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  auto pair = tf::concatenated(box.polygons(),
                               membership_reversed(box).polygons());
  auto operand = tf::test::make_tagged_operand(pair);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(graph);
  membership_require_rows(membership_rows_of(cells, imap),
                          {{membership_real_t(8), {0}}});
}

TEST_CASE("csg domains: an aligned coincident pair encloses twice",
          "[domains][membership][within][coincident]") {
  auto box = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  auto pair = tf::concatenated(box.polygons(), box.polygons());
  auto operand = tf::test::make_tagged_operand(pair);
  auto graph = tf::test::build_self_csg_graph(operand.form(), {});

  auto [cells, ids, imap] = tf::test::csg_domains_with_index_map_of(graph);
  membership_require_rows(membership_rows_of(cells, imap),
                          {{membership_real_t(8), {1}}});
}

TEST_CASE("csg domains: an operand with no faces reads empty",
          "[domains][membership]") {
  membership_mesh_t nothing;
  auto empty_operand = tf::test::make_tagged_operand(nothing);
  auto empty_graph = tf::test::build_self_csg_graph(empty_operand.form(), {});

  auto [cells, ids, imap] =
      tf::test::csg_domains_with_index_map_of(empty_graph);
  REQUIRE(cells.size() == 0);
  REQUIRE(ids.size() == 0);
  REQUIRE(imap.n_tags == membership_index_t(1));
  REQUIRE(imap.inclusion.block_size() == std::size_t(1));
  REQUIRE(tf::test::csg_domains_of(empty_graph).first.size() == 0);

  auto box = tf::make_box_mesh<membership_index_t>(
      membership_real_t(2), membership_real_t(2), membership_real_t(2));
  std::vector<membership_operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(box));
  operands.push_back(tf::test::make_tagged_operand(membership_mesh_t{}));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});

  auto mixed = tf::test::csg_domains_with_index_map_of(graph);
  membership_require_rows(
      membership_rows_of(std::get<0>(mixed), std::get<2>(mixed)),
      {{membership_real_t(8), {1, 0}}});
}

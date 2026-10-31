/**
 * @file test_csg_open_fragments.cpp
 * @brief The open-fragment read: the surface every domain leaves out, handed
 *        to a later build that closes it.
 *
 * A box and an open bowl whose bottom stands inside it: the bowl's walls
 * above the box bound nothing yet, so no cell carries them. A later lid
 * across the bowl closes a pocket between the box and the lid. Handed the
 * cells alone, the later build cannot see the pocket; handed the cells and
 * the fragments, it states every domain the one-shot build states, at its
 * volume. The fragment keeps its form's declaration, volume or sheet.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/core/area.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/signed_volume.hpp>
#include <trueform/topology/is_closed.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

/// An axis-aligned box from `lo` to `hi`, wound outward; `open_top` drops
/// its z+ side.
template <typename Index, typename Real>
auto open_fragments_box(const std::array<Real, 3> &lo,
                        const std::array<Real, 3> &hi, bool open_top)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  for (int v = 0; v < 8; ++v)
    mesh.points_buffer().emplace_back(
        v & 1 ? hi[0] : lo[0], v & 2 ? hi[1] : lo[1], v & 4 ? hi[2] : lo[2]);
  constexpr std::array<std::array<int, 3>, 12> sides = {
      {{0, 4, 6}, {0, 6, 2}, {1, 3, 7}, {1, 7, 5}, {0, 1, 5}, {0, 5, 4},
       {2, 6, 7}, {2, 7, 3}, {0, 2, 3}, {0, 3, 1}, {4, 5, 7}, {4, 7, 6}}};
  for (std::size_t s = 0; s < (open_top ? 10 : 12); ++s)
    mesh.faces_buffer().emplace_back(Index(sides[s][0]), Index(sides[s][1]),
                                     Index(sides[s][2]));
  return mesh;
}

/// The domains of one build over `meshes`, as sorted volumes; `failed`
/// receives the planes the build left without a product.
template <typename Index, typename Real>
auto open_fragments_domain_volumes(
    const std::vector<tf::polygons_buffer<Index, Real, 3, 3>> &meshes,
    const std::vector<int> &sheets, std::size_t &failed)
    -> std::vector<double> {
  std::vector<tf::test::tagged_operand<Index, Real>> operands;
  operands.reserve(meshes.size());
  for (const auto &mesh : meshes)
    operands.emplace_back(mesh);
  auto forms = tf::test::tagged_forms(operands);
  const auto graph = tf::test::build_range_csg_graph(
      tf::test::forms_range(forms), tf::test::sheets_of(sheets), {});
  failed = graph.failed().size();
  auto [cells, ids] = tf::test::csg_domains_of(graph);
  std::vector<double> volumes;
  for (const auto &cell : cells)
    volumes.push_back(double(tf::signed_volume(cell.polygons())));
  std::sort(volumes.begin(), volumes.end());
  return volumes;
}

template <typename Index, typename Real>
void check_open_fragment_hand_off(bool bowl_is_sheet) {
  using Catch::Matchers::WithinAbs;
  using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;
  const mesh_t box =
      open_fragments_box<Index, Real>({0, 0, 0}, {4, 4, 2}, false);
  const mesh_t bowl =
      open_fragments_box<Index, Real>({1, 1, 1}, {3, 3, 5}, true);
  const mesh_t lid =
      open_fragments_box<Index, Real>({0.5, 0.5, 3.5}, {3.5, 3.5, 4}, false);
  const std::vector<int> bowl_sheet =
      bowl_is_sheet ? std::vector<int>{1} : std::vector<int>{};

  std::size_t failed = 0;
  const auto oracle = open_fragments_domain_volumes<Index, Real>(
      {box, bowl, lid}, bowl_sheet, failed);
  REQUIRE(failed == 0);
  // the box outside the bowl, the box inside it, the lid inside and outside
  // the bowl's walls, and the pocket between the box and the lid
  const std::vector<double> expected{2, 2.5, 4, 6, 28};
  REQUIRE(oracle.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    REQUIRE_THAT(oracle[i], WithinAbs(expected[i], 1e-4));

  std::vector<tf::test::tagged_operand<Index, Real>> step1_operands;
  step1_operands.emplace_back(box);
  step1_operands.emplace_back(bowl);
  auto step1_forms = tf::test::tagged_forms(step1_operands);
  const auto g1 = tf::test::build_range_csg_graph(
      tf::test::forms_range(step1_forms), tf::test::sheets_of(bowl_sheet), {});
  REQUIRE(g1.failed().size() == 0);
  auto [cells, cell_ids] = tf::test::csg_domains_of(g1);
  auto [fragments, fragment_tags] = tf::test::csg_open_fragments_of(g1);
  REQUIRE(cells.size() == 2);

  // one fragment per form: the box bounds everything it carries, the bowl's
  // walls above the box bound nothing
  REQUIRE(fragments.size() == 2);
  REQUIRE(fragment_tags.size() == 2);
  REQUIRE(fragment_tags[0] == Index(0));
  REQUIRE(fragment_tags[1] == Index(1));
  REQUIRE(fragments[0].size() == 0);
  REQUIRE(fragments[1].size() > 0);
  REQUIRE_FALSE(tf::is_closed(fragments[1].polygons()));
  double fragment_area = 0;
  for (const auto &face : fragments[1].polygons())
    fragment_area += double(tf::area(face));
  REQUIRE_THAT(fragment_area, WithinAbs(4 * 2 * 3, 1e-4));
  for (const auto &point : fragments[1].points())
    REQUIRE(double(point[2]) >= 2);

  std::vector<mesh_t> cells_only(cells.begin(), cells.end());
  cells_only.push_back(lid);
  const auto without =
      open_fragments_domain_volumes<Index, Real>(cells_only, {}, failed);
  REQUIRE(failed == 0);
  REQUIRE(std::none_of(without.begin(), without.end(),
                       [](double v) { return std::abs(v - 6) < 1e-4; }));

  std::vector<mesh_t> with_fragment(cells.begin(), cells.end());
  with_fragment.push_back(fragments[1]);
  with_fragment.push_back(lid);
  const std::vector<int> fragment_sheet =
      bowl_is_sheet ? std::vector<int>{int(cells.size())} : std::vector<int>{};
  const auto with = open_fragments_domain_volumes<Index, Real>(
      with_fragment, fragment_sheet, failed);
  REQUIRE(failed == 0);
  REQUIRE(with.size() == oracle.size());
  for (std::size_t i = 0; i < oracle.size(); ++i)
    REQUIRE_THAT(with[i], WithinAbs(oracle[i], 1e-4));
}

} // namespace

TEMPLATE_TEST_CASE("open fragments: a bowl a later lid closes is handed over",
                   "[csg][domains][open_fragments]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  SECTION("volume bowl") {
    check_open_fragment_hand_off<index_t, real_t>(false);
  }
  SECTION("sheet bowl") { check_open_fragment_hand_off<index_t, real_t>(true); }
}

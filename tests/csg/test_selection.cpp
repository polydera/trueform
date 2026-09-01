/**
 * @file test_selection.cpp
 * @brief tf::csg::selection — the provenance selection, standalone and
 *        as a restriction on a boolean expression.
 *
 * Standalone, selection({i}) is the embedded read: form i's entire
 * surface with every intersection imprinted, original winding, no
 * classification — for a closed form it stays closed and keeps its
 * volume. As a restriction it splits one boolean result by provenance:
 * the per-form parts partition the unrestricted result's faces, and
 * their signed volumes sum to the solid's.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <trueform/csg.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/topology/is_manifold.hpp>
#include <trueform/trueform.hpp>

#include "csg_builders.hpp"
#include "csg_readers.hpp"
#include "tagged_operand.hpp"

#include "type_traits.hpp"

#include <utility>
#include <vector>

namespace {

template <typename Real> auto identity_frame() -> tf::frame<Real, 3> {
  return tf::make_frame(
      tf::make_transformation_from_translation(tf::vector<Real, 3>{0, 0, 0}));
}

template <typename Operand> struct two_forms {
  using form_type = decltype(std::declval<Operand &>().form());

  std::vector<Operand> operands;
  std::vector<form_type> forms;
  decltype(tf::test::build_range_csg_graph(
      tf::test::forms_range(std::declval<std::vector<form_type> &>()),
      tf::test::no_sheets(), tf::arrangement_config{})) graph;

  two_forms(std::vector<Operand> ops, tf::arrangement_config config)
      : operands(std::move(ops)), forms(tf::test::tagged_forms(operands)),
        graph(tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                              tf::test::no_sheets(), config)) {}

  explicit two_forms(std::vector<Operand> ops)
      : two_forms(std::move(ops), tf::arrangement_config{}) {}

  two_forms(const two_forms &) = delete;
  two_forms &operator=(const two_forms &) = delete;
};

/// Two 2x2x2 boxes, B shifted by (1, 0.5, 0.25) — a generic overlap, no
/// coplanar walls (coincident walls dedup under their stack's owner tag,
/// which is its own scenario, not this one). Overlap = 1 * 1.5 * 1.75 =
/// 2.625; vol(A) = vol(B) = 8; A - B = 5.375; union = 13.375. All dyadic,
/// exact in double.
template <typename Index, typename Real> auto overlapping_boxes() {
  auto a = tf::make_box_mesh<Index, Real>(Real(2), Real(2), Real(2));
  auto b = tf::make_box_mesh<Index, Real>(Real(2), Real(2), Real(2));
  auto shift = tf::make_frame(tf::make_transformation_from_translation(
      tf::vector<Real, 3>{Real(1), Real(0.5), Real(0.25)}));
  using operand_t = decltype(tf::test::make_tagged_operand(a));
  std::vector<operand_t> operands;
  operands.reserve(2);
  operands.push_back(tf::test::make_tagged_operand(a));
  operands.push_back(tf::test::make_tagged_operand(
      b, tf::transformation<Real, 3>(shift.transformation())));
  struct fixture {
    decltype(a) mesh_a, mesh_b;
    two_forms<operand_t> tf2;
  };
  return fixture{std::move(a), std::move(b),
                 two_forms<operand_t>(std::move(operands))};
}

template <typename Polygons> auto n_boundary(const Polygons &polygons) {
  return std::size_t(tf::make_boundary_edges(polygons).size());
}

} // namespace

TEMPLATE_TEST_CASE("selection: standalone is the embedded read",
                   "[csg][selection]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  auto fx = overlapping_boxes<index_t, real_t>();

  auto sa = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection({0}));
  REQUIRE(tf::is_closed(sa.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(sa.polygons())),
               Catch::Matchers::WithinAbs(8.0, 1e-9));
  // cut by B: more faces than the input box
  REQUIRE(sa.polygons().size() > fx.mesh_a.polygons().size());

  auto sb = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection({1}));
  REQUIRE(tf::is_closed(sb.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(sb.polygons())),
               Catch::Matchers::WithinAbs(8.0, 1e-9));

  // both surfaces = the full arrangement mesh, face for face
  auto both = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection({0, 1}));
  auto full = tf::make_csg_mesh(fx.tf2.graph);
  REQUIRE(both.polygons().size() == full.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(both.polygons())),
               Catch::Matchers::WithinAbs(16.0, 1e-9));
}

TEMPLATE_TEST_CASE("selection: a boolean splits by provenance",
                   "[csg][selection]",
                   (tf::test::type_pair<std::int32_t, double>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  auto fx = overlapping_boxes<index_t, real_t>();

  const auto e = tf::csg::difference(0, 1);
  auto full = tf::test::csg_mesh_of(fx.tf2.graph, e);
  REQUIRE(tf::is_closed(full.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(full.polygons())),
               Catch::Matchers::WithinAbs(5.375, 1e-9));

  auto part_a = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection(0, e));
  auto part_b = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection(1, e));

  // the parts partition the result's faces...
  REQUIRE(part_a.polygons().size() + part_b.polygons().size() ==
          full.polygons().size());
  // ...are individually open along the provenance seam...
  REQUIRE(n_boundary(part_a.polygons()) > 0);
  REQUIRE(n_boundary(part_b.polygons()) > 0);
  // ...and concatenated they re-form the solid: each part compacts its
  // own points, so the seam arrives as coincident duplicates and welds.
  auto recomposed = tf::cleaned(
      tf::concatenated(part_a.polygons(), part_b.polygons()).polygons());
  REQUIRE(tf::is_closed(recomposed.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(recomposed.polygons())),
               Catch::Matchers::WithinAbs(5.375, 1e-9));

  // restricting to every operand is no restriction at all
  auto all = tf::test::csg_mesh_of(fx.tf2.graph, tf::csg::selection({0, 1}, e));
  REQUIRE(all.polygons().size() == full.polygons().size());
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(all.polygons())),
               Catch::Matchers::WithinAbs(5.375, 1e-9));
}

TEMPLATE_TEST_CASE("selection: domains split by provenance the same way",
                   "[csg][selection]",
                   (tf::test::type_pair<std::int32_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;
  auto fx = overlapping_boxes<index_t, real_t>();

  const auto e = tf::csg::difference(0, 1);
  auto [cells, ids] = tf::test::csg_domains_of(fx.tf2.graph, e);
  auto [cells_a, ids_a] =
      tf::test::csg_domains_of(fx.tf2.graph, tf::csg::selection(0, e));
  auto [cells_b, ids_b] =
      tf::test::csg_domains_of(fx.tf2.graph, tf::csg::selection(1, e));
  REQUIRE(cells.size() == 1u);
  REQUIRE(cells_a.size() == 1u);
  REQUIRE(cells_b.size() == 1u);
  REQUIRE(ids_a[0] == ids[0]);
  REQUIRE(ids_b[0] == ids[0]);
  // the cell's walls partition by contributor and re-weld into the cell
  REQUIRE(cells_a[0].polygons().size() + cells_b[0].polygons().size() ==
          cells[0].polygons().size());
  REQUIRE(n_boundary(cells_a[0].polygons()) > 0);
  REQUIRE(n_boundary(cells_b[0].polygons()) > 0);
  auto recomposed = tf::cleaned(
      tf::concatenated(cells_a[0].polygons(), cells_b[0].polygons())
          .polygons());
  REQUIRE(tf::is_closed(recomposed.polygons()));
  REQUIRE_THAT(static_cast<double>(tf::signed_volume(recomposed.polygons())),
               Catch::Matchers::WithinAbs(5.375, 1e-9));
}

/**
 * @file test_csg_component_names.cpp
 * @brief A component's name is its rank by its first member
 *
 * The component floods name what they reach in arrival order, which no two
 * schedules share; the published name is the component's rank by its first
 * member — its smallest exposed-triangle row, or, for a component of uncut
 * faces alone, its first such face after every row, tag-major. So along the
 * exposed stream the names first appear in ascending order, and every
 * component that holds no exposed triangle ranks after every one that does,
 * in the order its first uncut face stands.
 *
 * Two overlapping boxes give the cut components; a third, apart, is uncut.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "csg_builders.hpp"
#include "tagged_operand.hpp"

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/geometry/make_box_mesh.hpp>

#include <cstddef>
#include <vector>

namespace {

using names_index_t = int;
using names_real_t = float;
using names_mesh_t = tf::polygons_buffer<names_index_t, names_real_t, 3, 3>;

auto names_box(names_real_t dx, names_real_t dy, names_real_t dz)
    -> names_mesh_t {
  auto mesh = tf::make_box_mesh<names_index_t>(
      names_real_t(1), names_real_t(1), names_real_t(1));
  for (std::size_t i = 0; i < mesh.points_buffer().size(); ++i) {
    mesh.points_buffer()[i][0] += dx;
    mesh.points_buffer()[i][1] += dy;
    mesh.points_buffer()[i][2] += dz;
  }
  return mesh;
}

} // namespace

TEST_CASE("csg component names: a name is the rank of its first member",
          "[csg][graph][labels]") {
  std::vector<tf::test::tagged_operand<names_index_t, names_real_t>> operands;
  operands.reserve(3);
  operands.emplace_back(names_box(names_real_t(0), names_real_t(0),
                                  names_real_t(0)));
  operands.emplace_back(names_box(names_real_t(0.5), names_real_t(0.25),
                                  names_real_t(0.3)));
  operands.emplace_back(names_box(names_real_t(5), names_real_t(5),
                                  names_real_t(5)));
  auto forms = tf::test::tagged_forms(operands);
  auto graph = tf::test::build_range_csg_graph(tf::test::forms_range(forms),
                                               tf::test::no_sheets(), {});
  const auto &labels = graph.labels();
  const auto none = names_index_t(-1);
  const auto n_components = labels.n_components();

  // along the exposed stream every name first appears as the next one
  names_index_t next = 0;
  for (const auto label : labels.triangle_labels()) {
    if (label == none)
      continue;
    REQUIRE(label <= next);
    if (label == next)
      ++next;
  }
  const auto n_cut = next;
  REQUIRE(n_cut > names_index_t(0));

  // the components the stream never names follow, in tag-major face order
  for (names_index_t tag = 0; tag < names_index_t(3); ++tag)
    for (const auto label : labels.polygon_labels(tag)) {
      if (label == none || label < n_cut)
        continue;
      REQUIRE(label <= next);
      if (label == next)
        ++next;
    }
  CHECK(next == n_components);
  // the third box is apart from both, so it is one uncut component
  CHECK(n_components > n_cut);
}

/**
 * @file test_plane_crossings.cpp
 * @brief The plane graph's junction detection on carriers that have no plane
 *
 * A carrier is stated by the points it stands on, and when those are collinear
 * it IS a line: no plane pair cuts its edges out, and the direction every edge
 * on it runs along is the line's own. The scenes here put such a carrier on
 * both sides of the question — as the plane whose edges are being ordered, and
 * as the partner whose trace states another plane's edge.
 *
 * The census is the surface: `undecided_order` counts the aligned pairs
 * detection could not order, so zero is the claim that every one of them was
 * decided. The scenes state aligned pairs (`collinear_pairs` is the premise
 * the claim needs) and the junctions they name reach the split tables.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "tagged_operand.hpp"
#include "input_lattice_for.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/range.hpp>
#include <trueform/exact/int32.hpp>
#include <trueform/exact/int64.hpp>
#include <trueform/intersect/graph/local_arrangement.hpp>
#include <trueform/intersect/graph/plane_census.hpp>
#include <trueform/intersect/intersect_config.hpp>
#include <trueform/intersect/intersect_mode.hpp>
#include <trueform/intersect/polygon_intersections.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace {

using crossings_index_t = int;

template <typename Int> struct real_of;
template <> struct real_of<tf::exact::int32> {
  using type = float;
};
template <> struct real_of<tf::exact::int64> {
  using type = double;
};

template <typename Real>
using crossings_mesh_t = tf::polygons_buffer<crossings_index_t, Real, 3, 3>;

template <typename Real>
auto triangle(std::array<std::array<Real, 3>, 3> corners)
    -> crossings_mesh_t<Real> {
  crossings_mesh_t<Real> mesh;
  for (const auto &corner : corners)
    mesh.points_buffer().emplace_back(corner[0], corner[1], corner[2]);
  mesh.faces_buffer().emplace_back(0, 1, 2);
  return mesh;
}

/// What one scene's local arrangement counted, plus what it split.
struct detection_facts_t {
  std::size_t collinear_pairs = 0;
  std::size_t undecided_order = 0;
  std::size_t crossings = 0;
  std::size_t split_roots = 0;
  crossings_index_t line_planes = 0;
};

template <typename Int, typename Real>
auto measure(std::vector<crossings_mesh_t<Real>> meshes) -> detection_facts_t {
  std::vector<
      std::unique_ptr<tf::test::tagged_operand<crossings_index_t, Real>>>
      scenes;
  for (auto &mesh : meshes)
    scenes.push_back(
        std::make_unique<tf::test::tagged_operand<crossings_index_t, Real>>(
            std::move(mesh)));
  std::vector<decltype(scenes[0]->form())> forms;
  for (auto &scene : scenes)
    forms.push_back(scene->form());

  const auto apply_to_face = [&](int tag, crossings_index_t object,
                                 const auto &apply) {
    apply(forms[std::size_t(tag)].faces()[object]);
  };
  const auto apply_to_form = [&](crossings_index_t tag, const auto &apply) {
    apply(forms[std::size_t(tag)]);
  };
  tf::buffer<crossings_index_t> face_offsets;
  face_offsets.allocate(forms.size() + 1);
  face_offsets[0] = 0;
  for (std::size_t tag = 0; tag < forms.size(); ++tag)
    face_offsets[tag + 1] =
        face_offsets[tag] + crossings_index_t(forms[tag].faces().size());

  tf::polygon_intersections<crossings_index_t, Real, Int> intersections;
  intersections.with_edge_splits(false);
  const auto intersections_lattice = tf::test::input_lattice_for(tf::make_range(forms.data(), forms.data() + forms.size()), 0.0);
  intersections.build(tf::make_range(forms.data(), forms.data() + forms.size()), intersections_lattice, tf::intersect_config{tf::intersect_mode::primitives |
                               tf::intersect_mode::resolve_crossing_contours,
                           0.0});
  const auto converter = intersections_lattice.converter();
  const auto get_mesh_point = [&](int tag,
                                  crossings_index_t id) -> tf::point<Int, 3> {
    return converter.convert(forms[std::size_t(tag)].points()[id]);
  };
  tf::intersect::graph::local_arrangement<crossings_index_t, Real, Int> world;
  world.build(std::move(intersections), get_mesh_point, apply_to_face,
              apply_to_form, tf::make_range(face_offsets), false, false);

  detection_facts_t facts;
  const auto &census = world.census();
  facts.collinear_pairs = census.collinear_pairs;
  facts.undecided_order = census.undecided_order;
  facts.crossings = census.crossings_ve + census.crossings_ee;
  facts.split_roots = world.split_roots().size();
  for (crossings_index_t plane = 0; plane < world.n_cut_planes(); ++plane) {
    const auto &normal = world.graph().frame(plane).plane_n;
    using normal_t = std::decay_t<decltype(normal[0])>;
    if (normal[0] == normal_t(0) && normal[1] == normal_t(0) &&
        normal[2] == normal_t(0))
      ++facts.line_planes;
  }
  return facts;
}

} // namespace

TEMPLATE_TEST_CASE("plane crossings: every aligned pair on a line is ordered",
                   "[intersect][graph]", tf::exact::int32, tf::exact::int64) {
  using Int = TestType;
  using Real = typename real_of<Int>::type;

  SECTION("two line carriers overlapping inside one area face") {
    // both faces collapse to one line, they overlap on it, and both lie in
    // the third face's plane — so that plane's two interior cuts are stated
    // by partners that have no plane of their own
    const auto facts = measure<Int, Real>(
        {triangle<Real>({{{{Real(-20), Real(0), Real(0)}},
                          {{Real(-10), Real(0), Real(0)}},
                          {{Real(0), Real(0), Real(0)}}}}),
         triangle<Real>({{{{Real(-10), Real(0), Real(0)}},
                          {{Real(5), Real(0), Real(0)}},
                          {{Real(20), Real(0), Real(0)}}}}),
         triangle<Real>({{{{Real(-30), Real(0), Real(-10)}},
                          {{Real(30), Real(0), Real(-10)}},
                          {{Real(0), Real(0), Real(10)}}}})});
    REQUIRE(facts.line_planes == 2);
    REQUIRE(facts.collinear_pairs != 0);
    CHECK(facts.undecided_order == 0);
    CHECK(facts.crossings != 0);
    CHECK(facts.split_roots != 0);
  }

  SECTION("a line carrier inside two coplanar cutters") {
    // the line carrier's own edges are two cutter traces plus its own sides,
    // and the cutters pool into one plane whose interior cuts are stated
    // against a partner with no plane
    const auto facts = measure<Int, Real>(
        {triangle<Real>({{{{Real(-20), Real(0), Real(0)}},
                          {{Real(0), Real(0), Real(0)}},
                          {{Real(20), Real(0), Real(0)}}}}),
         triangle<Real>({{{{Real(-10), Real(0), Real(-5)}},
                          {{Real(10), Real(0), Real(-5)}},
                          {{Real(0), Real(0), Real(5)}}}}),
         triangle<Real>({{{{Real(-5), Real(0), Real(-5)}},
                          {{Real(15), Real(0), Real(-5)}},
                          {{Real(5), Real(0), Real(5)}}}})});
    REQUIRE(facts.line_planes == 1);
    REQUIRE(facts.collinear_pairs != 0);
    CHECK(facts.undecided_order == 0);
  }
}

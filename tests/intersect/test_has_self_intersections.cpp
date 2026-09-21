/**
 * @file test_has_self_intersections.cpp
 * @brief tf::has_self_intersections against the producer it answers for.
 *
 * The predicate's contract IS the agreement: true exactly where
 * tf::polygon_intersections' one-form build states a self record. Every
 * case asserts the agreement AND the answer, so a battery that stopped
 * discriminating cannot pass.
 *
 * Structures are completed and never required, so every case is asked
 * twice: of the bare mesh, which builds them, and of the fully tagged
 * operand the producer itself reads.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include "input_lattice_for.hpp"
#include "tagged_operand.hpp"
#include "type_traits.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <trueform/core/none.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/exact/resolve_int_type.hpp>
#include <trueform/geometry/make_box_mesh.hpp>
#include <trueform/geometry/make_sphere_mesh.hpp>
#include <trueform/intersect/has_self_intersections.hpp>
#include <trueform/intersect/polygon_intersections.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/spatial/policy/tree.hpp>
#include <trueform/spatial/tree_config.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename Index, typename Real>
auto self_query_mesh(const std::vector<std::array<double, 3>> &points,
                     const std::vector<std::array<int, 3>> &faces)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  mesh.points_buffer().allocate(points.size());
  mesh.faces_buffer().allocate(faces.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.points()[i][d] = Real(points[i][d]);
  for (std::size_t i = 0; i < faces.size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.faces()[i][d] = Index(faces[i][d]);
  return mesh;
}

/// Two meshes read as one, so a self question is asked of interpenetrating
/// components rather than of two operands.
template <typename Index, typename Real>
auto self_query_merged(const tf::polygons_buffer<Index, Real, 3, 3> &a,
                       const tf::polygons_buffer<Index, Real, 3, 3> &b,
                       Real shift) -> tf::polygons_buffer<Index, Real, 3, 3> {
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  const auto base = Index(a.points().size());
  mesh.points_buffer().allocate(a.points().size() + b.points().size());
  mesh.faces_buffer().allocate(a.faces().size() + b.faces().size());
  for (std::size_t i = 0; i < a.points().size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.points()[i][d] = a.points()[i][d];
  for (std::size_t i = 0; i < b.points().size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.points()[a.points().size() + i][d] =
          b.points()[i][d] + (d == 0 ? shift : Real(0));
  for (std::size_t i = 0; i < a.faces().size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.faces()[i][d] = a.faces()[i][d];
  for (std::size_t i = 0; i < b.faces().size(); ++i)
    for (std::size_t d = 0; d < 3; ++d)
      mesh.faces()[a.faces().size() + i][d] = base + b.faces()[i][d];
  return mesh;
}

/// The fact the predicate answers for: does the one-form build state a
/// self record at all.
template <typename Index, typename Real, typename Form>
auto self_query_records(const Form &form) -> bool {
  using int_t = tf::exact::resolve_int_type<tf::none_t, Real>;
  const auto lattice = tf::test::input_lattice_for(form, 0.0);
  tf::polygon_intersections<Index, Real, int_t> intersections;
  intersections.build(form, lattice);
  return intersections.flat_intersections().size() != 0;
}

template <typename Index, typename Real> struct self_query_case {
  std::string name;
  tf::polygons_buffer<Index, Real, 3, 3> mesh;
  bool meets_itself;
};

template <typename Index, typename Real>
auto self_query_battery() -> std::vector<self_query_case<Index, Real>> {
  auto mesh = [](const std::vector<std::array<double, 3>> &points,
                 const std::vector<std::array<int, 3>> &faces) {
    return self_query_mesh<Index, Real>(points, faces);
  };
  std::vector<self_query_case<Index, Real>> cases;
  cases.push_back({"clean box",
                   tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)), false});
  cases.push_back(
      {"clean sphere", tf::make_sphere_mesh<Index>(Real(1), 16, 24), false});
  cases.push_back(
      {"interpenetrating boxes",
       self_query_merged<Index, Real>(
           tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
           tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)), Real(1)),
       true});
  cases.push_back(
      {"disjoint boxes",
       self_query_merged<Index, Real>(
           tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)),
           tf::make_box_mesh<Index>(Real(2), Real(2), Real(2)), Real(8)),
       false});
  cases.push_back({"crossing faces",
                   mesh({{-1, -1, 0},
                         {1, -1, 0},
                         {0, 1, 0},
                         {0, -0.5, -1},
                         {0, -0.5, 1},
                         {0, 0.75, 0}},
                        {{0, 1, 2}, {3, 4, 5}}),
                   true});
  cases.push_back({"shared edge fold",
                   mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
                        {{0, 1, 2}, {1, 0, 3}}),
                   false});
  cases.push_back(
      {"shared vertex only",
       mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 1}, {-1, 0, 1}},
            {{0, 1, 2}, {0, 3, 4}}),
       false});
  cases.push_back(
      {"coplanar overlap",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {1, 1, 0}, {5, 1, 0}, {1, 5, 0}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back(
      {"degenerate face on a face",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {1, 1, 0}, {2, 2, 0}, {3, 3, 0}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back({"open strip",
                   mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {2, 0, 0}},
                        {{0, 1, 2}, {1, 3, 2}, {1, 4, 3}}),
                   false});
  cases.push_back(
      {"split seam soup",
       mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 1}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back(
      {"vertex inside a face",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {1, 1, 0}, {1, 1, 2}, {2, 1, 2}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back(
      {"vertex inside an edge",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {2, 0, 0}, {2, 0, 2}, {3, 0, 2}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  // A degenerate face is the segment or the point it collapsed to, and the
  // producer states its contacts like any other: its own plane is what the
  // invalid support withholds, never the contact.
  cases.push_back(
      {"sliver pierces a face",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {1, 1, -1}, {1, 1, 1}, {1, 1, -1}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back({"collinear corners pierce a face",
                   mesh({{0, 0, 0},
                         {4, 0, 0},
                         {0, 4, 0},
                         {1, 1, -1},
                         {1, 1, 0.5},
                         {1, 1, 1}},
                        {{0, 1, 2}, {3, 4, 5}}),
                   true});
  cases.push_back({"sliver pierces below its coincident pair",
                   mesh({{0, 0, 0},
                         {4, 0, 0},
                         {0, 4, 0},
                         {1, 1, -1},
                         {1, 1, -1},
                         {1, 1, 1}},
                        {{0, 1, 2}, {3, 4, 5}}),
                   true});
  cases.push_back({"sliver lies on the face's own edge",
                   mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {2, 0, 0}},
                        {{0, 1, 2}, {0, 1, 3}}),
                   true});
  cases.push_back(
      {"point face inside a face",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {1, 1, 0}, {1, 1, 0}, {1, 1, 0}},
            {{0, 1, 2}, {3, 4, 5}}),
       true});
  cases.push_back(
      {"sliver apart",
       mesh({{0, 0, 0}, {4, 0, 0}, {0, 4, 0}, {9, 9, -1}, {9, 9, 1}, {9, 9, -1}},
            {{0, 1, 2}, {3, 4, 5}}),
       false});
  cases.push_back({"crossing faces beside a sliver",
                   mesh({{-1, -1, 0},
                         {1, -1, 0},
                         {0, 1, 0},
                         {0, -0.5, -1},
                         {0, -0.5, 1},
                         {0, 0.75, 0},
                         {9, 9, -1},
                         {9, 9, 1},
                         {9, 9, -1}},
                        {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}),
                   true});
  // The election is what keeps these silent: the feature is elected to the
  // face on the other side of the very pair that would state it.
  cases.push_back({"coincident vertices on a shared edge",
                   mesh({{2, 1, 1},
                         {1, 0, 2},
                         {1, 2, 0},
                         {1, 2, 0},
                         {2, 2, 1},
                         {0, 2, 0}},
                        {{5, 2, 0}, {0, 3, 2}}),
                   false});
  cases.push_back({"coincident vertices off a shared edge",
                   mesh({{2, 0, 2},
                         {2, 0, 2},
                         {0, 2, 1},
                         {1, 2, 0},
                         {2, 2, 2},
                         {0, 1, 1}},
                        {{0, 5, 3}, {0, 5, 1}}),
                   false});
  cases.push_back({"collinear face along a shared edge",
                   mesh({{0, 0, 0}, {0, 2, 0}, {1, 1, 0}, {2, 0, 0}},
                        {{1, 3, 2}, {1, 3, 0}}),
                   false});
  cases.push_back({"faces apart",
                   mesh({{0, 0, 0},
                         {1, 0, 0},
                         {0, 1, 0},
                         {10, 10, 10},
                         {11, 10, 10},
                         {10, 11, 10}},
                        {{0, 1, 2}, {3, 4, 5}}),
                   false});
  cases.push_back({"one face",
                   mesh({{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, {{0, 1, 2}}),
                   false});
  return cases;
}

} // namespace

TEMPLATE_TEST_CASE("has_self_intersections answers what the records state",
                   "[intersect][self_intersection]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  for (auto &state : self_query_battery<index_t, real_t>()) {
    INFO(state.name);
    auto operand = tf::test::make_tagged_operand(std::move(state.mesh));
    const auto records = self_query_records<index_t, real_t>(operand.form());
    CHECK(records == state.meets_itself);
    CHECK(tf::has_self_intersections(operand.form()) == state.meets_itself);
    CHECK(tf::has_self_intersections(operand.mesh.polygons()) ==
          state.meets_itself);
  }
}

TEMPLATE_TEST_CASE("has_self_intersections reads every entry shape alike",
                   "[intersect][self_intersection]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
  using index_t = typename TestType::index_type;
  using real_t = typename TestType::real_type;

  auto crossing = self_query_mesh<index_t, real_t>({{-1, -1, 0},
                                                    {1, -1, 0},
                                                    {0, 1, 0},
                                                    {0, -0.5, -1},
                                                    {0, -0.5, 1},
                                                    {0, 0.75, 0}},
                                                   {{0, 1, 2}, {3, 4, 5}});
  auto clean = tf::make_sphere_mesh<index_t>(real_t(1), 12, 16);

  tf::aabb_tree<index_t, real_t, 3> crossing_tree;
  crossing_tree.build(crossing.polygons(), tf::config_tree(4, 12));
  tf::aabb_tree<index_t, real_t, 3> clean_tree;
  clean_tree.build(clean.polygons(), tf::config_tree(4, 12));

  // a bare mesh builds all three, a tree-tagged one builds the topology
  // beside it, and a tagged operand builds nothing
  CHECK(tf::has_self_intersections(crossing.polygons()));
  CHECK(
      tf::has_self_intersections(crossing.polygons() | tf::tag(crossing_tree)));
  CHECK(!tf::has_self_intersections(clean.polygons()));
  CHECK(!tf::has_self_intersections(clean.polygons() | tf::tag(clean_tree)));

  auto crossing_operand = tf::test::make_tagged_operand(std::move(crossing));
  auto clean_operand = tf::test::make_tagged_operand(std::move(clean));
  CHECK(tf::has_self_intersections(crossing_operand.form()));
  CHECK(!tf::has_self_intersections(clean_operand.form()));

  tf::polygons_buffer<index_t, real_t, 3, 3> empty;
  CHECK(!tf::has_self_intersections(empty.polygons()));
}

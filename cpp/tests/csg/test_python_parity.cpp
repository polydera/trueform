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
#include "carriers.hpp"
#include "mixed_mesh.hpp"
#include "trueform/cpp/csg.hpp"
#include "trueform/cpp/csg/outer_shell.hpp"
#include "trueform/cpp/geometry/make_box_mesh.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"
#include "trueform/cpp/geometry/signed_volume.hpp"
#include "trueform/cpp/reindex/concatenated.hpp"
#include "trueform/cpp/topology/is_closed.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <typename T>
auto array(std::initializer_list<T> values, tf::small_vector<int, 3> shape)
    -> tf::cpp::nd_array<T> {
  tf::buffer<T> storage;
  storage.allocate(values.size());
  std::copy(values.begin(), values.end(), storage.begin());
  return tf::cpp::nd_array<T>::from_buffer(std::move(storage),
                                           std::move(shape));
}

/// The sixteen numbers a placement is, which is what an owner is placed by.
template <typename Real>
auto translation(Real x, Real y = Real{0}, Real z = Real{0})
    -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1}, Real{0}, y,
          Real{0}, Real{0}, Real{1}, z, Real{0}, Real{0}, Real{0}, Real{1}};
}

template <typename Real>
using parity_owned = tf::cpp::test::owned_mesh<tf::cpp::default_index_t, Real>;

template <typename Real>
using parity_view = tf::cpp::mesh<tf::cpp::default_index_t, Real, 3, 3>;

template <typename Real>
using parity_result = tf::polygons_buffer<tf::cpp::default_index_t, Real, 3, 3>;

/// The views an operand list is, over owners the caller keeps: the vector is
/// final before the first view is taken of it.
template <typename Real>
auto meshes_of(const std::vector<parity_owned<Real>> &owners)
    -> std::vector<parity_view<Real>> {
  std::vector<parity_view<Real>> meshes;
  meshes.reserve(owners.size());
  for (const auto &owner : owners)
    meshes.push_back(owner.mesh());
  return meshes;
}

/// The coordinates a result holds, as the array a check compares them with:
/// the array borrows core's own storage where it lies.
template <typename Real>
auto result_points(const parity_result<Real> &value)
    -> tf::cpp::nd_array<Real> {
  const auto &storage = value.points_buffer().data_buffer();
  const auto count = static_cast<int>(value.points_buffer().size());
  return tf::cpp::nd_array<Real>::from_borrowed(
      {}, const_cast<Real *>(storage.data()), storage.size(), {count, 3});
}

/// A result is core's own storage, so a check reads it as an operand of its
/// own: the cache and the identity placement are what a reading needs.
template <typename Real>
auto operand_of(parity_result<Real> polygons) -> parity_owned<Real> {
  parity_owned<Real> result;
  result.polygons = std::move(polygons);
  return result;
}

template <typename Real>
auto overlapping_unit_boxes() -> std::vector<parity_owned<Real>> {
  std::vector<parity_owned<Real>> result;
  result.reserve(2);
  result.push_back({tf::cpp::make_box_mesh(Real{1}, Real{1}, Real{1})});
  result.push_back(
      {tf::cpp::make_box_mesh(Real{1}, Real{1}, Real{1}, 2, 1, 1)});
  result[0].place(translation(Real{0.1}, Real{-0.2}, Real{0.05}));
  result[1].place(translation(Real{0.5}, Real{0.1}, Real{0.25}));
  return result;
}

template <typename Real>
auto three_overlapping_spheres() -> std::vector<parity_owned<Real>> {
  std::vector<parity_owned<Real>> result;
  result.reserve(3);
  for (int sphere = 0; sphere != 3; ++sphere)
    result.push_back({tf::cpp::make_sphere_mesh(Real{1}, 12, 16)});
  result[1].place(translation(Real{0.5}));
  result[2].place(translation(Real{0}, Real{0.5}, Real{0.1}));
  return result;
}

using point3 = std::array<double, 3>;

struct mesh_geometry {
  std::vector<std::array<std::int32_t, 3>> faces;
  std::vector<point3> points;
};

auto subtract(const point3 &a, const point3 &b) -> point3 {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

auto cross(const point3 &a, const point3 &b) -> point3 {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

auto dot(const point3 &a, const point3 &b) -> double {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename Real>
auto point_at(const parity_result<Real> &value, std::int32_t point) -> point3 {
  const auto coordinates = value.points()[static_cast<std::size_t>(point)];
  return {static_cast<double>(coordinates[0]),
          static_cast<double>(coordinates[1]),
          static_cast<double>(coordinates[2])};
}

template <typename Real>
auto capture_geometry(const parity_owned<Real> &mesh) -> mesh_geometry {
  mesh_geometry geometry;
  const auto &polygons = mesh.polygons;
  const auto faces = polygons.faces_buffer().data_buffer();
  const auto point_count = polygons.points_buffer().size();
  geometry.points.reserve(point_count);
  geometry.faces.reserve(polygons.size());

  const auto transformed = mesh.placement.placed;
  const auto &matrix = mesh.placement.values;
  for (std::size_t point = 0; point != point_count; ++point) {
    const auto local = point_at(polygons, static_cast<std::int32_t>(point));
    if (transformed) {
      geometry.points.push_back(
          {static_cast<double>(matrix[0]) * local[0] +
               static_cast<double>(matrix[1]) * local[1] +
               static_cast<double>(matrix[2]) * local[2] +
               static_cast<double>(matrix[3]),
           static_cast<double>(matrix[4]) * local[0] +
               static_cast<double>(matrix[5]) * local[1] +
               static_cast<double>(matrix[6]) * local[2] +
               static_cast<double>(matrix[7]),
           static_cast<double>(matrix[8]) * local[0] +
               static_cast<double>(matrix[9]) * local[1] +
               static_cast<double>(matrix[10]) * local[2] +
               static_cast<double>(matrix[11])});
    } else {
      geometry.points.push_back(local);
    }
  }
  for (std::size_t face = 0; face != polygons.size(); ++face) {
    const auto offset = face * 3;
    geometry.faces.push_back(
        {faces[offset], faces[offset + 1], faces[offset + 2]});
  }
  return geometry;
}

template <typename Real>
auto capture_geometry(const std::vector<parity_owned<Real>> &meshes)
    -> std::vector<mesh_geometry> {
  std::vector<mesh_geometry> geometry;
  geometry.reserve(meshes.size());
  for (const auto &mesh : meshes)
    geometry.push_back(capture_geometry(mesh));
  return geometry;
}

template <typename Real> constexpr auto geometry_tolerance() -> double {
  return sizeof(Real) == sizeof(float) ? 2e-5 : 1e-10;
}

template <typename Real>
auto check_point_coordinates(const parity_result<Real> &mesh,
                             std::int32_t point, const point3 &expected)
    -> void {
  REQUIRE(point >= 0);
  REQUIRE(static_cast<std::size_t>(point) < mesh.points_buffer().size());
  const auto actual = point_at(mesh, point);
  for (std::size_t axis = 0; axis < 3; ++axis)
    CHECK(actual[axis] ==
          Catch::Approx(expected[axis]).margin(geometry_tolerance<Real>()));
}

template <typename Real>
auto check_face_subdivision(const parity_result<Real> &mesh, std::int32_t face,
                            const std::vector<mesh_geometry> &inputs,
                            std::int32_t tag, std::int32_t source_face)
    -> void {
  REQUIRE(face >= 0);
  REQUIRE(static_cast<std::size_t>(face) < mesh.size());
  REQUIRE(tag >= 0);
  REQUIRE(static_cast<std::size_t>(tag) < inputs.size());
  const auto &input = inputs[static_cast<std::size_t>(tag)];
  REQUIRE(source_face >= 0);
  REQUIRE(static_cast<std::size_t>(source_face) < input.faces.size());

  const auto &source_ids = input.faces[static_cast<std::size_t>(source_face)];
  const auto &a = input.points[static_cast<std::size_t>(source_ids[0])];
  const auto &b = input.points[static_cast<std::size_t>(source_ids[1])];
  const auto &c = input.points[static_cast<std::size_t>(source_ids[2])];
  const auto ab = subtract(b, a);
  const auto ac = subtract(c, a);
  const auto normal = cross(ab, ac);
  const auto normal_squared = dot(normal, normal);
  REQUIRE(normal_squared > 0.0);

  const auto d00 = dot(ab, ab);
  const auto d01 = dot(ab, ac);
  const auto d11 = dot(ac, ac);
  const auto denominator = d00 * d11 - d01 * d01;
  REQUIRE(denominator > 0.0);

  const auto faces = mesh.faces_buffer().data_buffer();
  const auto face_offset = static_cast<std::size_t>(3 * face);
  std::array<point3, 3> output;
  for (std::size_t vertex = 0; vertex < 3; ++vertex) {
    const auto point = faces[face_offset + vertex];
    REQUIRE(point >= 0);
    REQUIRE(static_cast<std::size_t>(point) < mesh.points_buffer().size());
    output[vertex] = point_at(mesh, point);

    const auto ap = subtract(output[vertex], a);
    const auto plane_distance =
        std::abs(dot(ap, normal)) / std::sqrt(normal_squared);
    CHECK(plane_distance <= geometry_tolerance<Real>());

    const auto d20 = dot(ap, ab);
    const auto d21 = dot(ap, ac);
    const auto barycentric_b = (d11 * d20 - d01 * d21) / denominator;
    const auto barycentric_c = (d00 * d21 - d01 * d20) / denominator;
    const auto barycentric_a = 1.0 - barycentric_b - barycentric_c;
    const auto containment_tolerance = 20.0 * geometry_tolerance<Real>();
    CHECK(barycentric_a >= -containment_tolerance);
    CHECK(barycentric_b >= -containment_tolerance);
    CHECK(barycentric_c >= -containment_tolerance);
  }

  const auto output_normal =
      cross(subtract(output[1], output[0]), subtract(output[2], output[0]));
  const auto output_area_twice = std::sqrt(dot(output_normal, output_normal));
  REQUIRE(output_area_twice > geometry_tolerance<Real>());
  CHECK(output_area_twice <=
        std::sqrt(normal_squared) + geometry_tolerance<Real>());
}

template <typename Offsets, typename Data>
auto check_offset_blocks(const Offsets &offsets, const Data &data,
                         std::size_t number_of_blocks) -> void {
  REQUIRE(offsets.length() == number_of_blocks + 1);
  REQUIRE(offsets[0] == 0);
  for (std::size_t block = 0; block < number_of_blocks; ++block) {
    REQUIRE(offsets[block] >= 0);
    REQUIRE(offsets[block + 1] >= offsets[block]);
    REQUIRE(static_cast<std::size_t>(offsets[block + 1]) <= data.length());
  }
  CHECK(offsets[number_of_blocks] == static_cast<std::int32_t>(data.length()));
}

template <typename T>
auto same_array(const tf::cpp::nd_array<T> &first,
                const tf::cpp::nd_array<T> &second) -> bool {
  return first.raw_shape() == second.raw_shape() &&
         std::equal(first.begin(), first.end(), second.begin(), second.end());
}

template <typename Real>
auto absolute_volume(const parity_result<Real> &mesh) -> double {
  const auto owner = operand_of<Real>(mesh);
  return std::abs(static_cast<double>(tf::cpp::signed_volume(owner.mesh())));
}

template <typename Real>
auto check_closed_nonempty(const parity_result<Real> &mesh) -> void {
  REQUIRE(mesh.size() > 0);
  REQUIRE(mesh.points_buffer().size() > 0);
  const auto owner = operand_of<Real>(mesh);
  CHECK(tf::cpp::is_closed(owner.mesh()));
}

auto contains_id(const tf::cpp::nd_array<std::int32_t> &ids, std::int32_t id)
    -> bool {
  return std::find(ids.begin(), ids.end(), id) != ids.end();
}

template <typename Real> auto overlapping_spheres() -> parity_owned<Real> {
  std::vector<parity_owned<Real>> spheres;
  spheres.reserve(2);
  spheres.push_back({tf::cpp::make_sphere_mesh(Real{1}, 16, 24)});
  spheres.push_back({tf::cpp::make_sphere_mesh(Real{1}, 16, 24)});
  spheres[1].place(translation(Real{1}));
  return operand_of<Real>(tf::cpp::concatenate_meshes(meshes_of(spheres)));
}

constexpr auto domain_query_config = tf::domain_config::exclude_outer_shell |
                                     tf::domain_config::ignore_open_fragments;

} // namespace

TEMPLATE_TEST_CASE(
    "Python parity CSG build serves expression geometry and provenance",
    "[cpp][csg][python-parity][graph][provenance][ownership]", float, double) {
  const auto owners = overlapping_unit_boxes<TestType>();
  const auto input_geometry = capture_geometry(owners);
  REQUIRE(input_geometry[0].faces.size() != input_geometry[1].faces.size());
  auto graph = tf::cpp::make_csg_graph(meshes_of(owners));

  const auto merged_expression = tf::csg::op(0) | tf::csg::op(1);
  const auto overlap_expression = tf::csg::op(0) & tf::csg::op(1);
  const auto first_only_expression = tf::csg::op(0) - tf::csg::op(1);
  const auto second_only_expression = tf::csg::op(1) - tf::csg::op(0);

  auto merged = tf::cpp::make_csg_mesh(graph, merged_expression);
  auto overlap = tf::cpp::make_csg_mesh(graph, overlap_expression);
  auto first_only = tf::cpp::make_csg_mesh(graph, first_only_expression);
  auto second_only = tf::cpp::make_csg_mesh(graph, second_only_expression);
  auto full = tf::cpp::make_csg_mesh(graph);
  const auto full_labeled = tf::cpp::make_csg_mesh_with_labels(graph);
  const auto labeled =
      tf::cpp::make_csg_mesh_with_labels(graph, merged_expression);
  const auto mapped =
      tf::cpp::make_csg_mesh_with_index_map(graph, merged_expression);

  check_closed_nonempty(merged);
  check_closed_nonempty(overlap);
  check_closed_nonempty(first_only);
  check_closed_nonempty(second_only);
  REQUIRE(full.size() > merged.size());
  REQUIRE(full_labeled.mesh.size() == full.size());
  REQUIRE(full_labeled.tag_labels.length() == full.size());
  REQUIRE(full_labeled.face_labels.length() ==
          full_labeled.tag_labels.length());
  CHECK(std::find(full_labeled.tag_labels.begin(),
                  full_labeled.tag_labels.end(),
                  0) != full_labeled.tag_labels.end());
  CHECK(std::find(full_labeled.tag_labels.begin(),
                  full_labeled.tag_labels.end(),
                  1) != full_labeled.tag_labels.end());
  CHECK(absolute_volume(merged) == Catch::Approx(1.664).margin(2e-5));
  CHECK(absolute_volume(overlap) == Catch::Approx(0.336).margin(2e-5));
  CHECK(absolute_volume(first_only) == Catch::Approx(0.664).margin(2e-5));
  CHECK(absolute_volume(second_only) == Catch::Approx(0.664).margin(2e-5));
  CHECK(absolute_volume(merged) ==
        Catch::Approx(absolute_volume(first_only) +
                      absolute_volume(second_only) + absolute_volume(overlap))
            .margin(2e-5));

  REQUIRE(same_array(tf::cpp::test::face_indices_of(labeled.mesh),
                     tf::cpp::test::face_indices_of(mapped.mesh)));
  REQUIRE(same_array(result_points(labeled.mesh), result_points(mapped.mesh)));
  REQUIRE(same_array(labeled.tag_labels, mapped.face_tag_labels));
  REQUIRE(same_array(labeled.face_labels, mapped.face_labels));
  REQUIRE(labeled.tag_labels.length() == labeled.mesh.size());
  std::vector<std::vector<int>> pieces_per_source;
  pieces_per_source.reserve(input_geometry.size());
  for (const auto &input : input_geometry)
    pieces_per_source.emplace_back(input.faces.size(), 0);
  for (std::size_t face = 0; face < labeled.tag_labels.length(); ++face) {
    const auto tag = labeled.tag_labels[face];
    const auto source_face = labeled.face_labels[face];
    check_face_subdivision(labeled.mesh, static_cast<std::int32_t>(face),
                           input_geometry, tag, source_face);
    ++pieces_per_source[static_cast<std::size_t>(tag)]
                       [static_cast<std::size_t>(source_face)];
  }
  auto saw_subdivided_source = false;
  for (const auto &input : pieces_per_source)
    saw_subdivided_source = saw_subdivided_source ||
                            std::any_of(input.begin(), input.end(),
                                        [](auto pieces) { return pieces > 1; });
  REQUIRE(saw_subdivided_source);

  REQUIRE(mapped.number_of_tags == 2);
  REQUIRE(static_cast<std::size_t>(mapped.number_of_output_points) ==
          mapped.mesh.points_buffer().size());
  REQUIRE(mapped.number_of_original_points >= 0);
  REQUIRE(mapped.number_of_original_points <= mapped.number_of_output_points);
  REQUIRE(mapped.point_tag_labels.length() ==
          mapped.mesh.points_buffer().size());
  REQUIRE(mapped.point_labels.length() == mapped.point_tag_labels.length());
  check_offset_blocks(mapped.point_f_offsets, mapped.point_f_data,
                      input_geometry.size());
  for (std::size_t tag = 0; tag < input_geometry.size(); ++tag) {
    const auto begin = mapped.point_f_offsets[tag];
    const auto end = mapped.point_f_offsets[tag + 1];
    REQUIRE(static_cast<std::size_t>(end - begin) ==
            input_geometry[tag].points.size());
    for (std::size_t source_point = 0;
         source_point < input_geometry[tag].points.size(); ++source_point) {
      const auto output_point =
          mapped.point_f_data[static_cast<std::size_t>(begin) + source_point];
      if (output_point == mapped.number_of_output_points)
        continue;
      REQUIRE(output_point >= 0);
      REQUIRE(output_point < mapped.number_of_output_points);
      CHECK(mapped.point_tag_labels[static_cast<std::size_t>(output_point)] ==
            static_cast<std::int32_t>(tag));
      CHECK(mapped.point_labels[static_cast<std::size_t>(output_point)] ==
            static_cast<std::int32_t>(source_point));
      check_point_coordinates(mapped.mesh, output_point,
                              input_geometry[tag].points[source_point]);
    }
  }

  auto original_point_count = std::int32_t{0};
  auto created_point_count = std::int32_t{0};
  for (std::int32_t point = 0; point < mapped.number_of_output_points;
       ++point) {
    const auto tag = mapped.point_tag_labels[static_cast<std::size_t>(point)];
    const auto source_point =
        mapped.point_labels[static_cast<std::size_t>(point)];
    if (tag == mapped.number_of_tags) {
      ++created_point_count;
      CHECK(point >= mapped.number_of_original_points);
      CHECK(source_point == mapped.number_of_output_points);
      continue;
    }

    ++original_point_count;
    REQUIRE(point < mapped.number_of_original_points);
    REQUIRE(tag >= 0);
    REQUIRE(tag < mapped.number_of_tags);
    REQUIRE(source_point >= 0);
    REQUIRE(static_cast<std::size_t>(source_point) <
            input_geometry[static_cast<std::size_t>(tag)].points.size());
    check_point_coordinates(
        mapped.mesh, point,
        input_geometry[static_cast<std::size_t>(tag)]
            .points[static_cast<std::size_t>(source_point)]);
    const auto forward_offset =
        static_cast<std::size_t>(mapped.point_f_offsets[tag]) +
        static_cast<std::size_t>(source_point);
    CHECK(mapped.point_f_data[forward_offset] == point);
  }
  CHECK(original_point_count == mapped.number_of_original_points);
  CHECK(created_point_count ==
        mapped.number_of_output_points - mapped.number_of_original_points);
  REQUIRE(created_point_count > 0);

  const auto created = tf::cpp::csg_created_points(graph);
  REQUIRE((created.raw_shape() ==
           tf::small_vector<int, 3>{created.shape_at(0), 3}));
  CHECK(created.shape_at(0) > 0);
  const auto curves = tf::cpp::csg_intersection_curves(graph);
  const auto arrays = tf::cpp::test::arrays_of(curves);
  const auto &curve_points = arrays.points;
  REQUIRE(curves.size() > 0);
  REQUIRE((curve_points.raw_shape() ==
           tf::small_vector<int, 3>{curve_points.shape_at(0), 3}));
  REQUIRE(curve_points.shape_at(0) > 0);
  for (const auto path : curves.paths()) {
    REQUIRE(path.size() > 0);
    for (const auto point : path) {
      CHECK(point >= 0);
      CHECK(point < curve_points.shape_at(0));
    }
  }

  graph.destroy();
  CHECK(mapped.face_tag_labels.length() == mapped.mesh.size());
  CHECK(absolute_volume(mapped.mesh) == Catch::Approx(1.664).margin(2e-5));
}

TEMPLATE_TEST_CASE(
    "Python parity CSG domains align IDs labels maps and expression masks",
    "[cpp][csg][python-parity][graph][domains][provenance]", float, double) {
  const auto owners = overlapping_unit_boxes<TestType>();
  const auto input_geometry = capture_geometry(owners);
  auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
  const auto plain = tf::cpp::make_csg_domains(graph, domain_query_config);
  const auto labeled =
      tf::cpp::make_csg_domains_with_labels(graph, domain_query_config);
  const auto mapped =
      tf::cpp::make_csg_domains_with_index_map(graph, domain_query_config);
  const auto overlap = tf::cpp::make_csg_domains(
      graph, tf::csg::op(0) & tf::csg::op(1), domain_query_config);
  const auto first_only = tf::cpp::make_csg_domains(
      graph, tf::csg::op(0) - tf::csg::op(1), domain_query_config);
  const auto complement_form = tf::cpp::make_csg_domains(
      graph, tf::csg::op(0) & ~tf::csg::op(1), domain_query_config);
  const auto second_only = tf::cpp::make_csg_domains(
      graph, tf::csg::op(1) - tf::csg::op(0), domain_query_config);
  auto merged = tf::cpp::make_csg_mesh(graph, tf::csg::op(0) | tf::csg::op(1));

  REQUIRE(plain.meshes.size() == 3);
  REQUIRE(labeled.meshes.size() == plain.meshes.size());
  REQUIRE(mapped.meshes.size() == plain.meshes.size());
  REQUIRE(same_array(plain.ids, labeled.ids));
  REQUIRE(same_array(plain.ids, mapped.ids));
  REQUIRE(same_array(first_only.ids, complement_form.ids));
  REQUIRE(overlap.meshes.size() == 1);
  REQUIRE(first_only.meshes.size() == 1);
  REQUIRE(second_only.meshes.size() == 1);
  REQUIRE((mapped.inclusion.raw_shape() == tf::small_vector<int, 3>{3, 2}));

  REQUIRE(labeled.tag_offsets.length() == 4);
  REQUIRE(labeled.face_offsets.length() == 4);
  REQUIRE(mapped.face_tag_offsets.length() == 4);
  REQUIRE(mapped.face_offsets.length() == 4);
  REQUIRE(mapped.point_tag_offsets.length() == 4);
  REQUIRE(mapped.point_offsets.length() == 4);
  REQUIRE(mapped.number_of_tags == 2);
  REQUIRE(mapped.number_of_original_points >= 0);
  REQUIRE(mapped.number_of_original_points <= mapped.number_of_output_points);
  check_offset_blocks(mapped.face_tag_offsets, mapped.face_tag_data,
                      mapped.meshes.size());
  check_offset_blocks(mapped.face_offsets, mapped.face_data,
                      mapped.meshes.size());
  check_offset_blocks(mapped.point_tag_offsets, mapped.point_tag_data,
                      mapped.meshes.size());
  check_offset_blocks(mapped.point_offsets, mapped.point_data,
                      mapped.meshes.size());

  auto sum = 0.0;
  for (std::size_t cell = 0; cell < plain.meshes.size(); ++cell) {
    check_closed_nonempty(plain.meshes[cell]);
    sum += absolute_volume(plain.meshes[cell]);

    const auto face_count =
        static_cast<std::int32_t>(plain.meshes[cell].size());
    const auto point_count =
        static_cast<std::int32_t>(plain.meshes[cell].points_buffer().size());
    const auto tag_begin = static_cast<std::size_t>(labeled.tag_offsets[cell]);
    const auto face_begin =
        static_cast<std::size_t>(labeled.face_offsets[cell]);
    const auto map_tag_begin =
        static_cast<std::size_t>(mapped.face_tag_offsets[cell]);
    const auto map_face_begin =
        static_cast<std::size_t>(mapped.face_offsets[cell]);
    REQUIRE(labeled.tag_offsets[cell + 1] - labeled.tag_offsets[cell] ==
            face_count);
    REQUIRE(labeled.face_offsets[cell + 1] - labeled.face_offsets[cell] ==
            face_count);
    REQUIRE(mapped.face_tag_offsets[cell + 1] - mapped.face_tag_offsets[cell] ==
            face_count);
    REQUIRE(mapped.face_offsets[cell + 1] - mapped.face_offsets[cell] ==
            face_count);
    REQUIRE(mapped.point_tag_offsets[cell + 1] -
                mapped.point_tag_offsets[cell] ==
            point_count);
    REQUIRE(mapped.point_offsets[cell + 1] - mapped.point_offsets[cell] ==
            point_count);
    for (int face = 0; face < face_count; ++face) {
      const auto offset = static_cast<std::size_t>(face);
      const auto tag = mapped.face_tag_data[map_tag_begin + offset];
      const auto source_face = mapped.face_data[map_face_begin + offset];
      CHECK(labeled.tag_data[tag_begin + offset] == tag);
      CHECK(labeled.face_data[face_begin + offset] == source_face);
      check_face_subdivision(mapped.meshes[cell],
                             static_cast<std::int32_t>(face), input_geometry,
                             tag, source_face);
    }

    const auto point_tag_begin =
        static_cast<std::size_t>(mapped.point_tag_offsets[cell]);
    const auto point_begin =
        static_cast<std::size_t>(mapped.point_offsets[cell]);
    for (std::int32_t point = 0; point < point_count; ++point) {
      const auto offset = static_cast<std::size_t>(point);
      const auto tag = mapped.point_tag_data[point_tag_begin + offset];
      const auto source_point = mapped.point_data[point_begin + offset];
      if (tag == mapped.number_of_tags) {
        CHECK(source_point == mapped.number_of_output_points);
        continue;
      }
      REQUIRE(tag >= 0);
      REQUIRE(tag < mapped.number_of_tags);
      REQUIRE(source_point >= 0);
      REQUIRE(static_cast<std::size_t>(source_point) <
              input_geometry[static_cast<std::size_t>(tag)].points.size());
      check_point_coordinates(
          mapped.meshes[cell], point,
          input_geometry[static_cast<std::size_t>(tag)]
              .points[static_cast<std::size_t>(source_point)]);
    }

    const auto inclusion_offset = cell * 2;
    const auto inside_first = mapped.inclusion[inclusion_offset] == 1;
    const auto inside_second = mapped.inclusion[inclusion_offset + 1] == 1;
    const auto id = mapped.ids[cell];
    CHECK(contains_id(overlap.ids, id) == (inside_first && inside_second));
    CHECK(contains_id(first_only.ids, id) == (inside_first && !inside_second));
    CHECK(contains_id(second_only.ids, id) == (!inside_first && inside_second));
  }
  CHECK(sum == Catch::Approx(absolute_volume(merged)).margin(2e-5));

  graph.destroy();
  REQUIRE(mapped.ids.length() == 3);
  for (auto &mesh : mapped.meshes)
    check_closed_nonempty(mesh);
}

TEMPLATE_TEST_CASE(
    "Python parity three-form domains join inclusion to N-ary expressions",
    "[cpp][csg][python-parity][graph][domains][n-ary]", float, double) {
  const auto owners = three_overlapping_spheres<TestType>();
  auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
  const auto expression = tf::csg::op(0) & tf::csg::op(1) & tf::csg::op(2);
  const auto mapped =
      tf::cpp::make_csg_domains_with_index_map(graph, domain_query_config);
  const auto selected =
      tf::cpp::make_csg_domains(graph, expression, domain_query_config);
  const auto overlap_mesh = tf::cpp::make_csg_mesh(graph, expression);

  REQUIRE(mapped.number_of_tags == 3);
  REQUIRE(
      (mapped.inclusion.raw_shape() ==
       tf::small_vector<int, 3>{static_cast<int>(mapped.meshes.size()), 3}));
  REQUIRE(mapped.ids.length() == mapped.meshes.size());
  REQUIRE(selected.ids.length() == selected.meshes.size());

  std::vector<std::int32_t> inclusion_selected_ids;
  for (std::size_t domain = 0; domain < mapped.meshes.size(); ++domain) {
    const auto offset = domain * 3;
    for (std::size_t tag = 0; tag < 3; ++tag)
      REQUIRE((mapped.inclusion[offset + tag] == std::int8_t{0} ||
               mapped.inclusion[offset + tag] == std::int8_t{1}));
    if (mapped.inclusion[offset] == std::int8_t{1} &&
        mapped.inclusion[offset + 1] == std::int8_t{1} &&
        mapped.inclusion[offset + 2] == std::int8_t{1})
      inclusion_selected_ids.push_back(mapped.ids[domain]);
  }
  REQUIRE(inclusion_selected_ids.size() == 1);
  REQUIRE(selected.ids.length() == inclusion_selected_ids.size());
  for (const auto id : inclusion_selected_ids)
    CHECK(contains_id(selected.ids, id));
  for (const auto id : selected.ids)
    CHECK(std::find(inclusion_selected_ids.begin(),
                    inclusion_selected_ids.end(),
                    id) != inclusion_selected_ids.end());

  auto selected_volume = 0.0;
  for (std::size_t selected_domain = 0;
       selected_domain < selected.meshes.size(); ++selected_domain) {
    const auto id = selected.ids[selected_domain];
    auto mapped_domain = mapped.meshes.size();
    for (std::size_t candidate = 0; candidate < mapped.meshes.size();
         ++candidate) {
      if (mapped.ids[candidate] == id) {
        mapped_domain = candidate;
        break;
      }
    }
    REQUIRE(mapped_domain < mapped.meshes.size());
    REQUIRE(same_array(
        tf::cpp::test::face_indices_of(selected.meshes[selected_domain]),
        tf::cpp::test::face_indices_of(mapped.meshes[mapped_domain])));
    REQUIRE(same_array(result_points(selected.meshes[selected_domain]),
                       result_points(mapped.meshes[mapped_domain])));
    CHECK(absolute_volume(selected.meshes[selected_domain]) ==
          Catch::Approx(absolute_volume(mapped.meshes[mapped_domain]))
              .margin(geometry_tolerance<TestType>()));
    selected_volume += absolute_volume(selected.meshes[selected_domain]);
  }
  check_closed_nonempty(overlap_mesh);
  CHECK(selected_volume ==
        Catch::Approx(absolute_volume(overlap_mesh)).margin(2e-5));
}

TEMPLATE_TEST_CASE(
    "Python parity CSG refined triangulation preserves query semantics",
    "[cpp][csg][python-parity][graph][triangulation]", float, double) {
  const auto stock_owners = overlapping_unit_boxes<TestType>();
  const auto refined_owners = overlapping_unit_boxes<TestType>();
  auto stock_graph = tf::cpp::make_csg_graph(meshes_of(stock_owners));
  auto refined_graph = tf::cpp::make_csg_graph(
      meshes_of(refined_owners), {},
      tf::arrangement_config{tf::triangulation_type::refined_cdt});

  const auto stock_points = tf::cpp::csg_created_points(stock_graph);
  const auto refined_points = tf::cpp::csg_created_points(refined_graph);
  CHECK(refined_points.shape_at(0) > stock_points.shape_at(0));

  auto difference =
      tf::cpp::make_csg_mesh(refined_graph, tf::csg::op(0) - tf::csg::op(1));
  check_closed_nonempty(difference);
  CHECK(absolute_volume(difference) == Catch::Approx(0.664).margin(1e-4));
  const auto domains =
      tf::cpp::make_csg_domains(refined_graph, domain_query_config);
  CHECK(domains.meshes.size() == 3);
}

TEMPLATE_TEST_CASE("Python parity CSG validates domain expressions",
                   "[cpp][csg][python-parity][graph][validation]", float,
                   double) {
  const auto owners = overlapping_unit_boxes<TestType>();
  auto graph = tf::cpp::make_csg_graph(meshes_of(owners));
  CHECK_THROWS_AS(
      tf::cpp::make_csg_domains(graph, tf::csg::op(2), domain_query_config),
      std::out_of_range);
  const auto malformed = tf::csg::expr(tf::csg::expr::kind::complement, {});
  CHECK_THROWS_AS(
      tf::cpp::make_csg_domains(graph, malformed, domain_query_config),
      std::invalid_argument);
}

TEMPLATE_TEST_CASE(
    "Python parity outer shell merges overlaps and honors fragment policy",
    "[cpp][csg][python-parity][outer-shell][ownership]", float, double) {
  const auto soup = overlapping_spheres<TestType>();
  const auto soup_face_count = soup.polygons.size();
  const auto single_volume =
      absolute_volume<TestType>(tf::cpp::make_sphere_mesh(TestType{1}, 16, 24));

  const auto shell = operand_of<TestType>(tf::cpp::outer_shell(soup.mesh()));
  check_closed_nonempty(shell.polygons);
  CHECK(shell.polygons.size() < soup_face_count);
  CHECK(tf::cpp::signed_volume(shell.mesh()) > single_volume);
  CHECK(tf::cpp::signed_volume(shell.mesh()) < 2 * single_volume);
}

TEMPLATE_TEST_CASE(
    "Python parity outer shell removes cavities in the input local frame",
    "[cpp][csg][python-parity][outer-shell][transformation][cavity]", float,
    double) {
  std::vector<parity_owned<TestType>> spheres;
  spheres.reserve(2);
  spheres.push_back({tf::cpp::make_sphere_mesh(TestType{1}, 16, 24)});
  spheres.push_back({tf::cpp::make_sphere_mesh(TestType{0.4}, 16, 24)});
  const auto outer_volume = tf::cpp::signed_volume(spheres[0].mesh());
  auto soup =
      operand_of<TestType>(tf::cpp::concatenate_meshes(meshes_of(spheres)));
  soup.place(translation(TestType{25}, TestType{-10}, TestType{5}));

  // the shell is the operand's own boundary, so it is read where the operand
  // was authored rather than where its placement puts it
  const auto shell = operand_of<TestType>(tf::cpp::outer_shell(soup.mesh()));
  check_closed_nonempty(shell.polygons);
  CHECK(tf::cpp::signed_volume(shell.mesh()) ==
        Catch::Approx(outer_volume).epsilon(1e-4));
  for (const auto point : shell.polygons.points())
    for (const auto coordinate : point)
      CHECK(std::abs(coordinate) <= TestType{1.001});
}

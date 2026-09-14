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
#include "../common/canonicalize.hpp"
#include "../common/carriers.hpp"
#include "../common/fixtures.hpp"
#include "../common/temporary_directory.hpp"

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/io/async/read_obj.hpp"
#include "trueform/cpp/io/async/read_stl.hpp"
#include "trueform/cpp/io/async/write_obj.hpp"
#include "trueform/cpp/io/async/write_stl.hpp"
#include "trueform/cpp/io/bytes.hpp"
#include "trueform/cpp/io/read_obj.hpp"
#include "trueform/cpp/io/read_stl.hpp"
#include "trueform/cpp/io/write_obj.hpp"
#include "trueform/cpp/io/write_stl.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace {

template <typename Index, typename Real> struct io_matrix_row {
  using real_type = Real;
  using index_type = Index;
};

using io_matrix_rows = std::tuple<
    io_matrix_row<std::int32_t, float>, io_matrix_row<std::int64_t, float>,
    io_matrix_row<std::int32_t, double>, io_matrix_row<std::int64_t, double>>;

using stl_index_rows = std::tuple<io_matrix_row<std::int32_t, float>,
                                  io_matrix_row<std::int64_t, float>>;

auto copy_bytes(std::string_view text) -> tf::cpp::io_bytes {
  return tf::cpp::io_bytes::copy(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
}

auto copy_bytes(const tf::cpp::nd_array<std::int8_t> &bytes, std::size_t size)
    -> tf::cpp::io_bytes {
  return tf::cpp::io_bytes::copy(bytes.raw_data(), size);
}

auto copy_bytes(const tf::cpp::nd_array<std::int8_t> &bytes)
    -> tf::cpp::io_bytes {
  return copy_bytes(bytes, bytes.length());
}

auto same_bytes(const tf::cpp::nd_array<std::int8_t> &left,
                const tf::cpp::nd_array<std::int8_t> &right) -> bool {
  return left.length() == right.length() &&
         std::equal(left.begin(), left.end(), right.begin(), right.end());
}

auto same_file_bytes(const std::filesystem::path &path,
                     const tf::cpp::nd_array<std::int8_t> &expected) -> bool {
  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    return false;
  for (const auto expected_byte : expected) {
    char actual_byte = 0;
    if (!input.get(actual_byte) ||
        static_cast<unsigned char>(actual_byte) !=
            static_cast<unsigned char>(expected_byte))
      return false;
  }
  return input.peek() == std::char_traits<char>::eof();
}

auto write_file(const std::filesystem::path &path, const std::int8_t *data,
                std::size_t size) -> void {
  std::ofstream output(path, std::ios::binary);
  REQUIRE(output.is_open());
  output.write(reinterpret_cast<const char *>(data),
               static_cast<std::streamsize>(size));
  REQUIRE(output.good());
}

auto write_file(const std::filesystem::path &path, std::string_view text)
    -> void {
  write_file(path, reinterpret_cast<const std::int8_t *>(text.data()),
             text.size());
}

auto write_file(const std::filesystem::path &path,
                const tf::cpp::nd_array<std::int8_t> &bytes, std::size_t size)
    -> void {
  write_file(path, bytes.raw_data(), size);
}

template <typename Index, typename Real>
using parity_triangles = tf::polygons_buffer<Index, Real, 3, 3>;

template <typename Index, typename Real>
auto require_valid_mesh(const parity_triangles<Index, Real> &mesh, int faces,
                        int points) -> void {
  REQUIRE(mesh.size() == static_cast<std::size_t>(faces));
  REQUIRE(mesh.points_buffer().size() == static_cast<std::size_t>(points));
  for (const auto point_id : mesh.faces_buffer().data_buffer()) {
    CHECK(point_id >= 0);
    CHECK(point_id < static_cast<Index>(points));
  }
}

template <typename Real>
auto require_translated_unit_box(
    const parity_triangles<tf::cpp::default_index_t, Real> &mesh) -> void {
  require_valid_mesh(mesh, 12, 8);
  const auto points = mesh.points();
  std::array<bool, 8> found{};
  for (std::size_t point = 0; point != points.size(); ++point) {
    const auto x = points[point][0];
    const auto y = points[point][1];
    const auto z = points[point][2];
    REQUIRE((x == Real{5} || x == Real{6}));
    REQUIRE((y == Real{-2} || y == Real{-1}));
    REQUIRE((z == Real{10} || z == Real{11}));
    const auto id = static_cast<std::size_t>((x == Real{6} ? 1 : 0) |
                                             (y == Real{-1} ? 2 : 0) |
                                             (z == Real{11} ? 4 : 0));
    found[id] = true;
  }
  CHECK(std::all_of(found.begin(), found.end(),
                    [](bool value) { return value; }));
}

auto obj_with_comments_and_face_fields() -> std::string {
  return "# comments and ignored per-corner fields\n"
         "v 1.0000000000000002 0 0\n"
         "v 1 0 0\n"
         "v 0 1 0\n"
         "v 0 0 1\n"
         "vt 0 0\nvt 1 0\nvt 0 1\n"
         "vn 0 0 1\n"
         "f 1/1/1 2/2/1 3/3/1\n"
         "# v//vn is accepted too\n"
         "f 1//1 3//1 4//1\n";
}

auto shared_edge_ascii_stl() -> std::string {
  return "solid shared\n"
         " facet normal 0 0 1\n"
         "  outer loop\n"
         "   vertex 0 0 0\n"
         "   vertex 1 0 0\n"
         "   vertex 0 1 0\n"
         "  endloop\n"
         " endfacet\n"
         " facet normal 0 0 1\n"
         "  outer loop\n"
         "   vertex 1 0 0\n"
         "   vertex 1 1 0\n"
         "   vertex 0 1 0\n"
         "  endloop\n"
         " endfacet\n"
         "endsolid shared\n";
}

auto simple_quad_obj() -> std::string {
  return "# Simple quad\n"
         "v 0 0 0\n"
         "v 1 0 0\n"
         "v 1 1 0\n"
         "v 0 1 0\n"
         "f 1 2 3 4\n";
}

auto python_mixed_obj() -> std::string {
  return "# Mixed: triangles and quads\n"
         "v 0 0 0\n"
         "v 1 0 0\n"
         "v 1 1 0\n"
         "v 0 1 0\n"
         "v 2 0 0\n"
         "f 1 2 3\n"
         "f 1 3 4\n"
         "f 1 2 5 3\n";
}

template <typename Buffer>
auto same_typed_mesh(const Buffer &left, const Buffer &right) -> bool {
  const auto &left_faces = left.faces_buffer().data_buffer();
  const auto &right_faces = right.faces_buffer().data_buffer();
  const auto &left_points = left.points_buffer().data_buffer();
  const auto &right_points = right.points_buffer().data_buffer();
  return std::equal(left_faces.begin(), left_faces.end(), right_faces.begin(),
                    right_faces.end()) &&
         std::equal(left_points.begin(), left_points.end(),
                    right_points.begin(), right_points.end());
}

template <typename Index, typename Real>
auto typed_triangle_mesh() -> tf::cpp::test::owned_mesh<Index, Real> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {Index{0}, Index{1}, Index{2}},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}})};
}

template <typename Index, typename Real>
auto typed_mixed_mesh()
    -> tf::cpp::test::owned_mesh<Index, Real, 3, tf::dynamic_size> {
  return {tf::cpp::test::polygons_of<Index, Real>(
      {Index{0}, Index{3}, Index{7}},
      {Index{0}, Index{1}, Index{2}, Index{0}, Index{1}, Index{3}, Index{2}},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Real>
auto translated(Real x, Real y, Real z) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1}, Real{0}, y,
          Real{0}, Real{0}, Real{1}, z, Real{0}, Real{0}, Real{0}, Real{1}};
}

/// STL takes the triangles a mesh is typed on, so whether a mesh can be
/// written is a fact about its type.
template <typename Mesh, typename = void>
struct writes_stl : std::false_type {};
template <typename Mesh>
struct writes_stl<Mesh, std::void_t<decltype(tf::cpp::write_stl(
                            std::declval<const Mesh &>()))>> : std::true_type {
};

} // namespace

TEMPLATE_TEST_CASE(
    "Python parity OBJ parsing is path independent and owns fixed results",
    "[cpp][io][python-parity][obj][read]", float, double) {
  tf::cpp::test::temporary_directory directory("trueform-obj-parity");
  const auto path = directory.path() / "fields.obj";
  auto text = obj_with_comments_and_face_fields();
  write_file(path, text);

  const auto from_path =
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(path.string());
  const auto from_owned =
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
          copy_bytes(text));
  auto from_span = tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());

  require_valid_mesh(from_path, 2, 4);
  CHECK(tf::cpp::test::canonicalize_mesh_topology(from_path) ==
        tf::cpp::test::canonicalize_mesh_topology(from_owned));
  CHECK(tf::cpp::test::canonicalize_mesh_topology(from_owned) ==
        tf::cpp::test::canonicalize_mesh_topology(from_span));
  const auto &indices = from_path.faces_buffer().data_buffer();
  CHECK(indices[0] == 0);
  CHECK(indices[1] == 1);
  CHECK(indices[2] == 2);
  CHECK(indices[3] == 0);
  CHECK(indices[4] == 2);
  CHECK(indices[5] == 3);
  if constexpr (std::is_same_v<TestType, double>) {
    CHECK(from_path.points()[0][0] == std::nextafter(1.0, 2.0));
    CHECK(from_path.points()[0][0] != 1.0);
  } else {
    CHECK(from_path.points()[0][0] == 1.0F);
  }

  std::fill(text.begin(), text.end(), 'x');
  CHECK(from_span.size() == 2);
  const auto owned_first_point = from_owned.points()[0][0];
  from_span.points()[0][0] = TestType{99};
  CHECK(from_owned.points()[0][0] == owned_first_point);
  CHECK(from_path.points()[0][0] == owned_first_point);
}

TEST_CASE(
    "Python parity STL parsing deduplicates shared vertices across routes",
    "[cpp][io][python-parity][stl][read]") {
  tf::cpp::test::temporary_directory directory("trueform-stl-parity");
  const auto path = directory.path() / "shared.stl";
  auto text = shared_edge_ascii_stl();
  write_file(path, text);

  const auto from_path = tf::cpp::read_stl(path.string());
  const auto from_owned = tf::cpp::read_stl(copy_bytes(text));
  auto from_span = tf::cpp::read_stl(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());

  require_valid_mesh(from_path, 2, 4);
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(from_path) ==
        tf::cpp::test::canonicalize_mesh_geometry(from_owned));
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(from_owned) ==
        tf::cpp::test::canonicalize_mesh_geometry(from_span));

  const auto &indices = from_path.faces_buffer().data_buffer();
  int shared_vertices = 0;
  for (std::size_t first = 0; first != 3; ++first)
    for (std::size_t second = 0; second != 3; ++second)
      shared_vertices += indices[first] == indices[3 + second] ? 1 : 0;
  CHECK(shared_vertices == 2);

  std::fill(text.begin(), text.end(), 'x');
  CHECK(from_span.points_buffer().size() == 4);
  const auto owned_first_point = from_owned.points()[0][0];
  from_span.points()[0][0] = 99.0F;
  CHECK(from_owned.points()[0][0] == owned_first_point);
  CHECK(from_path.points()[0][0] == owned_first_point);
}

TEMPLATE_LIST_TEST_CASE(
    "Python OBJ readers cover both arities of the Real Index matrix",
    "[cpp][io][python-parity][obj][read][matrix]", io_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  tf::cpp::test::temporary_directory directory("trueform-obj-matrix");
  const auto triangle_path = directory.path() / "triangle.obj";
  auto triangle_text = obj_with_comments_and_face_fields();
  write_file(triangle_path, triangle_text);

  const auto triangles =
      tf::cpp::read_obj<Index, Real, 3>(triangle_path.string());
  const auto triangle_bytes = tf::cpp::read_obj<Index, Real, 3>(
      reinterpret_cast<const std::int8_t *>(triangle_text.data()),
      triangle_text.size());
  STATIC_REQUIRE(
      std::is_same_v<decltype(triangles), const parity_triangles<Index, Real>>);
  require_valid_mesh(triangles, 2, 4);
  CHECK(same_typed_mesh(triangles, triangle_bytes));
  const auto &triangle_indices = triangles.faces_buffer().data_buffer();
  CHECK(triangle_indices[0] == Index{0});
  CHECK(triangle_indices[1] == Index{1});
  CHECK(triangle_indices[2] == Index{2});
  CHECK(triangle_indices[5] == Index{3});
  if constexpr (std::is_same_v<Real, double>)
    CHECK(triangles.points()[0][0] == std::nextafter(1.0, 2.0));
  else
    CHECK(triangles.points()[0][0] == 1.0F);

  // a quad file is read at the arity the file states, not at a third one
  const auto quad_path = directory.path() / "quad.obj";
  const auto quad_text = simple_quad_obj();
  write_file(quad_path, quad_text);
  auto quad = tf::cpp::read_obj<Index, Real>(quad_path.string());
  const auto quad_span = tf::cpp::read_obj<Index, Real>(
      reinterpret_cast<const std::int8_t *>(quad_text.data()),
      quad_text.size());
  REQUIRE(quad.size() == 1);
  REQUIRE(quad.faces()[0].size() == 4);
  REQUIRE(quad.points_buffer().size() == 4);
  for (std::size_t index = 0; index != 4; ++index) {
    CHECK(quad.faces()[0][index] == static_cast<Index>(index));
    CHECK(quad_span.faces()[0][index] == static_cast<Index>(index));
  }
  quad.points()[0][0] = Real{99};
  CHECK(quad_span.points()[0][0] == Real{0});

  const auto mixed_path = directory.path() / "mixed.obj";
  auto mixed_text = python_mixed_obj();
  write_file(mixed_path, mixed_text);
  auto mixed = tf::cpp::read_obj<Index, Real>(mixed_path.string());
  auto mixed_owned = tf::cpp::read_obj<Index, Real>(copy_bytes(mixed_text));
  STATIC_REQUIRE(
      std::is_same_v<decltype(mixed),
                     tf::polygons_buffer<Index, Real, 3, tf::dynamic_size>>);
  REQUIRE(mixed.size() == 3);
  const auto &offsets = mixed.faces_buffer().offsets_buffer();
  const std::array<Index, 4> expected_offsets{Index{0}, Index{3}, Index{6},
                                              Index{10}};
  CHECK(std::equal(offsets.begin(), offsets.end(), expected_offsets.begin(),
                   expected_offsets.end()));
  const std::array<Index, 10> expected_indices{
      Index{0}, Index{1}, Index{2}, Index{0}, Index{2},
      Index{3}, Index{0}, Index{1}, Index{4}, Index{2}};
  const auto &mixed_indices = mixed.faces_buffer().data_buffer();
  CHECK(std::equal(mixed_indices.begin(), mixed_indices.end(),
                   expected_indices.begin(), expected_indices.end()));
  REQUIRE(mixed.points_buffer().size() == 5);
  mixed.points()[0][0] = Real{99};
  CHECK(mixed_owned.points()[0][0] == Real{0});

  auto future = tf::cpp::async::read_obj<Index, Real>(
      reinterpret_cast<const std::int8_t *>(mixed_text.data()),
      mixed_text.size());
  STATIC_REQUIRE(
      std::is_same_v<
          decltype(future),
          std::future<tf::polygons_buffer<Index, Real, 3, tf::dynamic_size>>>);
  std::fill(mixed_text.begin(), mixed_text.end(), 'x');
  const auto async_result = future.get();
  CHECK(async_result.size() == 3);
  CHECK(async_result.points_buffer().size() == 5);
}

TEMPLATE_LIST_TEST_CASE(
    "Python STL readers cover both index dtypes with owned deduplication",
    "[cpp][io][python-parity][stl][read][matrix]", stl_index_rows) {
  using Index = typename TestType::index_type;

  tf::cpp::test::temporary_directory directory("trueform-stl-index-matrix");
  const auto path = directory.path() / "shared.stl";
  auto text = shared_edge_ascii_stl();
  write_file(path, text);

  auto from_path = tf::cpp::read_stl<Index>(path.string());
  auto from_owned = tf::cpp::read_stl<Index>(copy_bytes(text));
  auto future = tf::cpp::async::read_stl<Index>(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
  STATIC_REQUIRE(
      std::is_same_v<decltype(from_path), parity_triangles<Index, float>>);
  STATIC_REQUIRE(std::is_same_v<decltype(future),
                                std::future<parity_triangles<Index, float>>>);
  std::fill(text.begin(), text.end(), 'x');
  auto from_async = future.get();

  require_valid_mesh(from_path, 2, 4);
  CHECK(same_typed_mesh(from_path, from_owned));
  CHECK(same_typed_mesh(from_owned, from_async));
  from_path.points()[0][0] = 99.0F;
  CHECK(from_owned.points()[0][0] != 99.0F);
}

TEMPLATE_LIST_TEST_CASE(
    "Python OBJ writers cover both arities and the frame they are placed by",
    "[cpp][io][python-parity][obj][write][matrix]", io_matrix_rows) {
  using Real = typename TestType::real_type;
  using Index = typename TestType::index_type;

  auto triangles = typed_triangle_mesh<Index, Real>();
  triangles.place(translated(Real{5}, Real{0}, Real{10}));
  const auto triangle_bytes = tf::cpp::write_obj(triangles.mesh());
  const auto triangle_roundtrip =
      tf::cpp::read_obj<Index, Real, 3>(copy_bytes(triangle_bytes));
  REQUIRE(triangle_roundtrip.size() == 1);
  CHECK(triangle_roundtrip.points()[0][0] == Real{5});
  CHECK(triangle_roundtrip.points()[0][2] == Real{10});
  CHECK(triangles.polygons.points()[0][0] == Real{0});

  tf::cpp::test::temporary_directory directory("trueform-obj-write-matrix");
  const auto base_path = directory.path() / "typed";
  auto obj_path = base_path;
  obj_path += ".obj";
  REQUIRE(tf::cpp::write_obj(triangles.mesh(), base_path));
  CHECK(same_file_bytes(obj_path, triangle_bytes));

  auto mixed = typed_mixed_mesh<Index, Real>();
  mixed.place(translated(Real{2}, Real{3}, Real{0}));
  const auto mixed_bytes = tf::cpp::write_obj(mixed.mesh());
  const auto mixed_roundtrip =
      tf::cpp::read_obj<Index, Real>(copy_bytes(mixed_bytes));
  REQUIRE(mixed_roundtrip.size() == 2);
  CHECK(mixed_roundtrip.faces()[0].size() == 3);
  CHECK(mixed_roundtrip.faces()[1].size() == 4);
  CHECK(mixed_roundtrip.points()[0][0] == Real{2});
  CHECK(mixed_roundtrip.points()[0][1] == Real{3});
  CHECK(mixed.polygons.points()[0][0] == Real{0});

  // a write borrows one reading: the mesh the job carries names storage the
  // caller keeps until the future completes
  auto future = tf::cpp::async::write_obj(mixed.mesh());
  STATIC_REQUIRE(std::is_same_v<decltype(future),
                                std::future<tf::cpp::nd_array<std::int8_t>>>);
  const auto async_roundtrip =
      tf::cpp::read_obj<Index, Real>(copy_bytes(future.get()));
  CHECK(async_roundtrip.size() == 2);
  CHECK(async_roundtrip.points()[0][0] == Real{2});
}

TEMPLATE_LIST_TEST_CASE(
    "Python STL writers cover both index dtypes transform and mixed refusal",
    "[cpp][io][python-parity][stl][write][matrix]", stl_index_rows) {
  using Index = typename TestType::index_type;

  auto source = typed_triangle_mesh<Index, float>();
  source.place(translated(5.0F, 0.0F, 10.0F));
  const auto bytes = tf::cpp::write_stl(source.mesh());
  REQUIRE(bytes.length() == 134);
  const auto roundtrip = tf::cpp::read_stl<Index>(copy_bytes(bytes));
  REQUIRE(roundtrip.size() == 1);
  CHECK(roundtrip.points()[0][0] == 5.0F);
  CHECK(roundtrip.points()[0][2] == 10.0F);
  CHECK(source.polygons.points()[0][0] == 0.0F);

  tf::cpp::test::temporary_directory directory("trueform-stl-write-matrix");
  const auto base_path = directory.path() / "typed";
  auto stl_path = base_path;
  stl_path += ".stl";
  REQUIRE(tf::cpp::write_stl(source.mesh(), base_path));
  CHECK(same_file_bytes(stl_path, bytes));

  auto future = tf::cpp::async::write_stl(source.mesh());
  CHECK(future.get().length() == 134);

  // STL is triangles, so a mixed mesh is not something this entry can be
  // asked: the layout is the mesh's type, and the refusal is the compiler's
  STATIC_REQUIRE(writes_stl<tf::cpp::mesh<Index, float, 3, 3>>::value);
  STATIC_REQUIRE_FALSE(
      writes_stl<tf::cpp::mesh<Index, float, 3, tf::dynamic_size>>::value);
}

TEMPLATE_TEST_CASE(
    "Python parity writers are deterministic transformed and nonmutating",
    "[cpp][io][python-parity][write][round-trip]", float, double) {
  tf::cpp::test::temporary_directory directory("trueform-write-parity");
  auto source = tf::cpp::test::box_mesh<TestType>();
  source.place(translated(TestType{5}, TestType{-2}, TestType{10}));
  const auto source_signature =
      tf::cpp::test::canonicalize_mesh_topology(source.polygons);

  const auto obj_first = tf::cpp::write_obj(source.mesh());
  const auto obj_second = tf::cpp::write_obj(source.mesh());
  const auto stl_first = tf::cpp::write_stl(source.mesh());
  const auto stl_second = tf::cpp::write_stl(source.mesh());
  REQUIRE_FALSE(obj_first.empty());
  REQUIRE(stl_first.length() == 84 + 50 * 12);
  CHECK(same_bytes(obj_first, obj_second));
  CHECK(same_bytes(stl_first, stl_second));
  CHECK(tf::cpp::test::canonicalize_mesh_topology(source.polygons) ==
        source_signature);

  const auto obj_base_path = directory.path() / "roundtrip-obj";
  auto obj_path = obj_base_path;
  obj_path += ".obj";
  const auto obj_explicit_path = directory.path() / "explicit.obj";
  auto obj_duplicate_path = obj_explicit_path;
  obj_duplicate_path += ".obj";
  const auto stl_base_path = directory.path() / "roundtrip-stl";
  auto stl_path = stl_base_path;
  stl_path += ".stl";
  const auto stl_explicit_path = directory.path() / "explicit.stl";
  auto stl_duplicate_path = stl_explicit_path;
  stl_duplicate_path += ".stl";

  REQUIRE(tf::cpp::write_obj(source.mesh(), obj_base_path));
  REQUIRE(tf::cpp::write_obj(source.mesh(), obj_explicit_path));
  REQUIRE(tf::cpp::write_stl(source.mesh(), stl_base_path));
  REQUIRE(tf::cpp::write_stl(source.mesh(), stl_explicit_path));
  CHECK_FALSE(std::filesystem::exists(obj_base_path));
  CHECK_FALSE(std::filesystem::exists(obj_duplicate_path));
  CHECK_FALSE(std::filesystem::exists(stl_base_path));
  CHECK_FALSE(std::filesystem::exists(stl_duplicate_path));
  CHECK(same_file_bytes(obj_path, obj_first));
  CHECK(same_file_bytes(obj_explicit_path, obj_first));
  CHECK(same_file_bytes(stl_path, stl_first));
  CHECK(same_file_bytes(stl_explicit_path, stl_first));

  const auto obj_from_path =
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
          obj_path.string());
  const auto obj_from_owned =
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
          copy_bytes(obj_first));
  const auto obj_from_span =
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
          obj_first.raw_data(), obj_first.length());
  require_translated_unit_box(obj_from_path);
  CHECK(tf::cpp::test::canonicalize_mesh_topology(obj_from_path) ==
        tf::cpp::test::canonicalize_mesh_topology(obj_from_owned));
  CHECK(tf::cpp::test::canonicalize_mesh_topology(obj_from_owned) ==
        tf::cpp::test::canonicalize_mesh_topology(obj_from_span));
  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, TestType> rewritten{
      tf::cpp::read_obj<tf::cpp::default_index_t, TestType, 3>(
          obj_path.string())};
  CHECK(same_bytes(tf::cpp::write_obj(rewritten.mesh()), obj_first));

  const auto stl_from_path = tf::cpp::read_stl(stl_path.string());
  const auto stl_from_owned = tf::cpp::read_stl(copy_bytes(stl_first));
  const auto stl_from_span =
      tf::cpp::read_stl(stl_first.raw_data(), stl_first.length());
  require_translated_unit_box(stl_from_path);
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(stl_from_path) ==
        tf::cpp::test::canonicalize_mesh_geometry(stl_from_owned));
  CHECK(tf::cpp::test::canonicalize_mesh_geometry(stl_from_owned) ==
        tf::cpp::test::canonicalize_mesh_geometry(stl_from_span));
  const tf::cpp::test::owned_mesh<tf::cpp::default_index_t, float> restated{
      tf::cpp::read_stl(stl_path.string())};
  CHECK(same_bytes(tf::cpp::write_stl(restated.mesh()), stl_first));
}

TEST_CASE("Python parity malformed IO fails equally for paths and bytes",
          "[cpp][io][python-parity][malformed]") {
  tf::cpp::test::temporary_directory directory("trueform-malformed-parity");

  const std::string malformed_obj = "v 0 nope 0\nf 1 2 3\n";
  const auto obj_path = directory.path() / "malformed.obj";
  write_file(obj_path, malformed_obj);
  CHECK(tf::cpp::read_obj<tf::cpp::default_index_t, float, 3>(obj_path.string())
            .size() == 0);
  CHECK(tf::cpp::read_obj<tf::cpp::default_index_t, float, 3>(
            copy_bytes(malformed_obj))
            .size() == 0);
  CHECK(tf::cpp::read_obj<tf::cpp::default_index_t, float, 3>(
            reinterpret_cast<const std::int8_t *>(malformed_obj.data()),
            malformed_obj.size())
            .size() == 0);

  const auto source = tf::cpp::test::triangle_mesh<float>();
  const auto valid_stl = tf::cpp::write_stl(source.mesh());
  REQUIRE(valid_stl.length() > 1);
  const auto truncated_size = valid_stl.length() - 1;
  const auto stl_path = directory.path() / "truncated.stl";
  write_file(stl_path, valid_stl, truncated_size);
  CHECK(tf::cpp::read_stl(stl_path.string()).size() == 0);
  CHECK(tf::cpp::read_stl(copy_bytes(valid_stl, truncated_size)).size() == 0);
  CHECK(tf::cpp::read_stl(valid_stl.raw_data(), truncated_size).size() == 0);
}

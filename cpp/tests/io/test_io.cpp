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
#include "../common/async_resolver.hpp"
#include "../common/carriers.hpp"
#include "../common/fixtures.hpp"
#include "../common/temporary_directory.hpp"

#include "trueform/core/buffer.hpp"
#include "trueform/core/polygons_buffer.hpp"
#include "trueform/core/static_size.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/mesh.hpp"
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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using io_index_t = tf::cpp::default_index_t;

template <typename Real>
using io_triangles = tf::polygons_buffer<io_index_t, Real, 3, 3>;

auto bytes_from_string(const std::string &text)
    -> tf::cpp::nd_array<std::int8_t> {
  tf::buffer<std::int8_t> bytes;
  bytes.allocate(text.size());
  if (!text.empty())
    std::memcpy(bytes.data(), text.data(), text.size());
  return tf::cpp::nd_array<std::int8_t>::from_buffer(
      std::move(bytes), {static_cast<int>(text.size())});
}

auto bytes_to_string(const tf::cpp::nd_array<std::int8_t> &bytes)
    -> std::string {
  return std::string(reinterpret_cast<const char *>(bytes.raw_data()),
                     bytes.length());
}

auto io_bytes_from_array(const tf::cpp::nd_array<std::int8_t> &bytes)
    -> tf::cpp::io_bytes {
  return tf::cpp::io_bytes::copy(bytes.raw_data(), bytes.length());
}

auto check_empty_io_bytes(const tf::cpp::io_bytes &bytes) -> void {
  CHECK(bytes.size() == 0);
  REQUIRE(bytes.data() != nullptr);
  CHECK(bytes.data()[0] == 0);
  const auto range = bytes.make_range();
  CHECK(range.empty());
  CHECK(range.begin() == range.end());
  CHECK(range.begin() == reinterpret_cast<const char *>(bytes.data()));
}

auto ascii_stl() -> std::string {
  return "solid triangle\n"
         " facet normal 0 0 1\n"
         "  outer loop\n"
         "   vertex 0 0 0\n"
         "   vertex 2 0 0\n"
         "   vertex 0 1 0\n"
         "  endloop\n"
         " endfacet\n"
         "endsolid triangle";
}

auto triangle_obj() -> std::string {
  return "v 0 0 0\nv 2.123456789012345 0 0\nv 0 1 0\nf 1 2 3\n";
}

auto mixed_obj() -> std::string {
  return "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 1 0\n"
         "f 1 2 3\nf 1 3 4 5\n";
}

class temporary_file {
  std::string _path;

public:
  temporary_file(const std::string &suffix,
                 const tf::cpp::nd_array<std::int8_t> &bytes) {
    const auto stamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    _path = "/tmp/trueform_cpp_io_" + std::to_string(stamp) + suffix;
    std::ofstream output(_path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(bytes.raw_data()),
                 static_cast<std::streamsize>(bytes.length()));
  }

  temporary_file(const std::string &suffix, const std::string &text)
      : temporary_file(suffix, bytes_from_string(text)) {}

  temporary_file(const temporary_file &) = delete;
  auto operator=(const temporary_file &) -> temporary_file & = delete;
  ~temporary_file() { std::remove(_path.c_str()); }

  auto path() const -> const std::string & { return _path; }
};

/// A read hands back core's own storage, so parity is that storage compared
/// where it lies.
template <typename Real>
auto same_read_mesh(const io_triangles<Real> &first,
                    const io_triangles<Real> &second) -> bool {
  const auto &first_faces = first.faces_buffer().data_buffer();
  const auto &second_faces = second.faces_buffer().data_buffer();
  const auto &first_points = first.points_buffer().data_buffer();
  const auto &second_points = second.points_buffer().data_buffer();
  return std::equal(first_faces.begin(), first_faces.end(),
                    second_faces.begin(), second_faces.end()) &&
         std::equal(first_points.begin(), first_points.end(),
                    second_points.begin(), second_points.end());
}

template <typename Real> auto translated(Real x) -> std::array<Real, 16> {
  return {Real{1}, Real{0}, Real{0}, x,       Real{0}, Real{1},
          Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0},
          Real{0}, Real{0}, Real{0}, Real{1}};
}

} // namespace

TEST_CASE("STL reads ASCII and binary with path and byte parity",
          "[cpp][io][stl][read]") {
  auto ascii = bytes_from_string(ascii_stl());
  temporary_file ascii_file(".stl", ascii);
  const auto from_path = tf::cpp::read_stl(ascii_file.path());
  const auto from_owned = tf::cpp::read_stl(io_bytes_from_array(ascii));
  const auto from_span = tf::cpp::read_stl(ascii.raw_data(), ascii.length());
  REQUIRE(from_path.size() == 1);
  CHECK(same_read_mesh(from_path, from_owned));
  CHECK(same_read_mesh(from_owned, from_span));

  const auto source = tf::cpp::test::triangle_mesh<float>();
  const auto binary = tf::cpp::write_stl(source.mesh());
  temporary_file binary_file(".stl", binary);
  const auto binary_path = tf::cpp::read_stl(binary_file.path());
  const auto binary_bytes = tf::cpp::read_stl(io_bytes_from_array(binary));
  REQUIRE(binary.length() == 134);
  CHECK(binary_path.size() == 1);
  CHECK(same_read_mesh(binary_path, binary_bytes));
}

TEMPLATE_TEST_CASE("OBJ reads at triangle arity preserve precision and path "
                   "parity",
                   "[cpp][io][obj][read]", float, double) {
  auto bytes = bytes_from_string(triangle_obj());
  temporary_file file(".obj", bytes);
  const auto from_path =
      tf::cpp::read_obj<io_index_t, TestType, 3>(file.path());
  const auto from_owned =
      tf::cpp::read_obj<io_index_t, TestType, 3>(io_bytes_from_array(bytes));
  const auto from_span = tf::cpp::read_obj<io_index_t, TestType, 3>(
      bytes.raw_data(), bytes.length());
  REQUIRE(from_path.size() == 1);
  CHECK(same_read_mesh(from_path, from_owned));
  CHECK(same_read_mesh(from_owned, from_span));
  if constexpr (std::is_same_v<TestType, double>)
    CHECK(from_owned.points()[1][0] == Catch::Approx(2.123456789012345));
  else
    CHECK(from_owned.points()[1][0] ==
          Catch::Approx(static_cast<float>(2.123456789012345)));
}

/// THE ARITY IS THE REQUEST: an OBJ states its own faces, so the default read
/// is the mixed one and a triangle read of a file holding anything else is
/// nothing.
TEMPLATE_TEST_CASE("OBJ reads state the arity they are asked for",
                   "[cpp][io][obj][dynamic]", float, double) {
  auto bytes = bytes_from_string(mixed_obj());
  const auto owned =
      tf::cpp::read_obj<io_index_t, TestType>(io_bytes_from_array(bytes));
  const auto span =
      tf::cpp::read_obj<io_index_t, TestType>(bytes.raw_data(), bytes.length());
  STATIC_REQUIRE(
      std::is_same_v<decltype(owned),
                     const tf::polygons_buffer<io_index_t, TestType, 3,
                                               tf::dynamic_size>>);
  REQUIRE(owned.size() == 2);
  CHECK(owned.faces()[0].size() == 3);
  CHECK(owned.faces()[1].size() == 4);
  CHECK(owned.points_buffer().size() == 5);
  CHECK(span.size() == owned.size());
  CHECK(span.points()[4][0] == owned.points()[4][0]);

  const auto triangles =
      tf::cpp::read_obj<io_index_t, TestType, 3>(io_bytes_from_array(bytes));
  CHECK(triangles.size() == 0);
  CHECK(triangles.points_buffer().size() == 0);
}

TEMPLATE_TEST_CASE("writers apply the mesh's frame and round trip",
                   "[cpp][io][write][transform]", float, double) {
  auto source = tf::cpp::test::triangle_mesh<TestType>();
  source.place(translated(TestType{10}));

  const auto obj = tf::cpp::write_obj(source.mesh());
  const auto obj_roundtrip =
      tf::cpp::read_obj<io_index_t, TestType, 3>(io_bytes_from_array(obj));
  REQUIRE(obj_roundtrip.size() == 1);
  CHECK(obj_roundtrip.points()[0][0] == Catch::Approx(10));
  CHECK(bytes_to_string(obj).find("v 10 ") != std::string::npos);

  const auto stl = tf::cpp::write_stl(source.mesh());
  const auto stl_roundtrip = tf::cpp::read_stl(io_bytes_from_array(stl));
  REQUIRE(stl_roundtrip.size() == 1);
  CHECK(stl_roundtrip.points()[0][0] == Catch::Approx(10));
  // the frame places the reading; the caller's own coordinates never move
  CHECK(source.polygons.points()[0][0] == TestType{0});
}

TEST_CASE("OBJ float64 serialization retains precision and STL downcasts",
          "[cpp][io][precision]") {
  const tf::cpp::test::owned_mesh<io_index_t, double> source{
      tf::cpp::test::polygons_of<io_index_t, double>(
          {0, 1, 2}, {0, 0, 0, 1.123456789012345, 0, 0, 0, 1, 0})};
  const auto obj = tf::cpp::write_obj(source.mesh());
  CHECK(bytes_to_string(obj).find("1.123456789012345") != std::string::npos);
  const auto obj_mesh =
      tf::cpp::read_obj<io_index_t, double, 3>(io_bytes_from_array(obj));
  CHECK(obj_mesh.points()[1][0] == Catch::Approx(1.123456789012345));

  const auto stl_mesh =
      tf::cpp::read_stl(io_bytes_from_array(tf::cpp::write_stl(source.mesh())));
  STATIC_REQUIRE(std::is_same_v<decltype(stl_mesh), const io_triangles<float>>);
  bool found_downcast_value = false;
  for (const auto value : stl_mesh.points_buffer().data_buffer())
    if (value == static_cast<float>(1.123456789012345))
      found_downcast_value = true;
  CHECK(found_downcast_value);
}

TEMPLATE_TEST_CASE("empty and failed IO returns valid empty results",
                   "[cpp][io][empty][failure]", float, double) {
  const auto nothing = tf::cpp::test::empty_mesh<TestType>();
  const auto obj_bytes = tf::cpp::write_obj(nothing.mesh());
  const auto stl_bytes = tf::cpp::write_stl(nothing.mesh());
  CHECK(obj_bytes.is_valid());
  CHECK(obj_bytes.empty());
  CHECK(stl_bytes.is_valid());
  CHECK(stl_bytes.length() == 84);
  CHECK(tf::cpp::read_stl(io_bytes_from_array(stl_bytes)).size() == 0);

  tf::cpp::test::temporary_directory directory("trueform-io-empty-path");
  const auto empty_obj_base = directory.path() / "empty-obj";
  auto empty_obj_path = empty_obj_base;
  empty_obj_path += ".obj";
  const auto empty_stl_path = directory.path() / "empty.stl";
  CHECK_FALSE(tf::cpp::write_obj(nothing.mesh(), empty_obj_base));
  CHECK_FALSE(std::filesystem::exists(empty_obj_path));
  REQUIRE(tf::cpp::write_stl(nothing.mesh(), empty_stl_path));
  CHECK(std::filesystem::file_size(empty_stl_path) == stl_bytes.length());
  CHECK(tf::cpp::read_stl(empty_stl_path.string()).size() == 0);

  const auto missing_obj = tf::cpp::read_obj<io_index_t, TestType>(
      "/tmp/trueform_cpp_io_file_that_does_not_exist.obj");
  const auto missing_stl =
      tf::cpp::read_stl("/tmp/trueform_cpp_io_file_that_does_not_exist.stl");
  CHECK(missing_obj.size() == 0);
  CHECK(missing_stl.size() == 0);

  const auto empty_triangles =
      tf::cpp::read_obj<io_index_t, TestType, 3>(nullptr, 0);
  const auto empty_mixed = tf::cpp::read_obj<io_index_t, TestType>(nullptr, 0);
  CHECK(empty_triangles.size() == 0);
  CHECK(empty_mixed.size() == 0);
  CHECK(empty_mixed.points_buffer().size() == 0);
}

TEST_CASE("malformed inputs preserve empty-result parsing behavior",
          "[cpp][io][malformed]") {
  auto malformed_obj = bytes_from_string("v 0 nope 0\nf 1 2 3\n");
  CHECK(tf::cpp::read_obj<io_index_t, float, 3>(
            io_bytes_from_array(malformed_obj))
            .size() == 0);
  CHECK(
      tf::cpp::read_obj<io_index_t, double>(io_bytes_from_array(malformed_obj))
          .size() == 0);

  auto malformed_stl =
      bytes_from_string("solid bad\nvertex nope 0 0\nendsolid bad");
  CHECK(tf::cpp::read_stl(io_bytes_from_array(malformed_stl)).size() == 0);

  const auto source = tf::cpp::test::triangle_mesh<float>();
  auto binary = tf::cpp::write_stl(source.mesh());
  CHECK(tf::cpp::read_stl(binary.raw_data(), binary.length() - 1).size() == 0);
}

/// A READ OWNS WHAT IT PARSES and a WRITE BORROWS WHAT IT READS: the bytes and
/// the path a reader is given are the job's from the moment it is submitted,
/// while a mesh is one reading of memory the caller keeps alive until the
/// future completes.
TEST_CASE("async IO has exact futures, custom resolvers, and owning captures",
          "[cpp][io][async][ownership]") {
  auto obj = bytes_from_string(triangle_obj());
  const auto source = tf::cpp::test::triangle_mesh<double>();
  tf::cpp::test::counting_resolver resolver;

  auto stl_read = tf::cpp::async::read_stl(obj.raw_data(), 0);
  auto obj_read =
      tf::cpp::async::read_obj<io_index_t, double, 3>(io_bytes_from_array(obj));
  auto obj_mixed = tf::cpp::async::read_obj<io_index_t, float>(
      resolver, obj.raw_data(), obj.length());
  auto stl_write = tf::cpp::async::write_stl(source.mesh());
  auto obj_write = tf::cpp::async::write_obj(source.mesh());
  STATIC_REQUIRE(
      std::is_same_v<decltype(stl_read), std::future<io_triangles<float>>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(obj_read), std::future<io_triangles<double>>>);
  STATIC_REQUIRE(
      std::is_same_v<decltype(obj_mixed),
                     std::future<tf::polygons_buffer<io_index_t, float, 3,
                                                     tf::dynamic_size>>>);
  STATIC_REQUIRE(std::is_same_v<decltype(stl_write),
                                std::future<tf::cpp::nd_array<std::int8_t>>>);
  STATIC_REQUIRE(std::is_same_v<decltype(obj_write),
                                std::future<tf::cpp::nd_array<std::int8_t>>>);
  CHECK(resolver.submissions() == 1);

  obj.destroy();
  CHECK(stl_read.get().size() == 0);
  CHECK(obj_read.get().size() == 1);
  CHECK(obj_mixed.get().size() == 1);
  CHECK(stl_write.get().length() == 134);
  CHECK_FALSE(obj_write.get().empty());

  std::vector<std::int8_t> ephemeral;
  const auto text = triangle_obj();
  ephemeral.assign(text.begin(), text.end());
  auto span_future = tf::cpp::async::read_obj<io_index_t, float, 3>(
      ephemeral.data(), ephemeral.size());
  ephemeral.clear();
  ephemeral.shrink_to_fit();
  CHECK(span_future.get().size() == 1);
}

TEMPLATE_TEST_CASE(
    "filesystem writers support synchronous future and resolver paths",
    "[cpp][io][write][path][async]", float, double) {
  tf::cpp::test::temporary_directory directory("trueform-io-path-async");
  const auto source = tf::cpp::test::triangle_mesh<TestType>();

  const auto sync_obj_base = directory.path() / "sync-obj";
  auto sync_obj_path = sync_obj_base;
  sync_obj_path += ".obj";
  const auto sync_stl_base = directory.path() / "sync-stl";
  auto sync_stl_path = sync_stl_base;
  sync_stl_path += ".stl";
  REQUIRE(tf::cpp::write_obj(source.mesh(), sync_obj_base));
  REQUIRE(tf::cpp::write_stl(source.mesh(), sync_stl_base));
  CHECK(std::filesystem::exists(sync_obj_path));
  CHECK(std::filesystem::exists(sync_stl_path));

  auto future_obj_input = directory.path() / "future-obj";
  auto future_obj_path = future_obj_input;
  future_obj_path += ".obj";
  auto future_stl_input = directory.path() / "future-stl";
  auto future_stl_path = future_stl_input;
  future_stl_path += ".stl";
  auto obj_future = tf::cpp::async::write_obj(source.mesh(), future_obj_input);
  auto stl_future = tf::cpp::async::write_stl(source.mesh(), future_stl_input);
  STATIC_REQUIRE(std::is_same_v<decltype(obj_future), std::future<bool>>);
  STATIC_REQUIRE(std::is_same_v<decltype(stl_future), std::future<bool>>);
  future_obj_input.clear();
  future_stl_input.clear();
  CHECK(obj_future.get());
  CHECK(stl_future.get());
  CHECK(std::filesystem::exists(future_obj_path));
  CHECK(std::filesystem::exists(future_stl_path));

  tf::cpp::test::counting_resolver resolver;
  const auto resolver_obj_base = directory.path() / "resolver-obj";
  auto resolver_obj_path = resolver_obj_base;
  resolver_obj_path += ".obj";
  const auto resolver_stl_base = directory.path() / "resolver-stl";
  auto resolver_stl_path = resolver_stl_base;
  resolver_stl_path += ".stl";
  auto obj_resolved =
      tf::cpp::async::write_obj(resolver, source.mesh(), resolver_obj_base);
  auto stl_resolved =
      tf::cpp::async::write_stl(resolver, source.mesh(), resolver_stl_base);
  STATIC_REQUIRE(std::is_same_v<decltype(obj_resolved), std::future<bool>>);
  STATIC_REQUIRE(std::is_same_v<decltype(stl_resolved), std::future<bool>>);
  CHECK(resolver.submissions() == 2);
  CHECK(obj_resolved.get());
  CHECK(stl_resolved.get());
  CHECK(std::filesystem::exists(resolver_obj_path));
  CHECK(std::filesystem::exists(resolver_stl_path));
}

TEST_CASE("owning IO bytes preserve size, sentinel, and moved storage",
          "[cpp][io][bytes][ownership]") {
  STATIC_REQUIRE(std::is_move_constructible_v<tf::cpp::io_bytes>);
  STATIC_REQUIRE(!std::is_copy_constructible_v<tf::cpp::io_bytes>);

  auto text = triangle_obj();
  auto bytes = tf::cpp::io_bytes::allocate(text.size());
  STATIC_REQUIRE(std::is_same_v<decltype(bytes.size()), std::size_t>);
  std::memcpy(bytes.data(), text.data(), text.size());
  auto *storage = bytes.data();
  CHECK(bytes.size() == text.size());
  CHECK(bytes.data()[bytes.size()] == 0);

  auto moved = std::move(bytes);
  CHECK(moved.data() == storage);
  CHECK(moved.size() == text.size());
  CHECK(tf::cpp::read_obj<io_index_t, float, 3>(std::move(moved)).size() == 1);

  auto async_bytes = tf::cpp::io_bytes::copy(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
  CHECK(async_bytes.data()[async_bytes.size()] == 0);
  auto future =
      tf::cpp::async::read_obj<io_index_t, double, 3>(std::move(async_bytes));
  text.clear();
  CHECK(future.get().size() == 1);
}

TEST_CASE("move construction leaves reusable empty IO bytes",
          "[cpp][io][bytes][move]") {
  const auto text = triangle_obj();
  auto source = tf::cpp::io_bytes::copy(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
  auto *storage = source.data();

  auto destination = std::move(source);
  CHECK(destination.data() == storage);
  check_empty_io_bytes(source);
  CHECK(
      tf::cpp::read_obj<io_index_t, float, 3>(std::move(destination)).size() ==
      1);
  check_empty_io_bytes(destination);

  CHECK(tf::cpp::read_obj<io_index_t, float, 3>(std::move(source)).size() == 0);
  check_empty_io_bytes(source);
  CHECK(tf::cpp::read_stl(std::move(source)).size() == 0);
  check_empty_io_bytes(source);
}

TEST_CASE("move assignment leaves source as reusable empty IO bytes",
          "[cpp][io][bytes][move]") {
  const auto text = triangle_obj();
  auto source = tf::cpp::io_bytes::copy(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
  auto destination = tf::cpp::io_bytes::allocate(4);
  auto *storage = source.data();

  destination = std::move(source);
  CHECK(destination.data() == storage);
  check_empty_io_bytes(source);
  CHECK(
      tf::cpp::read_obj<io_index_t, double, 3>(std::move(destination)).size() ==
      1);
  check_empty_io_bytes(destination);

  destination.data()[0] = 42;
  destination = std::move(source);
  check_empty_io_bytes(destination);
  check_empty_io_bytes(source);
}

TEST_CASE("self-moved IO bytes keep the payload they hold",
          "[cpp][io][bytes][move]") {
  const auto text = triangle_obj();
  auto bytes = tf::cpp::io_bytes::copy(
      reinterpret_cast<const std::int8_t *>(text.data()), text.size());
  auto *alias = &bytes;
  const auto *storage = bytes.data();

  bytes = std::move(*alias);
  CHECK(bytes.data() == storage);
  CHECK(bytes.size() == text.size());
  CHECK(tf::cpp::read_obj<io_index_t, float, 3>(std::move(bytes)).size() == 1);
}

TEST_CASE("IO byte sentinel allocation rejects size overflow before access",
          "[cpp][io][bytes][boundary]") {
  const auto max_size = std::numeric_limits<std::size_t>::max();
  const auto *unreadable =
      reinterpret_cast<const std::int8_t *>(std::uintptr_t{1});

  CHECK_THROWS_AS(tf::cpp::io_bytes::allocate(max_size), std::length_error);
  CHECK_THROWS_AS(tf::cpp::io_bytes::copy(unreadable, max_size),
                  std::length_error);
  CHECK_THROWS_AS(
      (tf::cpp::read_obj<io_index_t, float, 3>(unreadable, max_size)),
      std::length_error);
  CHECK_THROWS_AS(tf::cpp::read_stl(unreadable, max_size), std::length_error);
}

TEMPLATE_TEST_CASE(
    "filesystem writers answer the empty mesh and report file failures",
    "[cpp][io][write][path][failure]", float, double) {
  tf::cpp::test::temporary_directory directory("trueform-io-path-failure");
  // an OBJ of nothing is nothing, so the writer says so and writes no file,
  // while an STL of nothing is a header
  const auto nothing = tf::cpp::test::empty_mesh<TestType>();

  const auto empty_obj_path = directory.path() / "empty.obj";
  const auto empty_stl_path = directory.path() / "empty.stl";
  CHECK_FALSE(tf::cpp::write_obj(nothing.mesh(), empty_obj_path));
  CHECK(tf::cpp::write_stl(nothing.mesh(), empty_stl_path));
  CHECK_FALSE(tf::cpp::async::write_obj(nothing.mesh(), empty_obj_path).get());
  CHECK(tf::cpp::async::write_stl(nothing.mesh(), empty_stl_path).get());
  CHECK_FALSE(std::filesystem::exists(empty_obj_path));
  CHECK(std::filesystem::exists(empty_stl_path));

  const auto missing_parent = directory.path() / "missing-parent";
  const auto source = tf::cpp::test::triangle_mesh<TestType>();
  CHECK_FALSE(tf::cpp::write_obj(source.mesh(), missing_parent / "sync-obj"));
  CHECK_FALSE(tf::cpp::write_stl(source.mesh(), missing_parent / "sync-stl"));
  auto failed_obj =
      tf::cpp::async::write_obj(source.mesh(), missing_parent / "async-obj");
  auto failed_stl =
      tf::cpp::async::write_stl(source.mesh(), missing_parent / "async-stl");
  CHECK_FALSE(failed_obj.get());
  CHECK_FALSE(failed_stl.get());
  CHECK_FALSE(std::filesystem::exists(missing_parent));
}

TEST_CASE("async IO owns paths and transports exceptions",
          "[cpp][io][async][exception]") {
  std::string missing = "/tmp/trueform_cpp_io_async_missing.obj";
  auto path_future = tf::cpp::async::read_obj<io_index_t, float, 3>(missing);
  missing.clear();
  CHECK(path_future.get().size() == 0);

  const auto nothing = tf::cpp::test::empty_mesh<float>();
  CHECK(tf::cpp::async::write_stl(nothing.mesh()).get().length() > 0);
  CHECK(tf::cpp::async::write_obj(nothing.mesh()).get().length() == 0);
}

// One entry per operation: a reader names the index, the real and the arity it
// builds, and a writer takes the mesh.
TEMPLATE_TEST_CASE("IO overloads link from the native archive",
                   "[cpp][io][archive-link]", float, double) {
  using triangles = io_triangles<TestType>;
  using mixed = tf::polygons_buffer<io_index_t, TestType, 3, tf::dynamic_size>;
  using triangle_mesh_view = tf::cpp::mesh<io_index_t, TestType, 3, 3>;

  auto (*obj_path)(std::string_view)->triangles =
      &tf::cpp::read_obj<io_index_t, TestType, 3>;
  auto (*obj_owned)(tf::cpp::io_bytes)->triangles =
      &tf::cpp::read_obj<io_index_t, TestType, 3>;
  auto (*obj_span)(const std::int8_t *, std::size_t)->triangles =
      &tf::cpp::read_obj<io_index_t, TestType, 3>;
  auto (*mixed_owned)(tf::cpp::io_bytes)->mixed =
      &tf::cpp::read_obj<io_index_t, TestType, tf::dynamic_size>;
  auto (*mixed_span)(const std::int8_t *, std::size_t)->mixed =
      &tf::cpp::read_obj<io_index_t, TestType, tf::dynamic_size>;
  auto (*writer)(const triangle_mesh_view &)->tf::cpp::nd_array<std::int8_t> =
      &tf::cpp::write_obj<io_index_t, TestType, 3, 3>;
  auto (*path_writer)(const triangle_mesh_view &, const std::filesystem::path &)
      ->bool = &tf::cpp::write_obj<io_index_t, TestType, 3, 3>;
  auto (*stl_writer)(const triangle_mesh_view &)
      ->tf::cpp::nd_array<std::int8_t> =
      &tf::cpp::write_stl<io_index_t, TestType, 3, 3>;
  auto (*stl_path_writer)(const triangle_mesh_view &,
                          const std::filesystem::path &)
      ->bool = &tf::cpp::write_stl<io_index_t, TestType, 3, 3>;

  auto bytes = bytes_from_string(triangle_obj());
  temporary_file file(".obj", bytes);
  CHECK(obj_path(file.path()).size() == 1);
  CHECK(obj_owned(io_bytes_from_array(bytes)).size() == 1);
  CHECK(obj_span(bytes.raw_data(), bytes.length()).size() == 1);
  CHECK(mixed_owned(io_bytes_from_array(bytes)).size() == 1);
  CHECK(mixed_span(bytes.raw_data(), bytes.length()).size() == 1);
  const auto written = tf::cpp::test::triangle_mesh<TestType>();
  CHECK_FALSE(writer(written.mesh()).empty());
  CHECK(stl_writer(written.mesh()).length() == 134);
  tf::cpp::test::temporary_directory directory("trueform-io-archive-link");
  CHECK(path_writer(written.mesh(), directory.path() / "archive.obj"));
  CHECK(stl_path_writer(written.mesh(), directory.path() / "archive.stl"));

  auto (*stl_path)(std::string_view)->io_triangles<float> =
      &tf::cpp::read_stl<>;
  auto (*stl_owned)(tf::cpp::io_bytes)->io_triangles<float> =
      &tf::cpp::read_stl<>;
  auto (*stl_span)(const std::int8_t *, std::size_t)->io_triangles<float> =
      &tf::cpp::read_stl<>;
  auto ascii = bytes_from_string(ascii_stl());
  temporary_file stl_file(".stl", ascii);
  CHECK(stl_path(stl_file.path()).size() == 1);
  CHECK(stl_owned(io_bytes_from_array(ascii)).size() == 1);
  CHECK(stl_span(ascii.raw_data(), ascii.length()).size() == 1);
}

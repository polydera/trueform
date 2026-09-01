/**
 * @file test_mapped_file.cpp
 * @brief Tests for the reusable read-only file mapping owner.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/io.hpp>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

auto temporary_path(std::string_view suffix) -> std::filesystem::path {
  static std::atomic<int> counter{0};
  static const auto process_id = std::random_device{}();
  const auto id = counter.fetch_add(1);
  const auto name = std::string("trueform_mapped_file_") +
                    std::to_string(process_id) + "_" + std::to_string(id) +
                    std::string(suffix);
  return std::filesystem::temp_directory_path() / name;
}

class temporary_file {
public:
  explicit temporary_file(std::string_view suffix)
      : _path(temporary_path(suffix)) {}

  temporary_file(const temporary_file &) = delete;
  auto operator=(const temporary_file &) -> temporary_file & = delete;

  ~temporary_file() {
    std::error_code error;
    std::filesystem::remove(_path, error);
  }

  auto path() const -> const std::filesystem::path & { return _path; }

  auto write(std::string_view contents) const -> bool {
    std::ofstream output(_path, std::ios::binary | std::ios::trunc);
    if (!output)
      return false;
    output.write(contents.data(),
                 static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(output);
  }

private:
  std::filesystem::path _path;
};

auto has_contents(const tf::io::mapped_file &file, std::string_view contents)
    -> bool {
  return file && file.size() == contents.size() &&
         std::memcmp(file.data(), contents.data(), contents.size()) == 0;
}

auto is_empty(const tf::io::mapped_file &file) -> bool {
  return !file && file.data() == nullptr && file.size() == 0;
}

} // namespace

static_assert(!std::is_copy_constructible_v<tf::io::mapped_file>);
static_assert(!std::is_copy_assignable_v<tf::io::mapped_file>);
static_assert(std::is_nothrow_move_constructible_v<tf::io::mapped_file>);
static_assert(std::is_nothrow_move_assignable_v<tf::io::mapped_file>);

TEST_CASE("mapped_file represents operational failures as an empty mapping",
          "[io][mapped_file]") {
  const temporary_file empty_file(".empty");
  REQUIRE(empty_file.write({}));

  const auto missing_path = temporary_path(".missing");
  std::error_code error;
  std::filesystem::remove(missing_path, error);

  const tf::io::mapped_file default_file;
  const tf::io::mapped_file empty(empty_file.path().string());
  const tf::io::mapped_file missing(missing_path.string());
  REQUIRE(is_empty(default_file));
  REQUIRE(is_empty(empty));
  REQUIRE(is_empty(missing));
}

TEST_CASE("mapped_file preserves narrow path prefix semantics",
          "[io][mapped_file]") {
  constexpr std::string_view contents = "mapped prefix";
  const temporary_file prefix_file(".prefix");
  REQUIRE(prefix_file.write(contents));

  auto embedded_null_path = prefix_file.path().string();
  embedded_null_path.push_back('\0');
  embedded_null_path += ".suffix";
  const tf::io::mapped_file embedded_null(embedded_null_path);
  REQUIRE(has_contents(embedded_null, contents));
}

TEST_CASE("mapped_file transfers mapping ownership", "[io][mapped_file]") {
  constexpr std::string_view first_contents = "first mapped file";
  constexpr std::string_view second_contents = "second mapped file";
  const temporary_file first_file(".first");
  const temporary_file second_file(".second");
  REQUIRE(first_file.write(first_contents));
  REQUIRE(second_file.write(second_contents));

  tf::io::mapped_file assigned;
  {
    tf::io::mapped_file first(first_file.path().string());
    REQUIRE(has_contents(first, first_contents));

    tf::io::mapped_file moved(std::move(first));
    REQUIRE(is_empty(first));
    REQUIRE(has_contents(moved, first_contents));

    assigned = tf::io::mapped_file(second_file.path().string());
    REQUIRE(has_contents(assigned, second_contents));
    assigned = std::move(moved);
    REQUIRE(is_empty(moved));
    REQUIRE(has_contents(assigned, first_contents));
  }
  REQUIRE(has_contents(assigned, first_contents));

  std::error_code error;
  REQUIRE(std::filesystem::remove(second_file.path(), error));
  REQUIRE_FALSE(error);
  REQUIRE(second_file.write("replacement"));
  REQUIRE(has_contents(assigned, first_contents));
}

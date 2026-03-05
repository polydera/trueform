/**
 * @file test_read_obj.cpp
 * @brief Tests for OBJ reading functionality
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/io.hpp>
#include <atomic>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

namespace {

auto temp_obj_path() -> std::filesystem::path {
    static std::atomic<int> counter{0};
    static auto process_id = std::random_device{}();
    auto id = counter.fetch_add(1);
    auto name = std::string("trueform_test_") + std::to_string(process_id) + "_" + std::to_string(id) + ".obj";
    return std::filesystem::temp_directory_path() / name;
}

struct TempFileCleanup {
    std::filesystem::path path;
    ~TempFileCleanup() {
        std::filesystem::remove(path);
    }
};

void create_triangle_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Simple triangle
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
)";
}

void create_quad_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Simple quad
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
)";
}

void create_cube_triangles_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Cube with triangles
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 6 2
f 1 5 6
f 3 8 4
f 3 7 8
f 1 4 8
f 1 8 5
f 2 7 3
f 2 6 7
)";
}

void create_cube_quads_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Cube with quads
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 3 7 8 4
f 1 4 8 5
f 2 6 7 3
)";
}

void create_mixed_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Mixed: triangles and quads
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
f 1 2 3
f 1 3 4
f 1 2 5 3
)";
}

void create_face_formats_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# Triangle with texture coords and normals
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
)";
}

void create_comments_obj(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(# This is a comment
# Another comment
v 0 0 0
v 1 0 0
v 0 1 0
# Comment between data
f 1 2 3
# Trailing comment
)";
}

} // namespace

// =============================================================================
// read_obj fixed size (triangles)
// =============================================================================

TEMPLATE_TEST_CASE("read_obj<3> simple triangle", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_triangle_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t, 3>(path.string());

    auto faces = polygons.faces();
    auto points = polygons.points();

    REQUIRE(faces.size() == 1);
    REQUIRE(points.size() == 3);

    // Check 0-based indices
    REQUIRE(faces[0][0] == 0);
    REQUIRE(faces[0][1] == 1);
    REQUIRE(faces[0][2] == 2);

    // Check vertex positions
    REQUIRE(points[0][0] == 0.0f);
    REQUIRE(points[0][1] == 0.0f);
    REQUIRE(points[0][2] == 0.0f);
    REQUIRE(points[1][0] == 1.0f);
    REQUIRE(points[2][1] == 1.0f);
}

TEMPLATE_TEST_CASE("read_obj<3> cube", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_cube_triangles_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t, 3>(path.string());

    REQUIRE(polygons.faces().size() == 12);
    REQUIRE(polygons.points().size() == 8);

    for (std::size_t i = 0; i < polygons.faces().size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(polygons.faces()[i][j] >= 0);
            REQUIRE(static_cast<std::size_t>(polygons.faces()[i][j]) < 8);
        }
    }
}

// =============================================================================
// read_obj fixed size (quads)
// =============================================================================

TEMPLATE_TEST_CASE("read_obj<4> simple quad", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_quad_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t, 4>(path.string());

    REQUIRE(polygons.faces().size() == 1);
    REQUIRE(polygons.points().size() == 4);

    REQUIRE(polygons.faces()[0][0] == 0);
    REQUIRE(polygons.faces()[0][1] == 1);
    REQUIRE(polygons.faces()[0][2] == 2);
    REQUIRE(polygons.faces()[0][3] == 3);
}

TEMPLATE_TEST_CASE("read_obj<4> cube quads", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_cube_quads_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t, 4>(path.string());

    REQUIRE(polygons.faces().size() == 6);
    REQUIRE(polygons.points().size() == 8);
}

// =============================================================================
// read_obj fixed - mixed file should fail
// =============================================================================

TEST_CASE("read_obj<3> on mixed file returns empty", "[io][read_obj]") {
    auto path = temp_obj_path();
    create_mixed_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<3>(path.string());
    REQUIRE(polygons.faces().size() == 0);
}

// =============================================================================
// read_obj dynamic
// =============================================================================

TEMPLATE_TEST_CASE("read_obj dynamic - pure triangles", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_cube_triangles_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t>(path.string());

    REQUIRE(polygons.faces().size() == 12);
    REQUIRE(polygons.points().size() == 8);

    for (std::size_t i = 0; i < polygons.faces().size(); ++i) {
        REQUIRE(polygons.faces()[i].size() == 3);
    }
}

TEMPLATE_TEST_CASE("read_obj dynamic - pure quads", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_cube_quads_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t>(path.string());

    REQUIRE(polygons.faces().size() == 6);
    REQUIRE(polygons.points().size() == 8);

    for (std::size_t i = 0; i < polygons.faces().size(); ++i) {
        REQUIRE(polygons.faces()[i].size() == 4);
    }
}

TEMPLATE_TEST_CASE("read_obj dynamic - mixed polygons", "[io][read_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    create_mixed_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<index_t>(path.string());

    REQUIRE(polygons.faces().size() == 3);
    REQUIRE(polygons.points().size() == 5);

    // First two faces are triangles, third is a quad
    REQUIRE(polygons.faces()[0].size() == 3);
    REQUIRE(polygons.faces()[1].size() == 3);
    REQUIRE(polygons.faces()[2].size() == 4);

    // Check 0-based indices of first face
    REQUIRE(polygons.faces()[0][0] == 0);
    REQUIRE(polygons.faces()[0][1] == 1);
    REQUIRE(polygons.faces()[0][2] == 2);
}

// =============================================================================
// Face formats (v/vt/vn)
// =============================================================================

TEST_CASE("read_obj face formats v/vt/vn", "[io][read_obj]") {
    auto path = temp_obj_path();
    create_face_formats_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<3>(path.string());

    REQUIRE(polygons.faces().size() == 1);
    REQUIRE(polygons.points().size() == 3);

    REQUIRE(polygons.faces()[0][0] == 0);
    REQUIRE(polygons.faces()[0][1] == 1);
    REQUIRE(polygons.faces()[0][2] == 2);
}

// =============================================================================
// Comments
// =============================================================================

TEST_CASE("read_obj with comments", "[io][read_obj]") {
    auto path = temp_obj_path();
    create_comments_obj(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_obj<3>(path.string());

    REQUIRE(polygons.faces().size() == 1);
    REQUIRE(polygons.points().size() == 3);
}

// =============================================================================
// Error cases
// =============================================================================

TEST_CASE("read_obj nonexistent file", "[io][read_obj]") {
    auto result = tf::read_obj<3>("/nonexistent/path/file.obj");
    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

TEST_CASE("read_obj dynamic nonexistent file", "[io][read_obj]") {
    auto result = tf::read_obj("/nonexistent/path/file.obj");
    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

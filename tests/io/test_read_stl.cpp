/**
 * @file test_read_stl.cpp
 * @brief Tests for STL reading functionality
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

// Helper to create a unique temporary file path
auto temp_stl_path() -> std::filesystem::path {
    static std::atomic<int> counter{0};
    static auto process_id = std::random_device{}();
    auto id = counter.fetch_add(1);
    auto name = std::string("trueform_test_") + std::to_string(process_id) + "_" + std::to_string(id) + ".stl";
    return std::filesystem::temp_directory_path() / name;
}

// Create a simple ASCII STL with one triangle
void create_simple_stl(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
)";
}

// Create an ASCII STL cube (12 triangles, 8 vertices after deduplication)
void create_cube_stl(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 1 0 0
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 0
      vertex 0 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 1 0
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
endsolid cube
)";
}

// Create STL with two triangles sharing vertices (for deduplication test)
void create_shared_vertices_stl(const std::filesystem::path& path) {
    std::ofstream f(path);
    f << R"(solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test
)";
}

// RAII cleanup for temp files
struct TempFileCleanup {
    std::filesystem::path path;
    ~TempFileCleanup() {
        std::filesystem::remove(path);
    }
};

} // namespace

// =============================================================================
// read_stl tests
// =============================================================================

TEMPLATE_TEST_CASE("read_stl simple triangle", "[io][read_stl]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_stl_path();
    create_simple_stl(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_stl<index_t>(path.string());

    auto faces = polygons.faces();
    auto points = polygons.points();

    // Check we have at least one face
    REQUIRE(faces.size() >= 1);

    // Check face has 3 vertices (triangle)
    REQUIRE(faces[0].size() == 3);

    // Check we have at least 3 points
    REQUIRE(points.size() >= 3);

    // Check face indices are valid
    for (std::size_t i = 0; i < faces.size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(faces[i][j] >= 0);
            REQUIRE(static_cast<std::size_t>(faces[i][j]) < points.size());
        }
    }
}

TEMPLATE_TEST_CASE("read_stl cube", "[io][read_stl]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_stl_path();
    create_cube_stl(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_stl<index_t>(path.string());

    auto faces = polygons.faces();
    auto points = polygons.points();

    // Cube has 12 triangles
    REQUIRE(faces.size() == 12);

    // Cube has 8 unique vertices after deduplication
    REQUIRE(points.size() == 8);

    // Check all face indices are valid (0-7)
    for (std::size_t i = 0; i < faces.size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(faces[i][j] >= 0);
            REQUIRE(faces[i][j] < 8);
        }
    }

    // Check points are within [0, 1] range
    for (std::size_t i = 0; i < points.size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(points[i][j] >= 0.0f);
            REQUIRE(points[i][j] <= 1.0f);
        }
    }
}

TEMPLATE_TEST_CASE("read_stl vertex deduplication", "[io][read_stl]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_stl_path();
    create_shared_vertices_stl(path);
    TempFileCleanup cleanup{path};

    auto polygons = tf::read_stl<index_t>(path.string());

    auto faces = polygons.faces();
    auto points = polygons.points();

    // Two triangles
    REQUIRE(faces.size() == 2);

    // 4 unique vertices: (0,0,0), (1,0,0), (0,1,0), (1,1,0)
    REQUIRE(points.size() == 4);

    // Count shared vertices between the two faces
    int shared = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (faces[0][i] == faces[1][j]) {
                ++shared;
            }
        }
    }
    // Should share exactly 2 vertices: (1,0,0) and (0,1,0)
    REQUIRE(shared == 2);
}

TEST_CASE("read_stl nonexistent file", "[io][read_stl]") {
    auto result = tf::read_stl("/nonexistent/path/file.stl");
    // Nonexistent file returns empty mesh
    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

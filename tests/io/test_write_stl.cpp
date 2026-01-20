/**
 * @file test_write_stl.cpp
 * @brief Tests for STL writing functionality
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/io.hpp>
#include <trueform/core.hpp>
#include <atomic>
#include <filesystem>
#include <random>
#include "canonicalize_mesh.hpp"

namespace {

auto temp_stl_path() -> std::filesystem::path {
    static std::atomic<int> counter{0};
    static auto process_id = std::random_device{}();
    auto id = counter.fetch_add(1);
    auto name = std::string("trueform_write_test_") + std::to_string(process_id) + "_" + std::to_string(id) + ".stl";
    return std::filesystem::temp_directory_path() / name;
}

struct TempFileCleanup {
    std::filesystem::path path;
    ~TempFileCleanup() {
        std::filesystem::remove(path);
    }
};

// Helper to create a simple triangle mesh buffer
template <typename IndexT>
auto make_triangle_mesh() -> tf::polygons_buffer<IndexT, float, 3, 3> {
    tf::polygons_buffer<IndexT, float, 3, 3> mesh;
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(2)});
    return mesh;
}

// Helper to create a cube mesh buffer
template <typename IndexT>
auto make_cube_mesh() -> tf::polygons_buffer<IndexT, float, 3, 3> {
    tf::polygons_buffer<IndexT, float, 3, 3> mesh;
    // 8 vertices
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 0.0f, 1.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 1.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 1.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 1.0f});
    // 12 triangles
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(2)});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(2), IndexT(3)});
    mesh.faces_buffer().push_back({IndexT(4), IndexT(6), IndexT(5)});
    mesh.faces_buffer().push_back({IndexT(4), IndexT(7), IndexT(6)});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(5), IndexT(1)});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(4), IndexT(5)});
    mesh.faces_buffer().push_back({IndexT(2), IndexT(7), IndexT(3)});
    mesh.faces_buffer().push_back({IndexT(2), IndexT(6), IndexT(7)});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(3), IndexT(7)});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(7), IndexT(4)});
    mesh.faces_buffer().push_back({IndexT(1), IndexT(6), IndexT(2)});
    mesh.faces_buffer().push_back({IndexT(1), IndexT(5), IndexT(6)});
    return mesh;
}

} // namespace

// =============================================================================
// write_stl tests
// =============================================================================

TEMPLATE_TEST_CASE("write_stl simple triangle", "[io][write_stl]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_stl_path();
    TempFileCleanup cleanup{path};

    auto mesh = make_triangle_mesh<index_t>();

    bool success = tf::write_stl(mesh.polygons(), path.string());
    REQUIRE(success);

    // Verify file was created and is not empty
    REQUIRE(std::filesystem::exists(path));
    REQUIRE(std::filesystem::file_size(path) > 0);
}

TEMPLATE_TEST_CASE("write_stl round trip", "[io][write_stl]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_stl_path();
    TempFileCleanup cleanup{path};

    auto mesh_orig = make_cube_mesh<index_t>();

    // Write
    bool success = tf::write_stl(mesh_orig.polygons(), path.string());
    REQUIRE(success);

    // Read back
    auto mesh_read = tf::read_stl<index_t>(path.string());

    // Canonicalize both meshes and compare
    auto canonical_orig = tf::test::canonicalize_mesh(mesh_orig);
    auto canonical_read = tf::test::canonicalize_mesh(mesh_read);

    REQUIRE(tf::test::meshes_equal(canonical_orig, canonical_read));
}

TEST_CASE("write_stl appends extension", "[io][write_stl]") {
    auto base_path = std::filesystem::temp_directory_path() / "trueform_no_ext";
    auto expected_path = std::filesystem::temp_directory_path() / "trueform_no_ext.stl";

    // Cleanup both possible paths
    std::filesystem::remove(base_path);
    std::filesystem::remove(expected_path);

    auto mesh = make_triangle_mesh<int>();
    bool success = tf::write_stl(mesh.polygons(), base_path.string());
    REQUIRE(success);

    // The .stl extension should be appended
    REQUIRE(std::filesystem::exists(expected_path));

    // Cleanup
    std::filesystem::remove(expected_path);
}

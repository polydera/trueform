/**
 * @file test_write_obj.cpp
 * @brief Tests for OBJ writing functionality
 *
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <trueform/io.hpp>
#include <trueform/core.hpp>
#include <atomic>
#include <filesystem>
#include <random>

namespace {

auto temp_obj_path() -> std::filesystem::path {
    static std::atomic<int> counter{0};
    static auto process_id = std::random_device{}();
    auto id = counter.fetch_add(1);
    auto name = std::string("trueform_write_test_") + std::to_string(process_id) + "_" + std::to_string(id) + ".obj";
    return std::filesystem::temp_directory_path() / name;
}

struct TempFileCleanup {
    std::filesystem::path path;
    ~TempFileCleanup() {
        std::filesystem::remove(path);
    }
};

template <typename IndexT>
auto make_triangle_mesh() -> tf::polygons_buffer<IndexT, float, 3, 3> {
    tf::polygons_buffer<IndexT, float, 3, 3> mesh;
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(2)});
    return mesh;
}

template <typename IndexT>
auto make_quad_mesh() -> tf::polygons_buffer<IndexT, float, 3, 4> {
    tf::polygons_buffer<IndexT, float, 3, 4> mesh;
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(2), IndexT(3)});
    return mesh;
}

template <typename IndexT>
auto make_cube_mesh() -> tf::polygons_buffer<IndexT, float, 3, 3> {
    tf::polygons_buffer<IndexT, float, 3, 3> mesh;
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 0.0f, 1.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 1.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 1.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 1.0f});
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

template <typename IndexT>
auto make_dynamic_mesh() -> tf::polygons_buffer<IndexT, float, 3, tf::dynamic_size> {
    tf::polygons_buffer<IndexT, float, 3, tf::dynamic_size> mesh;
    mesh.points_buffer().push_back({0.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 0.0f, 0.0f});
    mesh.points_buffer().push_back({1.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({0.0f, 1.0f, 0.0f});
    mesh.points_buffer().push_back({2.0f, 0.0f, 0.0f});
    // triangle
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(2)});
    // quad
    mesh.faces_buffer().push_back({IndexT(0), IndexT(1), IndexT(4), IndexT(2)});
    return mesh;
}

} // namespace

// =============================================================================
// write_obj triangles
// =============================================================================

TEMPLATE_TEST_CASE("write_obj simple triangle", "[io][write_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};

    auto mesh = make_triangle_mesh<index_t>();

    bool success = tf::write_obj(mesh.polygons(), path.string());
    REQUIRE(success);

    REQUIRE(std::filesystem::exists(path));
    REQUIRE(std::filesystem::file_size(path) > 0);
}

// =============================================================================
// write_obj quads
// =============================================================================

TEMPLATE_TEST_CASE("write_obj simple quad", "[io][write_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};

    auto mesh = make_quad_mesh<index_t>();

    bool success = tf::write_obj(mesh.polygons(), path.string());
    REQUIRE(success);

    REQUIRE(std::filesystem::exists(path));
    REQUIRE(std::filesystem::file_size(path) > 0);
}

// =============================================================================
// Round-trip: write → read (fixed triangles)
// =============================================================================

TEMPLATE_TEST_CASE("write_obj round trip triangles", "[io][write_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};

    auto mesh_orig = make_cube_mesh<index_t>();

    bool success = tf::write_obj(mesh_orig.polygons(), path.string());
    REQUIRE(success);

    auto mesh_read = tf::read_obj<index_t, 3>(path.string());

    REQUIRE(mesh_read.faces().size() == mesh_orig.faces().size());
    REQUIRE(mesh_read.points().size() == mesh_orig.points().size());

    // Check points match
    for (std::size_t i = 0; i < mesh_orig.points().size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(mesh_read.points()[i][j] ==
                    Catch::Approx(mesh_orig.points()[i][j]).margin(1e-5f));
        }
    }

    // Check faces match
    for (std::size_t i = 0; i < mesh_orig.faces().size(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            REQUIRE(mesh_read.faces()[i][j] == mesh_orig.faces()[i][j]);
        }
    }
}

// =============================================================================
// Round-trip: write → read (fixed quads)
// =============================================================================

TEMPLATE_TEST_CASE("write_obj round trip quads", "[io][write_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};

    auto mesh_orig = make_quad_mesh<index_t>();

    bool success = tf::write_obj(mesh_orig.polygons(), path.string());
    REQUIRE(success);

    auto mesh_read = tf::read_obj<index_t, 4>(path.string());

    REQUIRE(mesh_read.faces().size() == mesh_orig.faces().size());
    REQUIRE(mesh_read.points().size() == mesh_orig.points().size());

    for (std::size_t i = 0; i < mesh_orig.faces().size(); ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            REQUIRE(mesh_read.faces()[i][j] == mesh_orig.faces()[i][j]);
        }
    }
}

// =============================================================================
// Round-trip: write → read (dynamic)
// =============================================================================

TEMPLATE_TEST_CASE("write_obj round trip dynamic", "[io][write_obj]",
    std::int32_t, std::int64_t)
{
    using index_t = TestType;

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};

    auto mesh_orig = make_dynamic_mesh<index_t>();

    bool success = tf::write_obj(mesh_orig.polygons(), path.string());
    REQUIRE(success);

    // Read back as dynamic
    auto mesh_read = tf::read_obj<index_t>(path.string());

    REQUIRE(mesh_read.faces().size() == mesh_orig.faces().size());
    REQUIRE(mesh_read.points().size() == mesh_orig.points().size());

    // Check face sizes match
    for (std::size_t i = 0; i < mesh_orig.faces().size(); ++i) {
        REQUIRE(mesh_read.faces()[i].size() == mesh_orig.faces()[i].size());
        for (std::size_t j = 0; j < mesh_orig.faces()[i].size(); ++j) {
            REQUIRE(mesh_read.faces()[i][j] == mesh_orig.faces()[i][j]);
        }
    }
}

// =============================================================================
// Extension appending
// =============================================================================

TEST_CASE("write_obj appends extension", "[io][write_obj]") {
    auto base_path = std::filesystem::temp_directory_path() / "trueform_no_ext_obj";
    auto expected_path = std::filesystem::temp_directory_path() / "trueform_no_ext_obj.obj";

    std::filesystem::remove(base_path);
    std::filesystem::remove(expected_path);

    auto mesh = make_triangle_mesh<int>();
    bool success = tf::write_obj(mesh.polygons(), base_path.string());
    REQUIRE(success);

    REQUIRE(std::filesystem::exists(expected_path));

    std::filesystem::remove(expected_path);
}

// =============================================================================
// write_obj_to_buffer
// =============================================================================

TEST_CASE("write_obj_to_buffer produces valid data", "[io][write_obj]") {
    auto mesh = make_triangle_mesh<int>();
    auto buf = tf::write_obj_to_buffer(mesh.polygons());

    REQUIRE(buf.size() > 0);

    // Parse the buffer content — should contain "v " and "f " lines
    std::string content(buf.begin(), buf.begin() + buf.size());
    REQUIRE(content.find("v ") != std::string::npos);
    REQUIRE(content.find("f ") != std::string::npos);
}

// =============================================================================
// Double-precision (float64) round-trip
// =============================================================================

TEST_CASE("write_obj + read_obj double-precision round-trip", "[io][write_obj][float64]") {
    // Coordinates that cannot be represented exactly as float32.
    tf::polygons_buffer<int, double, 3, 3> mesh;
    mesh.points_buffer().push_back({1.0000000000000002, 0.0, 0.0});
    mesh.points_buffer().push_back({0.0, 2.0000000000000004, 0.0});
    mesh.points_buffer().push_back({0.0, 0.0, 3.0000000000000007});
    mesh.faces_buffer().push_back({0, 1, 2});

    auto path = temp_obj_path();
    TempFileCleanup cleanup{path};
    REQUIRE(tf::write_obj(mesh.polygons(), path.string()));

    auto mesh_read = tf::read_obj<int, 3, double>(path.string());
    REQUIRE(mesh_read.points().size() == 3);

    auto p0 = mesh_read.points().front();
    REQUIRE(p0[0] == 1.0000000000000002);
    REQUIRE(p0[1] == 0.0);
    REQUIRE(p0[2] == 0.0);

    auto p1 = mesh_read.points()[1];
    REQUIRE(p1[1] == 2.0000000000000004);

    auto p2 = mesh_read.points()[2];
    REQUIRE(p2[2] == 3.0000000000000007);
}

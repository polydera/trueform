/**
 * @file test_triangulation.cpp
 * @brief Tests for triangulation functions
 *
 * Tests for:
 * - triangulated_faces
 * - triangulated (polygon mesh)
 *
 * Key verification: area preservation after triangulation using tf::area
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <cmath>
#include <iostream>

namespace {

/**
 * @brief Create a unit square quad mesh (single quad face)
 */
template <typename Index, typename Real>
auto create_unit_quad() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));

    return result;
}

/**
 * @brief Create a regular pentagon
 */
template <typename Index, typename Real>
auto create_pentagon() -> tf::polygons_buffer<Index, Real, 3, 5> {
    tf::polygons_buffer<Index, Real, 3, 5> result;

    Real pi = Real(3.14159265358979323846);
    for (int i = 0; i < 5; ++i) {
        Real angle = Real(2) * pi * Real(i) / Real(5);
        result.points_buffer().emplace_back(std::cos(angle), std::sin(angle), Real(0));
    }

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3), Index(4));

    return result;
}

/**
 * @brief Create a regular hexagon
 */
template <typename Index, typename Real>
auto create_hexagon() -> tf::polygons_buffer<Index, Real, 3, 6> {
    tf::polygons_buffer<Index, Real, 3, 6> result;

    Real pi = Real(3.14159265358979323846);
    for (int i = 0; i < 6; ++i) {
        Real angle = Real(2) * pi * Real(i) / Real(6);
        result.points_buffer().emplace_back(std::cos(angle), std::sin(angle), Real(0));
    }

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3), Index(4), Index(5));

    return result;
}

/**
 * @brief Create two quads sharing an edge
 */
template <typename Index, typename Real>
auto create_two_quads() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(2), Real(1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(1), Index(4), Index(5), Index(2));

    return result;
}

/**
 * @brief Create a mixed mesh with triangle and quad
 */
template <typename Index, typename Real>
auto create_mixed_mesh() -> tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> {
    tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> result;

    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));   // 0
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));   // 1
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));   // 2
    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));   // 3
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));   // 4

    // Triangle
    result.faces_buffer().push_back({Index(0), Index(2), Index(3)});
    // Quad
    result.faces_buffer().push_back({Index(0), Index(1), Index(4), Index(2)});

    return result;
}

} // anonymous namespace

// =============================================================================
// Single Quad - Triangle Count
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_quad_triangle_count", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quad = create_unit_quad<index_t, real_t>();
    auto tri_mesh = tf::triangulated(quad.polygons());

    // 1 quad → 2 triangles
    REQUIRE(tri_mesh.faces().size() == 2);
    REQUIRE(tri_mesh.points().size() == 4);
}

// =============================================================================
// Single Quad - Area Preservation
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_quad_area_preserved", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quad = create_unit_quad<index_t, real_t>();
    real_t original_area = tf::area(quad.polygons());

    auto tri_mesh = tf::triangulated(quad.polygons());
    real_t tri_area = tf::area(tri_mesh.polygons());

    // Debug output
    std::cout << "=== triangulated_quad_area_preserved ===" << std::endl;
    std::cout << "Original points:" << std::endl;
    for (std::size_t i = 0; i < quad.points().size(); ++i) {
        std::cout << "  " << i << ": (" << quad.points()[i][0] << ", " << quad.points()[i][1] << ", " << quad.points()[i][2] << ")" << std::endl;
    }
    std::cout << "Original area: " << original_area << std::endl;
    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    for (std::size_t i = 0; i < tri_mesh.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated points:" << std::endl;
    for (std::size_t i = 0; i < tri_mesh.points().size(); ++i) {
        std::cout << "  " << i << ": (" << tri_mesh.points()[i][0] << ", " << tri_mesh.points()[i][1] << ", " << tri_mesh.points()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated area: " << tri_area << std::endl;

    // Unit square has area 1.0
    REQUIRE(std::abs(original_area - real_t(1)) < real_t(1e-5));
    REQUIRE(std::abs(tri_area - original_area) < real_t(1e-5));
}

// =============================================================================
// Pentagon - Triangle Count and Area
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_pentagon", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto pentagon = create_pentagon<index_t, real_t>();
    real_t original_area = tf::area(pentagon.polygons());

    auto tri_mesh = tf::triangulated(pentagon.polygons());

    // Debug output
    std::cout << "=== triangulated_pentagon ===" << std::endl;
    std::cout << "Original points:" << std::endl;
    for (std::size_t i = 0; i < pentagon.points().size(); ++i) {
        std::cout << "  " << i << ": (" << pentagon.points()[i][0] << ", " << pentagon.points()[i][1] << ", " << pentagon.points()[i][2] << ")" << std::endl;
    }
    std::cout << "Original area: " << original_area << std::endl;
    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    for (std::size_t i = 0; i < tri_mesh.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated points:" << std::endl;
    for (std::size_t i = 0; i < tri_mesh.points().size(); ++i) {
        std::cout << "  " << i << ": (" << tri_mesh.points()[i][0] << ", " << tri_mesh.points()[i][1] << ", " << tri_mesh.points()[i][2] << ")" << std::endl;
    }
    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;

    // 5-gon → 3 triangles
    REQUIRE(tri_mesh.faces().size() == 3);
    REQUIRE(tri_mesh.points().size() == 5);

    // Area preserved
    REQUIRE(std::abs(tri_area - original_area) < real_t(1e-5));
}

// =============================================================================
// Hexagon - Area Preservation
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_hexagon_area_preserved", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto hexagon = create_hexagon<index_t, real_t>();

    // Debug output
    std::cout << "=== triangulated_hexagon_area_preserved ===" << std::endl;
    std::cout << "Original points:" << std::endl;
    for (std::size_t i = 0; i < hexagon.points().size(); ++i) {
        std::cout << "  " << i << ": (" << hexagon.points()[i][0] << ", " << hexagon.points()[i][1] << ", " << hexagon.points()[i][2] << ")" << std::endl;
    }

    // Regular hexagon with unit radius has area 3*sqrt(3)/2
    real_t expected_area = real_t(3) * std::sqrt(real_t(3)) / real_t(2);
    real_t original_area = tf::area(hexagon.polygons());
    std::cout << "Expected area: " << expected_area << std::endl;
    std::cout << "Original area: " << original_area << std::endl;
    REQUIRE(std::abs(original_area - expected_area) < real_t(1e-5));

    auto tri_mesh = tf::triangulated(hexagon.polygons());

    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    for (std::size_t i = 0; i < tri_mesh.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated points:" << std::endl;
    for (std::size_t i = 0; i < tri_mesh.points().size(); ++i) {
        std::cout << "  " << i << ": (" << tri_mesh.points()[i][0] << ", " << tri_mesh.points()[i][1] << ", " << tri_mesh.points()[i][2] << ")" << std::endl;
    }

    // 6-gon → 4 triangles
    REQUIRE(tri_mesh.faces().size() == 4);

    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;
    REQUIRE(std::abs(tri_area - original_area) < real_t(1e-5));
}

// =============================================================================
// Two Quads - Triangle Count and Area
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_two_quads", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();
    real_t original_area = tf::area(quads.polygons());

    auto tri_mesh = tf::triangulated(quads.polygons());

    // Debug output
    std::cout << "=== triangulated_two_quads ===" << std::endl;
    std::cout << "Original points:" << std::endl;
    for (std::size_t i = 0; i < quads.points().size(); ++i) {
        std::cout << "  " << i << ": (" << quads.points()[i][0] << ", " << quads.points()[i][1] << ", " << quads.points()[i][2] << ")" << std::endl;
    }
    std::cout << "Original faces:" << std::endl;
    for (std::size_t i = 0; i < quads.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << quads.faces()[i][0] << ", " << quads.faces()[i][1] << ", " << quads.faces()[i][2] << ", " << quads.faces()[i][3] << ")" << std::endl;
    }
    std::cout << "Original area: " << original_area << std::endl;
    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    for (std::size_t i = 0; i < tri_mesh.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated points:" << std::endl;
    for (std::size_t i = 0; i < tri_mesh.points().size(); ++i) {
        std::cout << "  " << i << ": (" << tri_mesh.points()[i][0] << ", " << tri_mesh.points()[i][1] << ", " << tri_mesh.points()[i][2] << ")" << std::endl;
    }
    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;

    // 2 quads → 4 triangles
    REQUIRE(tri_mesh.faces().size() == 4);
    REQUIRE(tri_mesh.points().size() == 6);

    // Two unit squares = area 2.0
    REQUIRE(std::abs(original_area - real_t(2)) < real_t(1e-5));

    REQUIRE(std::abs(tri_area - original_area) < real_t(1e-5));
}

// =============================================================================
// Mixed Mesh (Triangle + Quad)
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_mixed_mesh", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mixed = create_mixed_mesh<index_t, real_t>();
    real_t original_area = tf::area(mixed.polygons());

    auto tri_mesh = tf::triangulated(mixed.polygons());

    // Debug output
    std::cout << "=== triangulated_mixed_mesh ===" << std::endl;
    std::cout << "Original points:" << std::endl;
    for (std::size_t i = 0; i < mixed.points().size(); ++i) {
        std::cout << "  " << i << ": (" << mixed.points()[i][0] << ", " << mixed.points()[i][1] << ", " << mixed.points()[i][2] << ")" << std::endl;
    }
    std::cout << "Original area: " << original_area << std::endl;
    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    for (std::size_t i = 0; i < tri_mesh.faces().size(); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }
    std::cout << "Triangulated points:" << std::endl;
    for (std::size_t i = 0; i < tri_mesh.points().size(); ++i) {
        std::cout << "  " << i << ": (" << tri_mesh.points()[i][0] << ", " << tri_mesh.points()[i][1] << ", " << tri_mesh.points()[i][2] << ")" << std::endl;
    }
    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;

    // Triangle (1) + Quad (2) = 3 triangles
    REQUIRE(tri_mesh.faces().size() == 3);

    REQUIRE(std::abs(tri_area - original_area) < real_t(1e-5));
}

// =============================================================================
// All Indices Valid
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_indices_valid", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto hexagon = create_hexagon<index_t, real_t>();
    auto tri_mesh = tf::triangulated(hexagon.polygons());

    for (decltype(tri_mesh.faces().size()) i = 0; i < tri_mesh.faces().size(); ++i) {
        REQUIRE(tri_mesh.faces()[i][0] >= 0);
        REQUIRE(tri_mesh.faces()[i][1] >= 0);
        REQUIRE(tri_mesh.faces()[i][2] >= 0);
        REQUIRE(tri_mesh.faces()[i][0] < static_cast<index_t>(tri_mesh.points().size()));
        REQUIRE(tri_mesh.faces()[i][1] < static_cast<index_t>(tri_mesh.points().size()));
        REQUIRE(tri_mesh.faces()[i][2] < static_cast<index_t>(tri_mesh.points().size()));
    }
}

// =============================================================================
// triangulated_faces - Just Indices
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_faces_only", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();
    auto tri_faces = tf::triangulated_faces(quads.polygons());

    // 2 quads → 4 triangles
    REQUIRE(tri_faces.size() == 4);

    // All indices should be valid
    for (decltype(tri_faces.size()) i = 0; i < tri_faces.size(); ++i) {
        REQUIRE(tri_faces[i][0] >= 0);
        REQUIRE(tri_faces[i][1] >= 0);
        REQUIRE(tri_faces[i][2] >= 0);
        REQUIRE(tri_faces[i][0] < index_t(6));
        REQUIRE(tri_faces[i][1] < index_t(6));
        REQUIRE(tri_faces[i][2] < index_t(6));
    }
}

// =============================================================================
// Large Polygon (1000 vertices circle)
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_large_circle", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    int n = 1000;
    tf::points_buffer<real_t, 3> points;

    real_t pi = real_t(3.14159265358979323846);
    for (int i = 0; i < n; ++i) {
        real_t angle = real_t(2) * pi * real_t(i) / real_t(n);
        points.emplace_back(std::cos(angle), std::sin(angle), real_t(0));
    }

    auto polygon = tf::make_polygon(points);

    // Circle with radius 1 has area pi
    real_t expected_area = pi;
    real_t original_area = tf::area(polygon);

    // Debug output
    std::cout << "=== triangulated_large_circle ===" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "Expected area: " << expected_area << std::endl;
    std::cout << "Original area: " << original_area << std::endl;

    REQUIRE(std::abs(original_area - expected_area) < real_t(0.001));

    auto tri_mesh = tf::triangulated(polygon);

    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    std::cout << "Triangulated points: " << tri_mesh.points().size() << std::endl;
    // Print first few faces
    for (std::size_t i = 0; i < std::min(std::size_t(5), tri_mesh.faces().size()); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }

    // n-gon → n-2 triangles
    REQUIRE(int(tri_mesh.faces().size()) == n - 2);
    REQUIRE(int(tri_mesh.points().size()) == n);

    // Area preserved
    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;
    REQUIRE(std::abs(tri_area - original_area) < real_t(0.001));
}

// =============================================================================
// Large Polygon Clockwise (CW) - Tests auto-detection of winding order
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_large_circle_clockwise", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    int n = 1000;
    tf::points_buffer<real_t, 3> points;

    real_t pi = real_t(3.14159265358979323846);
    // Generate points in CLOCKWISE order (negative angle direction)
    for (int i = 0; i < n; ++i) {
        real_t angle = -real_t(2) * pi * real_t(i) / real_t(n);
        points.emplace_back(std::cos(angle), std::sin(angle), real_t(0));
    }

    auto polygon = tf::make_polygon(points);

    // Circle with radius 1 has area pi
    real_t expected_area = pi;
    real_t original_area = tf::area(polygon);

    // Debug output
    std::cout << "=== triangulated_large_circle_clockwise ===" << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "Expected area: " << expected_area << std::endl;
    std::cout << "Original area: " << original_area << std::endl;

    REQUIRE(std::abs(original_area - expected_area) < real_t(0.001));

    auto tri_mesh = tf::triangulated(polygon);

    std::cout << "Triangulated faces: " << tri_mesh.faces().size() << std::endl;
    std::cout << "Triangulated points: " << tri_mesh.points().size() << std::endl;
    // Print first few faces
    for (std::size_t i = 0; i < std::min(std::size_t(5), tri_mesh.faces().size()); ++i) {
        std::cout << "  face " << i << ": (" << tri_mesh.faces()[i][0] << ", " << tri_mesh.faces()[i][1] << ", " << tri_mesh.faces()[i][2] << ")" << std::endl;
    }

    // n-gon → n-2 triangles
    REQUIRE(int(tri_mesh.faces().size()) == n - 2);
    REQUIRE(int(tri_mesh.points().size()) == n);

    // Area preserved
    real_t tri_area = tf::area(tri_mesh.polygons());
    std::cout << "Triangulated area: " << tri_area << std::endl;
    REQUIRE(std::abs(tri_area - original_area) < real_t(0.001));
}

// =============================================================================
// 2D Polygon Tests
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_quad_2d", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    tf::points_buffer<real_t, 2> points;
    points.emplace_back(real_t(0), real_t(0));
    points.emplace_back(real_t(1), real_t(0));
    points.emplace_back(real_t(1), real_t(1));
    points.emplace_back(real_t(0), real_t(1));

    auto polygon = tf::make_polygon(points);

    auto tri_mesh = tf::triangulated(polygon);

    // 4-gon → 2 triangles
    REQUIRE(tri_mesh.faces().size() == 2);
    REQUIRE(tri_mesh.points().size() == 4);

    // Area preserved (unit square = 1.0)
    real_t tri_area = tf::area(tri_mesh.polygons());
    REQUIRE(std::abs(tri_area - real_t(1)) < real_t(1e-5));
}

TEMPLATE_TEST_CASE("triangulated_large_circle_2d", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    int n = 1000;
    tf::points_buffer<real_t, 2> points;

    real_t pi = real_t(3.14159265358979323846);
    for (int i = 0; i < n; ++i) {
        real_t angle = real_t(2) * pi * real_t(i) / real_t(n);
        points.emplace_back(std::cos(angle), std::sin(angle));
    }

    auto polygon = tf::make_polygon(points);

    real_t expected_area = pi;
    real_t original_area = tf::area(polygon);

    REQUIRE(std::abs(original_area - expected_area) < real_t(0.001));

    auto tri_mesh = tf::triangulated(polygon);

    // n-gon → n-2 triangles
    REQUIRE(int(tri_mesh.faces().size()) == n - 2);
    REQUIRE(int(tri_mesh.points().size()) == n);

    // Area preserved
    real_t tri_area = tf::area(tri_mesh.polygons());
    REQUIRE(std::abs(tri_area - original_area) < real_t(0.001));
}

TEMPLATE_TEST_CASE("triangulated_large_circle_2d_clockwise", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    int n = 1000;
    tf::points_buffer<real_t, 2> points;

    real_t pi = real_t(3.14159265358979323846);
    // Generate points in CLOCKWISE order (negative angle direction)
    for (int i = 0; i < n; ++i) {
        real_t angle = -real_t(2) * pi * real_t(i) / real_t(n);
        points.emplace_back(std::cos(angle), std::sin(angle));
    }

    auto polygon = tf::make_polygon(points);

    real_t expected_area = pi;
    real_t original_area = tf::area(polygon);

    REQUIRE(std::abs(original_area - expected_area) < real_t(0.001));

    auto tri_mesh = tf::triangulated(polygon);

    // n-gon → n-2 triangles
    REQUIRE(int(tri_mesh.faces().size()) == n - 2);
    REQUIRE(int(tri_mesh.points().size()) == n);

    // Area preserved
    real_t tri_area = tf::area(tri_mesh.polygons());
    REQUIRE(std::abs(tri_area - original_area) < real_t(0.001));
}

// =============================================================================
// Triangle Mesh Unchanged
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_triangle_mesh_unchanged", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a simple triangle mesh
    tf::polygons_buffer<index_t, real_t, 3, 3> mesh;

    mesh.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2));
    mesh.faces_buffer().emplace_back(index_t(1), index_t(3), index_t(2));

    auto tri_mesh = tf::triangulated(mesh.polygons());

    // Should still have 2 triangles
    REQUIRE(tri_mesh.faces().size() == 2);
    REQUIRE(tri_mesh.points().size() == 4);
}

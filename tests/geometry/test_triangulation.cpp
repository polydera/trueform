/**
 * @file test_triangulation.cpp
 * @brief Tests for triangulation functions
 *
 * Tests for:
 * - triangulated (polygon mesh, single polygon, refused ids, index width)
 *
 * Key verification: area preservation after triangulation using tf::area
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <type_traits>
#include <vector>

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

/**
 * @brief A simple unit square beside a loop whose two edges cross
 *
 * Face 1 walks (3,0) -> (5,0) -> (3,1) -> (4,1): the second and fourth edges
 * cross at a point the input never named, so the face is resolved — it states
 * the crossing, mints the identity that names it, and holds a product over it.
 * Face 0 is untouched beside it.
 */
template <typename Index, typename Real>
auto create_quad_and_self_crossing() -> tf::polygons_buffer<Index, Real, 3, 4> {
    tf::polygons_buffer<Index, Real, 3, 4> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));

    result.points_buffer().emplace_back(Real(3), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(5), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(3), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(4), Real(1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(4), Index(5), Index(6), Index(7));

    return result;
}

/**
 * @brief A hexagon whose FIRST THREE corners are collinear
 *
 * A polygon's tagged normal is read off its first three corners, so this face
 * was silently absent from the mesh AND from the refusal surface. The tier
 * scans for its supporting triple, so the carrier bounds area and states the
 * face's whole triangulation. Shoelace area: 14.
 */
template <typename Index, typename Real>
auto create_collinear_leading_run() -> tf::polygons_buffer<Index, Real, 3, 6> {
    tf::polygons_buffer<Index, Real, 3, 6> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(2), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(4), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(4), Real(3), Real(0));
    result.points_buffer().emplace_back(Real(2), Real(4), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(3), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2), Index(3),
                                       Index(4), Index(5));

    return result;
}

/// The triangles as a canonical list. The generator's aggregate order is
/// unspecified, so the face order is not a property either call promises.
template <typename Faces>
auto canonical_triangles(const Faces &faces)
    -> std::vector<std::array<std::int64_t, 3>> {
    std::vector<std::array<std::int64_t, 3>> out;
    for (decltype(faces.size()) f = 0; f < faces.size(); ++f)
        out.push_back({std::int64_t(faces[f][0]), std::int64_t(faces[f][1]),
                       std::int64_t(faces[f][2])});
    std::sort(out.begin(), out.end());
    return out;
}

/// How many triangles name only corners below `bound`.
template <typename Faces, typename Index>
auto triangles_within(const Faces &faces, Index bound) -> int {
    int count = 0;
    for (decltype(faces.size()) f = 0; f < faces.size(); ++f)
        if (faces[f][0] < bound && faces[f][1] < bound && faces[f][2] < bound)
            ++count;
    return count;
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
// The requested index width
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_indices_within_the_point_table", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();
    auto tri_mesh = tf::triangulated(quads.polygons());

    // 2 quads → 4 triangles over the input's own six points
    REQUIRE(tri_mesh.faces().size() == 4);
    REQUIRE(tri_mesh.points().size() == 6);

    for (decltype(tri_mesh.faces().size()) i = 0; i < tri_mesh.faces().size(); ++i) {
        REQUIRE(tri_mesh.faces()[i][0] >= 0);
        REQUIRE(tri_mesh.faces()[i][1] >= 0);
        REQUIRE(tri_mesh.faces()[i][2] >= 0);
        REQUIRE(tri_mesh.faces()[i][0] < index_t(6));
        REQUIRE(tri_mesh.faces()[i][1] < index_t(6));
        REQUIRE(tri_mesh.faces()[i][2] < index_t(6));
    }
}

TEMPLATE_TEST_CASE("triangulated_honors_the_requested_index_width", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();

    // A corner is a position in the product's own point table, so the width it
    // is written in is the caller's choice and the corners are the same ones.
    auto own = tf::triangulated(quads.polygons());
    auto wide = tf::triangulated<std::int64_t>(quads.polygons());
    auto narrow = tf::triangulated<std::int32_t>(quads.polygons());

    STATIC_REQUIRE(std::is_same_v<
        std::decay_t<decltype(wide.faces_buffer()[0][0])>, std::int64_t>);
    STATIC_REQUIRE(std::is_same_v<
        std::decay_t<decltype(narrow.faces_buffer()[0][0])>, std::int32_t>);

    REQUIRE(canonical_triangles(wide.faces()) == canonical_triangles(own.faces()));
    REQUIRE(canonical_triangles(narrow.faces()) == canonical_triangles(own.faces()));
    REQUIRE(wide.points().size() == own.points().size());
    REQUIRE(narrow.points().size() == own.points().size());
}

// =============================================================================
// A soup — no index type of its own, and no shared identity until it is cleaned
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_soup_shares_its_vertices", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    // Two triangles sharing the edge (0,0,0)-(1,1,0), stated as six
    // independent corner points: a soup names one identity once per face that
    // carries it.
    tf::points_buffer<real_t, 3> corners;
    corners.emplace_back(real_t(0), real_t(0), real_t(0));
    corners.emplace_back(real_t(1), real_t(0), real_t(0));
    corners.emplace_back(real_t(1), real_t(1), real_t(0));
    corners.emplace_back(real_t(0), real_t(0), real_t(0));
    corners.emplace_back(real_t(1), real_t(1), real_t(0));
    corners.emplace_back(real_t(0), real_t(1), real_t(0));

    auto soup = tf::make_polygons(tf::make_mapped_range(
        tf::make_blocked_range<3>(corners.points()),
        [](auto &&block) { return tf::make_polygon(block); }));

    auto own = tf::triangulated(soup);
    auto wide = tf::triangulated<std::int64_t>(soup);

    // A soup carries no index type, so the default is a fixed one and any
    // other is the caller's naming of the output.
    STATIC_REQUIRE(std::is_same_v<
        std::decay_t<decltype(own.faces_buffer()[0][0])>, int>);
    STATIC_REQUIRE(std::is_same_v<
        std::decay_t<decltype(wide.faces_buffer()[0][0])>, std::int64_t>);

    // THE CLEAN RAN: six stated corners are four points, and the two triangles
    // name the shared pair by the same ids.
    REQUIRE(own.faces().size() == 2);
    REQUIRE(own.points().size() == 4);
    REQUIRE(wide.faces().size() == 2);
    REQUIRE(wide.points().size() == 4);

    std::vector<std::int64_t> first{std::int64_t(own.faces()[0][0]),
                                    std::int64_t(own.faces()[0][1]),
                                    std::int64_t(own.faces()[0][2])};
    std::vector<std::int64_t> second{std::int64_t(own.faces()[1][0]),
                                     std::int64_t(own.faces()[1][1]),
                                     std::int64_t(own.faces()[1][2])};
    std::sort(first.begin(), first.end());
    std::sort(second.begin(), second.end());
    std::vector<std::int64_t> shared;
    std::set_intersection(first.begin(), first.end(), second.begin(),
                          second.end(), std::back_inserter(shared));
    REQUIRE(shared.size() == 2);

    // The two widths name the same corners over the same table.
    REQUIRE(canonical_triangles(wide.faces()) == canonical_triangles(own.faces()));

    // Two half-unit-square triangles.
    REQUIRE(std::abs(tf::area(own.polygons()) - real_t(1)) < real_t(1e-5));
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

// =============================================================================
// Closed Loop - A repeated closing vertex names the same loop
// =============================================================================

namespace {

// L-shaped hexagon: no cyclic symmetry, so a loop read one vertex out of
// phase triangulates into overlapping and inverted triangles.
template <typename Real>
auto l_hexagon_coordinates() -> std::array<std::array<Real, 2>, 6> {
    return {std::array<Real, 2>{Real(0), Real(0)},
            std::array<Real, 2>{Real(4), Real(0)},
            std::array<Real, 2>{Real(4), Real(1)},
            std::array<Real, 2>{Real(2), Real(1)},
            std::array<Real, 2>{Real(2), Real(3)},
            std::array<Real, 2>{Real(0), Real(3)}};
}

template <typename Mesh, typename Real>
auto signed_triangle_area_sum(const Mesh &mesh) -> Real {
    Real total = Real(0);
    for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
        auto face = mesh.faces()[f];
        auto p0 = mesh.points()[std::size_t(face[0])];
        auto p1 = mesh.points()[std::size_t(face[1])];
        auto p2 = mesh.points()[std::size_t(face[2])];
        total += Real(0.5) * ((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                              (p2[0] - p0[0]) * (p1[1] - p0[1]));
    }
    return total;
}

} // namespace

TEMPLATE_TEST_CASE("triangulated_closed_loop_matches_open_loop", "[geometry][triangulation]",
    (float),
    (double))
{
    using real_t = TestType;

    auto coordinates = l_hexagon_coordinates<real_t>();

    tf::points_buffer<real_t, 2> open_points;
    for (const auto &c : coordinates)
        open_points.emplace_back(c[0], c[1]);

    tf::points_buffer<real_t, 2> closed_points;
    for (const auto &c : coordinates)
        closed_points.emplace_back(c[0], c[1]);
    closed_points.emplace_back(coordinates[0][0], coordinates[0][1]);

    auto open_mesh = tf::triangulated(tf::make_polygon(open_points));
    auto closed_mesh = tf::triangulated(tf::make_polygon(closed_points));

    // 6-gon -> 4 triangles either way
    REQUIRE(open_mesh.faces().size() == 4);
    REQUIRE(closed_mesh.faces().size() == 4);

    // The repeated closing vertex names the same loop, so it must name the
    // same triangles.
    for (std::size_t f = 0; f < open_mesh.faces().size(); ++f)
        for (int c = 0; c < 3; ++c)
            REQUIRE(open_mesh.faces()[f][std::size_t(c)] ==
                    closed_mesh.faces()[f][std::size_t(c)]);

    // Every triangle carries the loop's winding, and they tile it exactly.
    for (std::size_t f = 0; f < closed_mesh.faces().size(); ++f) {
        auto face = closed_mesh.faces()[f];
        auto p0 = closed_mesh.points()[std::size_t(face[0])];
        auto p1 = closed_mesh.points()[std::size_t(face[1])];
        auto p2 = closed_mesh.points()[std::size_t(face[2])];
        REQUIRE(((p1[0] - p0[0]) * (p2[1] - p0[1]) -
                 (p2[0] - p0[0]) * (p1[1] - p0[1])) > real_t(0));
    }
    REQUIRE(std::abs(signed_triangle_area_sum<decltype(closed_mesh), real_t>(
                         closed_mesh) -
                     real_t(8)) < real_t(1e-4));
}

// =============================================================================
// return_refused - the surface names whose emptiness it was
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_refused_empty_on_clean_input", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_mixed_mesh<index_t, real_t>();

    auto plain = tf::triangulated(mesh.polygons());
    auto [tagged, refused] = tf::triangulated(mesh.polygons(), tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(tagged.faces().size() == plain.faces().size());
    REQUIRE(canonical_triangles(tagged.faces()) ==
            canonical_triangles(plain.faces()));

    REQUIRE(tagged.points().size() == plain.points().size());
    for (std::size_t i = 0; i < std::size_t(plain.points().size()); ++i)
        for (std::size_t d = 0; d < 3; ++d)
            REQUIRE(tagged.points()[i][d] == plain.points()[i][d]);
}

TEMPLATE_TEST_CASE("triangulated_resolves_the_self_crossing_face", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_quad_and_self_crossing<index_t, real_t>();

    auto plain = tf::triangulated(mesh.polygons());
    auto [tagged, refused] = tf::triangulated(mesh.polygons(), tf::return_refused);

    // A crossing loop is resolved, not dropped: nobody refuses.
    REQUIRE(refused.size() == 0);

    // The crossing stands on a point the input never named, so the face mints
    // the identity that names it and holds a product over it.
    REQUIRE(tagged.points().size() > mesh.points().size());
    REQUIRE(tagged.faces().size() > 2);

    // The tagged call emits exactly what the untagged one does.
    REQUIRE(canonical_triangles(tagged.faces()) ==
            canonical_triangles(plain.faces()));

    // The simple quad beside it is untouched: its own two triangles over its
    // own four corners, and nothing else names them alone.
    REQUIRE(triangles_within(tagged.faces(), index_t(4)) == 2);
}

TEMPLATE_TEST_CASE("triangulated_resolves_2d", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 2, 4> mesh;
    mesh.points_buffer().emplace_back(real_t(0), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(0));
    mesh.points_buffer().emplace_back(real_t(1), real_t(1));
    mesh.points_buffer().emplace_back(real_t(0), real_t(1));
    mesh.points_buffer().emplace_back(real_t(3), real_t(0));
    mesh.points_buffer().emplace_back(real_t(5), real_t(0));
    mesh.points_buffer().emplace_back(real_t(3), real_t(1));
    mesh.points_buffer().emplace_back(real_t(4), real_t(1));
    mesh.faces_buffer().emplace_back(index_t(0), index_t(1), index_t(2), index_t(3));
    mesh.faces_buffer().emplace_back(index_t(4), index_t(5), index_t(6), index_t(7));

    auto plain = tf::triangulated(mesh.polygons());
    auto [tagged, refused] = tf::triangulated(mesh.polygons(), tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(tagged.faces().size() > 2);
    REQUIRE(tagged.points().size() > mesh.points().size());
    REQUIRE(triangles_within(tagged.faces(), index_t(4)) == 2);
    REQUIRE(canonical_triangles(tagged.faces()) ==
            canonical_triangles(plain.faces()));
}

// =============================================================================
// A collinear leading run — the face the float projector could not see
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_collinear_leading_run", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = create_collinear_leading_run<index_t, real_t>();

    auto plain = tf::triangulated(mesh.polygons());
    auto [tagged, refused] = tf::triangulated(mesh.polygons(), tf::return_refused);

    // The face is neither absent nor refused: the frame is the one the
    // carrier's own supporting triple gives it, so the carrier bounds area.
    REQUIRE(refused.size() == 0);
    REQUIRE(plain.faces().size() == 4);
    REQUIRE(canonical_triangles(tagged.faces()) ==
            canonical_triangles(plain.faces()));

    // It needs no resolution, so it names the input's own points and nothing
    // else, and its triangles tile exactly the area it bounds.
    REQUIRE(plain.points().size() == mesh.points().size());
    REQUIRE(triangles_within(plain.faces(), index_t(6)) == 4);
    REQUIRE(std::abs(tf::area(plain.polygons()) - real_t(14)) < real_t(1e-4));
}

TEMPLATE_TEST_CASE("triangulated_refused_empty_on_two_quads", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();

    auto plain = tf::triangulated(quads.polygons());
    auto [tagged, refused] = tf::triangulated(quads.polygons(), tf::return_refused);

    REQUIRE(refused.size() == 0);
    REQUIRE(tagged.faces().size() == 4);
    REQUIRE(tagged.points().size() == quads.points().size());
    REQUIRE(canonical_triangles(tagged.faces()) ==
            canonical_triangles(plain.faces()));
}

// =============================================================================
// Shared Edges - Neighbouring faces triangulate watertight
// =============================================================================

TEMPLATE_TEST_CASE("triangulated_shared_edge_watertight", "[geometry][triangulation]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto quads = create_two_quads<index_t, real_t>();
    auto tri_mesh = tf::triangulated(quads.polygons());

    // The triangulation names no point the input did not carry.
    REQUIRE(tri_mesh.points().size() == quads.points().size());

    // The edge the two quads share is used once from each side.
    int shared_uses = 0;
    for (std::size_t f = 0; f < tri_mesh.faces().size(); ++f) {
        auto face = tri_mesh.faces()[f];
        for (int e = 0; e < 3; ++e) {
            auto u = face[std::size_t(e)];
            auto v = face[std::size_t((e + 1) % 3)];
            if ((u == index_t(1) && v == index_t(2)) ||
                (u == index_t(2) && v == index_t(1)))
                ++shared_uses;
        }
    }
    REQUIRE(shared_uses == 2);
}

/**
 * @file test_duplicate_faces.cpp
 * @brief Tests for duplicate face detection via compute_unique_faces_mask
 *
 * Tests for:
 * - No duplicates case
 * - Identical duplicate faces
 * - Reversed winding duplicates
 * - Rotated index duplicates
 * - Multiple duplicates
 * - Dynamic polygon sizes (quads, pentagons)
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"

// =============================================================================
// Helper functions
// =============================================================================

template <typename Index, typename Real, std::size_t Ngon>
auto make_test_buffer(std::initializer_list<tf::point<Real, 3>> pts,
                      std::initializer_list<std::array<Index, Ngon>> fcs) {
    tf::polygons_buffer<Index, Real, 3, Ngon> buffer;
    for (auto& p : pts)
        buffer.points_buffer().push_back(p);
    for (auto& f : fcs)
        buffer.faces_buffer().push_back(f);
    return buffer;
}

template <typename Index, typename Real>
auto make_test_buffer_dynamic(
    std::initializer_list<tf::point<Real, 3>> pts,
    std::initializer_list<std::initializer_list<Index>> fcs) {
    tf::polygons_buffer<Index, Real, 3, tf::dynamic_size> buffer;
    for (auto& p : pts)
        buffer.points_buffer().push_back(p);
    for (auto& f : fcs)
        buffer.faces_buffer().push_back(f);
    return buffer;
}

// =============================================================================
// Test 1: Triangles - No Duplicates
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triangles_no_duplicates", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)},
         {real_t(0.5), real_t(0.5), real_t(1)}},
        {{{index_t(0), index_t(1), index_t(2)}},
         {{index_t(0), index_t(1), index_t(3)}},
         {{index_t(1), index_t(2), index_t(3)}},
         {{index_t(0), index_t(2), index_t(3)}}});

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // All faces should be unique
    for (std::size_t i = 0; i < mask.size(); ++i) {
        REQUIRE(mask[i]);
    }
}

// =============================================================================
// Test 2: Triangles - Identical Duplicate
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triangles_identical", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)}},
        {{{index_t(0), index_t(1), index_t(2)}},
         {{index_t(0), index_t(1), index_t(2)}}});

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique (has smallest ID)
    REQUIRE(mask[0]);
    // Face 1 should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 3: Triangles - Reversed Winding Duplicate
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triangles_reversed", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)}},
        {{{index_t(0), index_t(1), index_t(2)}},
         {{index_t(0), index_t(2), index_t(1)}}});  // Reversed winding

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique
    REQUIRE(mask[0]);
    // Face 1 (reversed) should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 4: Triangles - Rotated Index Duplicate
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triangles_rotated", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)}},
        {{{index_t(0), index_t(1), index_t(2)}},
         {{index_t(1), index_t(2), index_t(0)}}});  // Rotated indices

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique
    REQUIRE(mask[0]);
    // Face 1 (rotated) should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 5: Triangles - Multiple Duplicates
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triangles_multiple", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)},
         {real_t(2), real_t(0), real_t(0)},
         {real_t(1.5), real_t(1), real_t(0)}},
        {{{index_t(0), index_t(1), index_t(2)}},   // Face 0 - original
         {{index_t(1), index_t(3), index_t(4)}},   // Face 1 - original
         {{index_t(0), index_t(1), index_t(2)}},   // Face 2 - duplicate of 0
         {{index_t(3), index_t(4), index_t(1)}}});  // Face 3 - rotated duplicate of 1

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Faces 0 and 1 should be unique
    REQUIRE(mask[0]);
    REQUIRE(mask[1]);
    // Faces 2 and 3 should be duplicates
    REQUIRE_FALSE(mask[2]);
    REQUIRE_FALSE(mask[3]);
}

// =============================================================================
// Test 6: Dynamic Quad - Rotated
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_quad_rotated", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer_dynamic<index_t, real_t>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(1), real_t(1), real_t(0)},
         {real_t(0), real_t(1), real_t(0)}},
        {{index_t(0), index_t(1), index_t(2), index_t(3)},
         {index_t(2), index_t(3), index_t(0), index_t(1)}});  // Rotated by 2

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique
    REQUIRE(mask[0]);
    // Face 1 (rotated) should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 7: Dynamic Quad - Reversed
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_quad_reversed", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer_dynamic<index_t, real_t>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(1), real_t(1), real_t(0)},
         {real_t(0), real_t(1), real_t(0)}},
        {{index_t(0), index_t(1), index_t(2), index_t(3)},
         {index_t(0), index_t(3), index_t(2), index_t(1)}});  // Reversed winding

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique
    REQUIRE(mask[0]);
    // Face 1 (reversed) should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 8: Dynamic Pentagon - Rotated
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_pentagon_rotated", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer_dynamic<index_t, real_t>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(1.3), real_t(0.8), real_t(0)},
         {real_t(0.5), real_t(1.2), real_t(0)},
         {real_t(-0.3), real_t(0.8), real_t(0)}},
        {{index_t(0), index_t(1), index_t(2), index_t(3), index_t(4)},
         {index_t(3), index_t(4), index_t(0), index_t(1), index_t(2)}});  // Rotated by 3

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Face 0 should be unique
    REQUIRE(mask[0]);
    // Face 1 (rotated pentagon) should be marked as duplicate
    REQUIRE_FALSE(mask[1]);
}

// =============================================================================
// Test 9: Box Mesh - No Duplicates
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_box_no_duplicates", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto box = tf::make_box_mesh<index_t>(real_t(1), real_t(1), real_t(1));
    auto polygons = box.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(box.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // All box faces should be unique
    std::size_t unique_count = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) ++unique_count;
    }
    REQUIRE(unique_count == box.size());
}

// =============================================================================
// Test 10: Sphere Mesh - No Duplicates
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_sphere_no_duplicates", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto sphere = tf::make_sphere_mesh<index_t>(real_t(1), 20, 20);
    auto polygons = sphere.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(sphere.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // All sphere faces should be unique
    std::size_t unique_count = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) ++unique_count;
    }
    REQUIRE(unique_count == sphere.size());
}

// =============================================================================
// Test 11: Triple Duplicate
// =============================================================================

TEMPLATE_TEST_CASE("duplicate_faces_triple", "[topology][duplicate_faces]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto buffer = make_test_buffer<index_t, real_t, 3>(
        {{real_t(0), real_t(0), real_t(0)},
         {real_t(1), real_t(0), real_t(0)},
         {real_t(0.5), real_t(1), real_t(0)}},
        {{{index_t(0), index_t(1), index_t(2)}},
         {{index_t(1), index_t(2), index_t(0)}},   // Rotated
         {{index_t(2), index_t(0), index_t(1)}}});  // Also rotated

    auto polygons = buffer.polygons();
    tf::face_membership<index_t> fmem(polygons);

    tf::buffer<bool> mask;
    mask.allocate(buffer.size());
    tf::compute_unique_faces_mask(polygons.faces(), fmem, mask);

    // Exactly 1 face should be unique
    std::size_t unique_count = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) ++unique_count;
    }
    REQUIRE(unique_count == 1);
    // The first face should be the unique one
    REQUIRE(mask[0]);
}

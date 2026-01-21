/**
 * @file test_clean_polygons.cpp
 * @brief Tests for tf::cleaned(polygons, ...)
 *
 * Tests for:
 * - clean_polygons_no_duplicates
 * - clean_polygons_duplicate_vertices
 * - clean_polygons_tolerance
 * - clean_polygons_degenerate_face
 * - clean_polygons_unreferenced_points
 * - clean_polygons_with_index_map
 * - clean_polygons_box_mesh
 * - clean_polygons_sphere_mesh
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "canonicalize_mesh.hpp"
#include "mesh_generators.hpp"

// =============================================================================
// clean_polygons_no_duplicates
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_no_duplicates", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a clean mesh with no duplicates
    auto input = tf::test::create_triangle_polygons_3d<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    // Mesh should remain unchanged
    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());

    // Verify canonical comparison
    auto canonical_result = tf::test::canonicalize_mesh(result);
    auto canonical_expected = tf::test::canonicalize_mesh(input);
    REQUIRE(tf::test::meshes_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_polygons_duplicate_vertices
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_duplicate_vertices", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with duplicate point locations referenced by different faces
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    // Points with duplicates (point 3 is same as point 0)
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0
    input.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    // Two faces using duplicate vertices
    input.faces_buffer().emplace_back(0, 1, 2);  // uses point 0
    input.faces_buffer().emplace_back(3, 1, 4);  // uses point 3 (duplicate of 0)

    auto result = tf::cleaned(input.polygons());

    // Should merge duplicate points (0 and 3 -> single point)
    // Result: 4 unique points
    REQUIRE(result.points().size() == 4);
    REQUIRE(result.faces().size() == 2);
}

// =============================================================================
// clean_polygons_tolerance
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_tolerance", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with points within tolerance
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0.001), real_t(0), real_t(0));  // near point 0

    // Two faces
    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(3, 1, 2);  // uses point 3 (near 0)

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // Points 0 and 3 should merge
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_polygons_degenerate_face
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_degenerate_face", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with degenerate face (less than 3 unique vertices after cleaning)
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));  // duplicate of 1

    // Valid face
    input.faces_buffer().emplace_back(0, 1, 2);
    // Degenerate face (uses only 2 unique points after merging)
    input.faces_buffer().emplace_back(0, 3, 4);  // becomes (0, 0, 1) after merging

    auto result = tf::cleaned(input.polygons());

    // Degenerate face should be removed
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// clean_polygons_unreferenced_points
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_unreferenced_points", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with unreferenced points
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(99), real_t(99), real_t(99));  // unreferenced

    input.faces_buffer().emplace_back(0, 1, 2);

    auto result = tf::cleaned(input.polygons());

    // Unreferenced point should be removed
    REQUIRE(result.points().size() == 3);
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// clean_polygons_with_index_map
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_with_index_map", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with duplicates
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0

    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(3, 1, 2);  // uses duplicate

    auto [result, face_im, point_im] = tf::cleaned(input.polygons(), tf::return_index_map);

    // Duplicate points should merge
    REQUIRE(result.points().size() == 3);

    // Point index map should show duplicate mapping
    REQUIRE(point_im.f().size() == 4);
    REQUIRE(point_im.f()[0] == point_im.f()[3]);  // both map to same output
}

// =============================================================================
// clean_polygons_box_mesh
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_box_mesh", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a clean box mesh (no duplicates)
    auto input = tf::test::create_cube_polygons<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    // Box should be unchanged
    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());

    // Verify canonical round-trip
    auto canonical_result = tf::test::canonicalize_mesh(result);
    auto canonical_expected = tf::test::canonicalize_mesh(input);
    REQUIRE(tf::test::meshes_equal(canonical_result, canonical_expected));
}

// =============================================================================
// clean_polygons_empty
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_empty", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

// =============================================================================
// clean_polygons_single_face
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_single_face", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    input.faces_buffer().emplace_back(0, 1, 2);

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);
}

// =============================================================================
// clean_polygons_2d
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_2d", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Test with 2D mesh
    auto input = tf::test::create_triangle_polygons_2d<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());
}

// =============================================================================
// clean_polygons_dynamic_mesh
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_dynamic_mesh", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Test with dynamic mesh (mixed n-gons)
    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    // Clean mesh should remain unchanged
    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());
}

// =============================================================================
// clean_polygons_duplicate_faces
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_duplicate_faces", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create mesh with duplicate faces
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(0, 1, 2);  // exact duplicate
    input.faces_buffer().emplace_back(1, 2, 0);  // cyclic permutation (same face)

    auto result = tf::cleaned(input.polygons());

    // Duplicate faces should be removed
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// clean_polygons_all_degenerate
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_all_degenerate", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // All faces are degenerate
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0

    // Degenerate faces
    input.faces_buffer().emplace_back(0, 0, 1);  // 2 unique vertices
    input.faces_buffer().emplace_back(0, 2, 1);  // becomes (0, 0, 1) after merging

    auto result = tf::cleaned(input.polygons());

    // All degenerate faces should be removed
    REQUIRE(result.faces().size() == 0);
}

// =============================================================================
// clean_polygons_tetrahedron
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_tetrahedron", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Tetrahedron with no duplicates
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(0.5), real_t(1));

    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(0, 1, 3);
    input.faces_buffer().emplace_back(1, 2, 3);
    input.faces_buffer().emplace_back(2, 0, 3);

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 4);
    REQUIRE(result.points().size() == 4);
}

// =============================================================================
// clean_polygons_cube_with_duplicate_vertices
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_cube_with_duplicate_vertices", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a mesh with explicit duplicate vertices that share coordinates
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    // 4 original vertices forming a tetrahedron base + apex
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    // Duplicate of point 0
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    // Duplicate of point 1
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));

    // Two faces using different vertex indices but same positions
    input.faces_buffer().emplace_back(0, 1, 2);  // uses original vertices
    input.faces_buffer().emplace_back(3, 4, 2);  // uses duplicate vertices for 0, 1

    // Use tolerance to merge duplicate points
    const real_t tolerance = real_t(1e-6);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // Duplicate vertices merge: 5 -> 3 unique points
    // After merging, both faces become (0, 1, 2), so one is removed
    REQUIRE(result.points().size() == 3);
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// clean_polygons_manifold_with_holes
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_manifold_with_holes", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Grid of triangles with some duplicates
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    // 3x3 grid of vertices
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            input.points_buffer().emplace_back(real_t(x), real_t(y), real_t(0));
        }
    }

    // Create triangles for a 2x2 quad grid (8 triangles)
    auto idx = [](int x, int y) { return index_t(y * 3 + x); };

    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            input.faces_buffer().emplace_back(idx(x, y), idx(x+1, y), idx(x+1, y+1));
            input.faces_buffer().emplace_back(idx(x, y), idx(x+1, y+1), idx(x, y+1));
        }
    }

    // Add duplicate triangles
    input.faces_buffer().emplace_back(idx(0, 0), idx(1, 0), idx(1, 1));  // duplicate of first
    input.faces_buffer().emplace_back(idx(1, 1), idx(1, 0), idx(0, 0));  // rotated duplicate

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 8);
    REQUIRE(result.points().size() == 9);
}

// =============================================================================
// clean_polygons_tolerance_vertex_merge
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_tolerance_vertex_merge", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Two triangles with nearly coincident vertices that should merge
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    // First triangle
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    // Second triangle with vertices slightly offset
    input.points_buffer().emplace_back(real_t(1.001), real_t(0), real_t(0));     // near point 1
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(3, 4, 5);

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // Points 1 and 3 should merge
    REQUIRE(result.points().size() == 5);
    REQUIRE(result.faces().size() == 2);
}

// =============================================================================
// clean_polygons_quad_mesh
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_quad_mesh", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Quad mesh (4 vertices per face)
    tf::polygons_buffer<index_t, real_t, 3, 4> input;

    // 2x2 grid of vertices
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(1), real_t(0));

    // Two quads
    input.faces_buffer().emplace_back(0, 1, 4, 3);
    input.faces_buffer().emplace_back(1, 2, 5, 4);

    // Add duplicate quad
    input.faces_buffer().emplace_back(0, 1, 4, 3);  // exact duplicate
    input.faces_buffer().emplace_back(4, 3, 0, 1);  // rotated duplicate

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 2);
    REQUIRE(result.points().size() == 6);
}

// =============================================================================
// clean_polygons_many_duplicates_same_face
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_many_duplicates_same_face", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));

    // Original face
    input.faces_buffer().emplace_back(0, 1, 2);

    // Many duplicates (all rotations)
    input.faces_buffer().emplace_back(1, 2, 0);
    input.faces_buffer().emplace_back(2, 0, 1);
    input.faces_buffer().emplace_back(0, 1, 2);
    input.faces_buffer().emplace_back(0, 1, 2);

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// clean_polygons_index_map_face_tracking
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_index_map_face_tracking", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    input.faces_buffer().emplace_back(0, 1, 2);  // unique face 0
    input.faces_buffer().emplace_back(1, 2, 3);  // unique face 1
    input.faces_buffer().emplace_back(1, 2, 0);  // rotated duplicate of face 0
    input.faces_buffer().emplace_back(2, 0, 1);  // rotated duplicate of face 0

    auto [result, face_im, point_im] = tf::cleaned(input.polygons(), tf::return_index_map);

    // Should have 2 unique faces (face 0 and face 1)
    // Faces 2 and 3 are rotated duplicates of face 0, so they are removed
    REQUIRE(result.faces().size() == 2);
    REQUIRE(face_im.f().size() == 4);
    REQUIRE(face_im.kept_ids().size() == 2);

    // Kept faces map to valid output indices
    REQUIRE(static_cast<std::size_t>(face_im.f()[0]) < result.faces().size());
    REQUIRE(static_cast<std::size_t>(face_im.f()[1]) < result.faces().size());

    // Removed faces (rotated duplicates) map to sentinel value
    auto sentinel = static_cast<index_t>(face_im.f().size());
    REQUIRE(face_im.f()[2] == sentinel);
    REQUIRE(face_im.f()[3] == sentinel);
}

// =============================================================================
// clean_polygons_large_mesh
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_large_mesh", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Create a larger mesh grid
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    constexpr int grid_size = 10;

    // Grid of vertices
    for (int y = 0; y <= grid_size; ++y) {
        for (int x = 0; x <= grid_size; ++x) {
            input.points_buffer().emplace_back(real_t(x), real_t(y), real_t(0));
        }
    }

    // Triangulate the grid
    auto idx = [](int x, int y) -> index_t {
        constexpr int grid_size = 10;
        return index_t(y * (grid_size + 1) + x);
    };

    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            input.faces_buffer().emplace_back(idx(x, y), idx(x+1, y), idx(x+1, y+1));
            input.faces_buffer().emplace_back(idx(x, y), idx(x+1, y+1), idx(x, y+1));
        }
    }

    auto result = tf::cleaned(input.polygons());

    // Should have all unique triangles (no duplicates in clean grid)
    REQUIRE(result.faces().size() == 2 * grid_size * grid_size);
    REQUIRE(result.points().size() == (grid_size + 1) * (grid_size + 1));
}

// =============================================================================
// clean_polygons_tolerance_creates_degenerate
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_tolerance_creates_degenerate", "[clean][clean_polygons]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Triangle that becomes degenerate after tolerance merging
    tf::polygons_buffer<index_t, real_t, 3, 3> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.001), real_t(0), real_t(0));  // near 0
    input.points_buffer().emplace_back(real_t(0.002), real_t(0), real_t(0));  // near 0 and 1

    // Valid triangle
    input.points_buffer().emplace_back(real_t(10), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(11), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(10.5), real_t(1), real_t(0));

    input.faces_buffer().emplace_back(0, 1, 2);  // will become degenerate
    input.faces_buffer().emplace_back(3, 4, 5);  // will remain valid

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // First face becomes degenerate, only second remains
    REQUIRE(result.faces().size() == 1);
}

// =============================================================================
// Dynamic size polygon tests (tf::dynamic_size)
// =============================================================================

TEMPLATE_TEST_CASE("clean_polygons_dynamic_no_duplicates", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Clean dynamic mesh should remain unchanged
    auto input = tf::test::create_dynamic_polygons_3d<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_mixed_ngons", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Mixed mesh with triangles and quads
    auto input = tf::test::create_mixed_polygons_3d<index_t, real_t>();

    auto result = tf::cleaned(input.polygons());

    // Clean mixed mesh should remain unchanged
    REQUIRE(result.faces().size() == input.faces().size());
    REQUIRE(result.points().size() == input.points().size());
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_duplicate_vertices", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    // Points with duplicates
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    // Triangle and quad using some duplicate vertices
    input.faces_buffer().push_back({0, 1, 2});       // triangle
    input.faces_buffer().push_back({3, 1, 4, 2});    // quad using duplicate vertex 3

    const real_t tolerance = real_t(1e-6);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // Point 3 merges with point 0 -> 4 unique points
    REQUIRE(result.points().size() == 4);
    REQUIRE(result.faces().size() == 2);
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_duplicate_faces", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(1), real_t(0));

    // Quad and its duplicate (rotated)
    input.faces_buffer().push_back({0, 1, 2, 3});    // quad
    input.faces_buffer().push_back({1, 2, 3, 0});    // rotated duplicate

    auto result = tf::cleaned(input.polygons());

    // Duplicate quad should be removed
    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 4);
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_degenerate_face", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));

    // Valid triangle
    input.faces_buffer().push_back({0, 1, 2});
    // Degenerate quad (only 2 unique vertices: 0 and 1)
    input.faces_buffer().push_back({0, 0, 1, 1});

    auto result = tf::cleaned(input.polygons());

    // Degenerate face should be removed (only 2 unique vertices)
    REQUIRE(result.faces().size() == 1);
    REQUIRE(result.points().size() == 3);
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_unreferenced_points", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(99), real_t(99), real_t(99));  // unreferenced

    input.faces_buffer().push_back({0, 1, 2});

    auto result = tf::cleaned(input.polygons());

    // Unreferenced point should be removed
    REQUIRE(result.points().size() == 3);
    REQUIRE(result.faces().size() == 1);
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_with_index_map", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));  // duplicate of 0

    input.faces_buffer().push_back({0, 1, 2});
    input.faces_buffer().push_back({3, 1, 2});  // uses duplicate vertex

    const real_t tolerance = real_t(1e-6);
    auto [result, face_im, point_im] = tf::cleaned(input.polygons(), tolerance, tf::return_index_map);

    // Points 0 and 3 merge -> 3 unique points
    REQUIRE(result.points().size() == 3);
    // Faces become duplicates -> 1 face
    REQUIRE(result.faces().size() == 1);

    // Verify index maps
    REQUIRE(point_im.f().size() == 4);
    REQUIRE(point_im.f()[0] == point_im.f()[3]);  // duplicates map to same
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_mixed_with_tolerance", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    // Points with some within tolerance
    input.points_buffer().emplace_back(real_t(0), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(0.5), real_t(1), real_t(0));
    input.points_buffer().emplace_back(real_t(0.001), real_t(0), real_t(0));  // near 0
    input.points_buffer().emplace_back(real_t(2), real_t(0), real_t(0));
    input.points_buffer().emplace_back(real_t(1.5), real_t(1), real_t(0));

    // Triangle using original vertices
    input.faces_buffer().push_back({0, 1, 2});
    // Pentagon using some near-duplicate vertices
    input.faces_buffer().push_back({3, 1, 4, 5, 2});

    const real_t tolerance = real_t(0.01);
    auto result = tf::cleaned(input.polygons(), tolerance);

    // Point 3 merges with 0 -> 5 unique points
    REQUIRE(result.points().size() == 5);
    REQUIRE(result.faces().size() == 2);
}

TEMPLATE_TEST_CASE("clean_polygons_dynamic_empty", "[clean][clean_polygons][dynamic]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::polygons_buffer<index_t, real_t, 3, tf::dynamic_size> input;

    auto result = tf::cleaned(input.polygons());

    REQUIRE(result.faces().size() == 0);
    REQUIRE(result.points().size() == 0);
}

/**
 * @file test_boundary_rims.cpp
 * @brief Tests for boundary rim assembly
 *
 * Tests for:
 * - make_boundary_rims
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "type_traits.hpp"
#include "topology_generators.hpp"
#include <cstddef>
#include <set>
#include <type_traits>
#include <vector>

namespace {

/**
 * @brief Whether a face winds the directed pair (a, b) among its corners.
 */
template <typename Face, typename Index>
auto rim_face_winds_edge(const Face &face, Index a, Index b) -> bool {
    auto size = static_cast<Index>(face.size());
    Index prev = size - 1;
    for (Index i = 0; i < size; prev = i++) {
        if (static_cast<Index>(face[prev]) == a &&
            static_cast<Index>(face[i]) == b)
            return true;
    }
    return false;
}

/**
 * @brief How many times the mesh's faces hold the undirected edge {a, b}.
 */
template <typename Faces, typename Index>
auto rim_faces_on_edge(const Faces &faces, Index a, Index b) -> std::size_t {
    std::size_t count = 0;
    for (std::size_t f = 0; f < faces.size(); ++f) {
        const auto &face = faces[f];
        auto size = static_cast<Index>(face.size());
        Index prev = size - 1;
        for (Index i = 0; i < size; prev = i++) {
            auto x = static_cast<Index>(face[prev]);
            auto y = static_cast<Index>(face[i]);
            if ((x == a && y == b) || (x == b && y == a))
                ++count;
        }
    }
    return count;
}

/**
 * @brief Every rim edge is wound by the face the rim names, and that face
 *        is the only one holding it.
 */
template <typename Mesh, typename Rims>
auto rim_edges_match_their_faces(const Mesh &mesh, const Rims &rims) -> bool {
    auto mesh_faces = mesh.faces();
    for (std::size_t i = 0; i < rims.size(); ++i) {
        auto vertices = rims.vertices[i];
        auto rim_faces = rims.faces[i];
        auto n = vertices.size();
        auto n_edges = rims.closed[i] ? n : n - 1;
        if (rim_faces.size() != n_edges)
            return false;
        for (std::size_t k = 0; k < n_edges; ++k) {
            auto a = vertices[k];
            auto b = vertices[(k + 1) % n];
            if (!rim_face_winds_edge(
                    mesh_faces[static_cast<std::size_t>(rim_faces[k])], a, b))
                return false;
            if (rim_faces_on_edge(mesh_faces, a, b) != 1)
                return false;
        }
    }
    return true;
}

/**
 * @brief A rim's vertex ids as a vector.
 */
template <typename Block> auto rim_block_to_vector(const Block &block) {
    using Index = std::decay_t<decltype(block[0])>;
    std::vector<Index> result;
    for (std::size_t i = 0; i < block.size(); ++i)
        result.push_back(block[i]);
    return result;
}

/**
 * @brief One triangle.
 */
template <typename Index, typename Real>
auto rim_single_triangle_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));

    return result;
}

/**
 * @brief Two triangles meeting at vertex 0 alone: the boundary passes
 *        through that vertex twice.
 */
template <typename Index, typename Real>
auto rim_bowtie_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(3), Index(4));

    return result;
}

/**
 * @brief A three-sided tube with no caps: two rings of three vertices.
 */
template <typename Index, typename Real>
auto rim_open_tube_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(1));
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(1));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(4));
    result.faces_buffer().emplace_back(Index(0), Index(4), Index(3));
    result.faces_buffer().emplace_back(Index(1), Index(2), Index(5));
    result.faces_buffer().emplace_back(Index(1), Index(5), Index(4));
    result.faces_buffer().emplace_back(Index(2), Index(0), Index(3));
    result.faces_buffer().emplace_back(Index(2), Index(3), Index(5));

    return result;
}

/**
 * @brief Three triangles on the shared edge (0, 1): its endpoints carry
 *        three boundary edges each, so no rim closes.
 */
template <typename Index, typename Real>
auto rim_three_on_one_edge_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(0), Real(1));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(4));

    return result;
}

} // anonymous namespace

// =============================================================================
// make_boundary_rims - One Triangle
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_single_triangle", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = rim_single_triangle_3d<index_t, real_t>();
    auto rims = tf::make_boundary_rims(mesh.polygons());

    REQUIRE(rims.size() == 1);
    REQUIRE(rims.closed[0]);

    // The boundary is stated face by face and corner by corner, so the
    // first boundary edge is (face[2], face[0]) and the rim starts there.
    REQUIRE(rim_block_to_vector(rims.vertices[0]) ==
            std::vector<index_t>{2, 0, 1});
    REQUIRE(rim_block_to_vector(rims.faces[0]) ==
            std::vector<index_t>{0, 0, 0});
    REQUIRE(rim_edges_match_their_faces(mesh, rims));
}

// =============================================================================
// make_boundary_rims - Plane Patch (one closed rim)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_plane_patch", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(3, 3);
    auto rims = tf::make_boundary_rims(mesh.polygons());

    // A disk's boundary is one polyline that closes.
    REQUIRE(rims.size() == 1);
    REQUIRE(rims.closed[0]);
    REQUIRE(rims.vertices[0].size() == 8);
    REQUIRE(rims.faces[0].size() == 8);

    auto walked = rim_block_to_vector(rims.vertices[0]);
    REQUIRE(std::set<index_t>(walked.begin(), walked.end()) ==
            std::set<index_t>{0, 1, 2, 3, 5, 6, 7, 8});
    REQUIRE(rim_edges_match_their_faces(mesh, rims));
}

// =============================================================================
// make_boundary_rims - Open Tube (two closed rims)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_open_tube", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = rim_open_tube_3d<index_t, real_t>();
    auto rims = tf::make_boundary_rims(mesh.polygons());

    REQUIRE(rims.size() == 2);
    REQUIRE(rims.closed[0]);
    REQUIRE(rims.closed[1]);
    REQUIRE(rims.vertices[0].size() == 3);
    REQUIRE(rims.vertices[1].size() == 3);
    REQUIRE(rims.faces[0].size() == 3);
    REQUIRE(rims.faces[1].size() == 3);

    auto first = rim_block_to_vector(rims.vertices[0]);
    auto second = rim_block_to_vector(rims.vertices[1]);
    REQUIRE(std::set<index_t>(first.begin(), first.end()) ==
            std::set<index_t>{0, 1, 2});
    REQUIRE(std::set<index_t>(second.begin(), second.end()) ==
            std::set<index_t>{3, 4, 5});
    REQUIRE(rim_edges_match_their_faces(mesh, rims));
}

// =============================================================================
// make_boundary_rims - Pinched Rim (splits at the pinch)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_pinched_rim", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = rim_bowtie_3d<index_t, real_t>();
    auto rims = tf::make_boundary_rims(mesh.polygons());

    // The boundary reaches vertex 0 four times, so it is two rims there
    // and not one figure eight.
    REQUIRE(rims.size() == 2);
    REQUIRE(rims.closed[0]);
    REQUIRE(rims.closed[1]);

    REQUIRE(rim_block_to_vector(rims.vertices[0]) ==
            std::vector<index_t>{0, 1, 2});
    REQUIRE(rim_block_to_vector(rims.faces[0]) ==
            std::vector<index_t>{0, 0, 0});
    REQUIRE(rim_block_to_vector(rims.vertices[1]) ==
            std::vector<index_t>{0, 3, 4});
    REQUIRE(rim_block_to_vector(rims.faces[1]) ==
            std::vector<index_t>{1, 1, 1});
    REQUIRE(rim_edges_match_their_faces(mesh, rims));
}

// =============================================================================
// make_boundary_rims - Open Rims
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_open_rims", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = rim_three_on_one_edge_3d<index_t, real_t>();
    auto rims = tf::make_boundary_rims(mesh.polygons());

    REQUIRE(rims.size() == 3);
    for (std::size_t i = 0; i < rims.size(); ++i) {
        REQUIRE_FALSE(rims.closed[i]);
        REQUIRE(rims.vertices[i].size() == 3);
        REQUIRE(rims.faces[i].size() == 2);
    }

    REQUIRE(rim_block_to_vector(rims.vertices[0]) ==
            std::vector<index_t>{1, 2, 0});
    REQUIRE(rim_block_to_vector(rims.faces[0]) == std::vector<index_t>{0, 0});
    REQUIRE(rim_block_to_vector(rims.vertices[1]) ==
            std::vector<index_t>{1, 3, 0});
    REQUIRE(rim_block_to_vector(rims.faces[1]) == std::vector<index_t>{1, 1});
    REQUIRE(rim_block_to_vector(rims.vertices[2]) ==
            std::vector<index_t>{1, 4, 0});
    REQUIRE(rim_block_to_vector(rims.faces[2]) == std::vector<index_t>{2, 2});
    REQUIRE(rim_edges_match_their_faces(mesh, rims));
}

// =============================================================================
// make_boundary_rims - Closed Mesh (no rims)
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_closed_mesh", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_tetrahedron_3d<index_t, real_t>();
    auto rims = tf::make_boundary_rims(mesh.polygons());

    REQUIRE(rims.size() == 0);
    REQUIRE(rims.vertices.size() == 0);
    REQUIRE(rims.faces.size() == 0);
    REQUIRE(rims.vertices.data_buffer().size() == 0);
    REQUIRE(rims.faces.data_buffer().size() == 0);
    REQUIRE(rims.closed.size() == 0);
}

// =============================================================================
// make_boundary_rims - Determinism
// =============================================================================

TEMPLATE_TEST_CASE("make_boundary_rims_is_deterministic", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Wide enough that the boundary is gathered over many parallel blocks.
    const std::size_t side = 80;
    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(side, side);
    auto first = tf::make_boundary_rims(mesh.polygons());
    auto second = tf::make_boundary_rims(mesh.polygons());

    REQUIRE(first.size() == 1);
    REQUIRE(first.vertices[0].size() == 4 * (side - 1));

    // The boundary is the mesh's own order, so the rim opens on the first
    // boundary edge of face 0, which is (face[2], face[0]).
    REQUIRE(first.vertices[0][0] == index_t(side));
    REQUIRE(first.vertices[0][1] == index_t(0));

    REQUIRE(rim_block_to_vector(tf::make_range(first.closed)) ==
            rim_block_to_vector(tf::make_range(second.closed)));
    REQUIRE(rim_block_to_vector(tf::make_range(first.vertices.offsets_buffer())) ==
            rim_block_to_vector(tf::make_range(second.vertices.offsets_buffer())));
    REQUIRE(rim_block_to_vector(tf::make_range(first.vertices.data_buffer())) ==
            rim_block_to_vector(tf::make_range(second.vertices.data_buffer())));
    REQUIRE(rim_block_to_vector(tf::make_range(first.faces.offsets_buffer())) ==
            rim_block_to_vector(tf::make_range(second.faces.offsets_buffer())));
    REQUIRE(rim_block_to_vector(tf::make_range(first.faces.data_buffer())) ==
            rim_block_to_vector(tf::make_range(second.faces.data_buffer())));
}

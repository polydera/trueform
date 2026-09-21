/**
 * @file test_vertex_split.cpp
 * @brief Tests for split_non_manifold_vertices
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <trueform/trueform.hpp>
#include "topology_generators.hpp"
#include "type_traits.hpp"
#include <array>
#include <cstddef>
#include <vector>

namespace {

/**
 * @brief The faces of a triangle mesh as comparable triples.
 */
template <typename Index, typename Real>
auto split_nm_faces_of(const tf::polygons_buffer<Index, Real, 3, 3> &mesh) {
    std::vector<std::array<Index, 3>> out;
    for (std::size_t f = 0; f < mesh.faces().size(); ++f)
        out.push_back({Index(mesh.faces()[f][0]), Index(mesh.faces()[f][1]),
                       Index(mesh.faces()[f][2])});
    return out;
}

/**
 * @brief Two triangles meeting at vertex 0 alone.
 */
template <typename Index, typename Real>
auto split_nm_bowtie_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
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
 * @brief Three triangles meeting at vertex 0 alone.
 */
template <typename Index, typename Real>
auto split_nm_triple_bowtie_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(1));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(3), Index(4));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(6));

    return result;
}

/**
 * @brief Two bowties, one at vertex 0 and one at vertex 5.
 */
template <typename Index, typename Real>
auto split_nm_two_bowties_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    auto result = split_nm_bowtie_3d<Index, Real>();

    result.points_buffer().emplace_back(Real(5), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(6), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(5), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(4), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(5), Real(-1), Real(0));

    result.faces_buffer().emplace_back(Index(5), Index(6), Index(7));
    result.faces_buffer().emplace_back(Index(5), Index(8), Index(9));

    return result;
}

/**
 * @brief Two open strips of two triangles meeting at vertex 0 alone, so each
 * fan is walked across a shared edge before it ends on the boundary.
 */
template <typename Index, typename Real>
auto split_nm_boundary_fans_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(4), Index(5));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(6));

    return result;
}

/**
 * @brief Two closed cones meeting at vertex 0 alone, each fan a cycle.
 */
template <typename Index, typename Real>
auto split_nm_hourglass_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(1));
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(-1));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(-1));
    result.points_buffer().emplace_back(Real(-1), Real(-1), Real(-1));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(3), Index(1));
    result.faces_buffer().emplace_back(Index(0), Index(4), Index(5));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(6));
    result.faces_buffer().emplace_back(Index(0), Index(6), Index(4));

    return result;
}

/**
 * @brief An open strip of three triangles at vertex 0 whose MIDDLE triangle
 * is face 0, beside a fan of its own, so the seed claims corners on both
 * sides of itself.
 */
template <typename Index, typename Real>
auto split_nm_mid_seed_strip_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(3), Index(4));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(6));

    return result;
}

/**
 * @brief One vertex carrying both kinds at once: vertex 0 holds a fan of its
 * own in face 3, and an edge (0, 2) that three faces carry.
 */
template <typename Index, typename Real>
auto split_nm_shared_edge_at_bowtie_3d()
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(-1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(-1), Real(0));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(2), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(2), Index(4));
    result.faces_buffer().emplace_back(Index(0), Index(5), Index(6));

    return result;
}

/**
 * @brief A bowtie at vertex 0 beside three faces on edge (5, 6).
 */
template <typename Index, typename Real>
auto split_nm_bowtie_and_edge_3d() -> tf::polygons_buffer<Index, Real, 3, 3> {
    auto result = split_nm_bowtie_3d<Index, Real>();

    result.points_buffer().emplace_back(Real(4), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(5), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(4.5), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(4.5), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(4.5), Real(0), Real(1));

    result.faces_buffer().emplace_back(Index(5), Index(6), Index(7));
    result.faces_buffer().emplace_back(Index(6), Index(5), Index(8));
    result.faces_buffer().emplace_back(Index(5), Index(6), Index(9));

    return result;
}

/**
 * @brief The split answered of a bare form and of a face-membership tagged
 * one: the faces it writes, the point map it states, and the coordinates that
 * map names.
 */
template <typename Index, typename Real>
auto split_nm_check(const tf::polygons_buffer<Index, Real, 3, 3> &mesh,
                    const std::vector<std::array<Index, 3>> &expected_faces,
                    const std::vector<Index> &expected_map) -> void {
    auto polygons = mesh.polygons();
    auto fm = tf::make_face_membership(polygons);
    auto tagged = polygons | tf::tag(fm);

    auto bare = tf::split_non_manifold_vertices(polygons);
    auto from_tagged = tf::split_non_manifold_vertices(tagged);
    auto mapped =
        tf::split_non_manifold_vertices(polygons, tf::return_index_map);

    REQUIRE(split_nm_faces_of(bare) == expected_faces);
    REQUIRE(split_nm_faces_of(from_tagged) == expected_faces);
    REQUIRE(split_nm_faces_of(mapped.first) == expected_faces);

    const auto &point_map = mapped.second;
    REQUIRE(std::vector<Index>(point_map.begin(), point_map.end()) ==
            expected_map);
    REQUIRE(bare.points().size() == expected_map.size());

    for (std::size_t i = 0; i < mesh.points().size(); ++i)
        REQUIRE(expected_map[i] == Index(i));

    for (std::size_t i = 0; i < expected_map.size(); ++i)
        for (std::size_t d = 0; d < 3; ++d)
            REQUIRE(bare.points()[i][d] ==
                    mesh.points()[std::size_t(expected_map[i])][d]);
}

} // anonymous namespace

// =============================================================================
// split_non_manifold_vertices - Separable Fans
// =============================================================================

TEMPLATE_TEST_CASE("split_non_manifold_vertices_bowtie", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_bowtie_3d<index_t, real_t>();

    // Face 0 is the smallest at the apex, so its fan keeps vertex 0.
    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(5), index_t(3), index_t(4)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
    REQUIRE(tf::make_non_manifold_vertices(split.polygons()).size() == 0);
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_triple_bowtie", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_triple_bowtie_3d<index_t, real_t>();

    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(7), index_t(3), index_t(4)},
                    {index_t(8), index_t(5), index_t(6)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(0), index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_two_bowties", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_two_bowties_3d<index_t, real_t>();

    // Minted ids follow the vertices ascending: vertex 0 takes 10, vertex 5
    // takes 11.
    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(10), index_t(3), index_t(4)},
                    {index_t(5), index_t(6), index_t(7)},
                    {index_t(11), index_t(8), index_t(9)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(7), index_t(8), index_t(9),
                    index_t(0), index_t(5)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_boundary_fans", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_boundary_fans_3d<index_t, real_t>();

    // Both fans are open, and the second one carries its whole strip onto the
    // minted vertex.
    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(0), index_t(2), index_t(3)},
                    {index_t(7), index_t(4), index_t(5)},
                    {index_t(7), index_t(5), index_t(6)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_hourglass", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_hourglass_3d<index_t, real_t>();

    // Each fan closes on itself, so the walk returns to its seed and the
    // whole cone follows its apex.
    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(0), index_t(2), index_t(3)},
                    {index_t(0), index_t(3), index_t(1)},
                    {index_t(7), index_t(4), index_t(5)},
                    {index_t(7), index_t(5), index_t(6)},
                    {index_t(7), index_t(6), index_t(4)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
    REQUIRE(tf::make_non_manifold_vertices(split.polygons()).size() == 0);
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_mid_strip_seed", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_mid_seed_strip_3d<index_t, real_t>();

    // The smallest face sits in the middle of its strip, so the keeping fan
    // is only whole if both sides of the seed claim their corners.
    split_nm_check(mesh,
                   {{index_t(0), index_t(2), index_t(3)},
                    {index_t(0), index_t(1), index_t(2)},
                    {index_t(0), index_t(3), index_t(4)},
                    {index_t(7), index_t(5), index_t(6)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    REQUIRE(tf::is_manifold(split.polygons()));
}

// =============================================================================
// split_non_manifold_vertices - What It Leaves Alone
// =============================================================================

TEMPLATE_TEST_CASE("split_non_manifold_vertices_three_faces_on_edge", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_non_manifold_mesh_3d<index_t, real_t>();

    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(1), index_t(0), index_t(3)},
                    {index_t(0), index_t(1), index_t(4)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3),
                    index_t(4)});

    // The vertices the edge holds apart are still named.
    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    auto still = tf::make_non_manifold_vertices(split.polygons());
    REQUIRE(std::vector<index_t>(still.begin(), still.end()) ==
            std::vector<index_t>{index_t(0), index_t(1)});
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_manifold_mesh", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::test::create_grid_mesh_3d<index_t, real_t>(4, 4);
    auto [split, point_map] =
        tf::split_non_manifold_vertices(mesh.polygons(), tf::return_index_map);

    REQUIRE(split_nm_faces_of(split) == split_nm_faces_of(mesh));
    REQUIRE(point_map.size() == mesh.points().size());
    for (std::size_t i = 0; i < point_map.size(); ++i)
        REQUIRE(point_map[i] == index_t(i));
}

// =============================================================================
// split_non_manifold_vertices - Both Kinds At Once
// =============================================================================

TEMPLATE_TEST_CASE("split_non_manifold_vertices_bowtie_and_edge", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_bowtie_and_edge_3d<index_t, real_t>();

    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(10), index_t(3), index_t(4)},
                    {index_t(5), index_t(6), index_t(7)},
                    {index_t(6), index_t(5), index_t(8)},
                    {index_t(5), index_t(6), index_t(9)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6), index_t(7), index_t(8), index_t(9),
                    index_t(0)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    auto still = tf::make_non_manifold_vertices(split.polygons());
    REQUIRE(std::vector<index_t>(still.begin(), still.end()) ==
            std::vector<index_t>{index_t(5), index_t(6)});
}

TEMPLATE_TEST_CASE("split_non_manifold_vertices_shared_edge_at_bowtie", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_shared_edge_at_bowtie_3d<index_t, real_t>();

    // Vertex 0 carries a fan of its own in face 3 AND an edge three faces
    // hold. The edge wins: the vertex is left whole, mints nothing, and maps
    // to itself.
    split_nm_check(mesh,
                   {{index_t(0), index_t(1), index_t(2)},
                    {index_t(0), index_t(2), index_t(3)},
                    {index_t(0), index_t(2), index_t(4)},
                    {index_t(0), index_t(5), index_t(6)}},
                   {index_t(0), index_t(1), index_t(2), index_t(3), index_t(4),
                    index_t(5), index_t(6)});

    auto split = tf::split_non_manifold_vertices(mesh.polygons());
    auto still = tf::make_non_manifold_vertices(split.polygons());
    REQUIRE(std::vector<index_t>(still.begin(), still.end()) ==
            std::vector<index_t>{index_t(0), index_t(2)});
}

// =============================================================================
// split_non_manifold_vertices - Variable-Size Faces
// =============================================================================

TEMPLATE_TEST_CASE("split_non_manifold_vertices_dynamic_faces", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    const auto fixed = split_nm_bowtie_3d<index_t, real_t>();
    auto mesh = tf::make_dynamic(fixed);
    auto [split, point_map] =
        tf::split_non_manifold_vertices(mesh.polygons(), tf::return_index_map);

    REQUIRE(split.faces().size() == 2);
    REQUIRE(split.faces()[0].size() == 3);
    REQUIRE(index_t(split.faces()[0][0]) == index_t(0));
    REQUIRE(index_t(split.faces()[1][0]) == index_t(5));
    REQUIRE(point_map.size() == 6);
    REQUIRE(point_map[5] == index_t(0));
    REQUIRE(tf::is_manifold(split.polygons()));
}

// =============================================================================
// split_non_manifold_vertices - The Point Map
// =============================================================================

TEMPLATE_TEST_CASE("split_non_manifold_vertices_index_map_round_trip", "[topology][split]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = split_nm_triple_bowtie_3d<index_t, real_t>();
    auto [split, point_map] =
        tf::split_non_manifold_vertices(mesh.polygons(), tf::return_index_map);

    REQUIRE(point_map.size() == split.points().size());
    REQUIRE(point_map.size() == mesh.points().size() + 2);

    for (std::size_t i = 0; i < point_map.size(); ++i) {
        REQUIRE(point_map[i] >= index_t(0));
        REQUIRE(point_map[i] < index_t(mesh.points().size()));
        if (i < mesh.points().size())
            REQUIRE(point_map[i] == index_t(i));
        for (std::size_t d = 0; d < 3; ++d)
            REQUIRE(split.points()[i][d] ==
                    mesh.points()[std::size_t(point_map[i])][d]);
    }

    // Both minted points come from the apex the fans were separated at.
    REQUIRE(point_map[mesh.points().size()] == index_t(0));
    REQUIRE(point_map[mesh.points().size() + 1] == index_t(0));
}

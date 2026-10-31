/**
 * @file test_hole_flap_predicates.cpp
 * @brief Tests for the hole and flap predicates
 *
 * Tests for:
 * - has_holes
 * - has_flaps
 * - topology::boundary_rim_is_flap
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <trueform/trueform.hpp>
#include <trueform/topology/boundary/boundary_rim_is_flap.hpp>
#include "type_traits.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace {

/**
 * @brief How many boundary rims of a mesh are holes and how many flaps.
 */
struct hf_rim_census {
    std::size_t holes = 0;
    std::size_t flaps = 0;
};

template <typename Mesh> auto hf_census_of(const Mesh &mesh) -> hf_rim_census {
    auto fm = tf::make_face_membership(mesh.polygons());
    auto rims = tf::make_boundary_rims(mesh.faces(), fm);
    hf_rim_census census{};
    for (std::size_t i = 0; i < rims.size(); ++i) {
        if (tf::topology::boundary_rim_is_flap(mesh.faces(), fm, rims, i))
            ++census.flaps;
        else
            ++census.holes;
    }
    return census;
}

/**
 * @brief The unit box as 12 outward-wound triangles, less the faces named
 *        in `removed`.
 *
 * Faces: 0-1 bottom, 2-3 top, 4-5 front, 6-7 back, 8-9 left, 10-11 right.
 */
template <typename Index, typename Real>
auto hf_box_3d(const std::vector<std::size_t> &removed = {})
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(1));
    result.points_buffer().emplace_back(Real(1), Real(1), Real(1));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(1));

    const std::array<std::array<int, 3>, 12> faces{{{0, 2, 1},
                                                    {0, 3, 2},
                                                    {4, 5, 6},
                                                    {4, 6, 7},
                                                    {0, 1, 5},
                                                    {0, 5, 4},
                                                    {3, 7, 6},
                                                    {3, 6, 2},
                                                    {0, 4, 7},
                                                    {0, 7, 3},
                                                    {1, 2, 6},
                                                    {1, 6, 5}}};
    for (std::size_t f = 0; f < faces.size(); ++f) {
        if (std::find(removed.begin(), removed.end(), f) != removed.end())
            continue;
        result.faces_buffer().emplace_back(Index(faces[f][0]),
                                           Index(faces[f][1]),
                                           Index(faces[f][2]));
    }

    return result;
}

/**
 * @brief Glue a square sheet of four triangles onto the box's bottom front
 *        edge (0, 1), hanging below it in the plane y = 0: that edge then
 *        carries three faces.
 */
template <typename Mesh> auto hf_glue_flap(Mesh &mesh) -> void {
    using Index = std::decay_t<decltype(mesh.faces()[0][0])>;
    using Real = std::decay_t<decltype(mesh.points()[0][0])>;
    const auto base = Index(mesh.points().size());

    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(-0.5));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(-0.5));
    mesh.points_buffer().emplace_back(Real(0), Real(0), Real(-1));
    mesh.points_buffer().emplace_back(Real(1), Real(0), Real(-1));

    mesh.faces_buffer().emplace_back(Index(0), base, Index(base + 1));
    mesh.faces_buffer().emplace_back(Index(0), Index(base + 1), Index(1));
    mesh.faces_buffer().emplace_back(base, Index(base + 2), Index(base + 3));
    mesh.faces_buffer().emplace_back(base, Index(base + 3), Index(base + 1));
}

/**
 * @brief Three triangles on the shared edge (0, 1), plus, when `pinched`,
 *        a fourth meeting the first at vertex 2 alone.
 */
template <typename Index, typename Real>
auto hf_three_on_one_edge_3d(bool pinched)
    -> tf::polygons_buffer<Index, Real, 3, 3> {
    tf::polygons_buffer<Index, Real, 3, 3> result;

    result.points_buffer().emplace_back(Real(0), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(1), Real(0), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(-1), Real(0));
    result.points_buffer().emplace_back(Real(0), Real(0), Real(1));

    result.faces_buffer().emplace_back(Index(0), Index(1), Index(2));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(3));
    result.faces_buffer().emplace_back(Index(0), Index(1), Index(4));

    if (pinched) {
        result.points_buffer().emplace_back(Real(-1), Real(2), Real(0));
        result.points_buffer().emplace_back(Real(1), Real(2), Real(0));
        result.faces_buffer().emplace_back(Index(2), Index(5), Index(6));
    }

    return result;
}

} // anonymous namespace

// =============================================================================
// Closed box: no boundary, neither
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_closed_box", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_box_3d<index_t, real_t>();

    REQUIRE(tf::is_closed(mesh.polygons()));
    REQUIRE_FALSE(tf::has_holes(mesh.polygons()));
    REQUIRE_FALSE(tf::has_flaps(mesh.polygons()));
}

// =============================================================================
// Box with one face removed: a hole
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_box_missing_face", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_box_3d<index_t, real_t>({2});

    REQUIRE(tf::has_holes(mesh.polygons()));
    REQUIRE_FALSE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 1);
    REQUIRE(census.flaps == 0);
}

// =============================================================================
// Box with a sheet glued along one full edge: a flap
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_box_with_flap", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_box_3d<index_t, real_t>();
    hf_glue_flap(mesh);

    // The attachment edge alone carries three faces.
    REQUIRE(tf::make_non_manifold_edges(mesh.polygons()).size() == 1);
    REQUIRE_FALSE(tf::has_holes(mesh.polygons()));
    REQUIRE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 0);
    REQUIRE(census.flaps == 1);
}

// =============================================================================
// Bare open sheet: its free rim is a hole
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_open_sheet", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = tf::make_plane_mesh<index_t>(real_t(1), real_t(1), index_t(3),
                                             index_t(3));

    REQUIRE(tf::has_holes(mesh.polygons()));
    REQUIRE_FALSE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 1);
    REQUIRE(census.flaps == 0);
}

// =============================================================================
// Two holes pinched at one vertex of a closed surface: holes, no flap
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_pinched_holes", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // Faces (4, 6, 7) and (1, 2, 6) share vertex 6 and no edge.
    auto mesh = hf_box_3d<index_t, real_t>({3, 10});

    REQUIRE(tf::make_non_manifold_edges(mesh.polygons()).size() == 0);
    REQUIRE(tf::has_holes(mesh.polygons()));
    REQUIRE_FALSE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 2);
    REQUIRE(census.flaps == 0);
}

// =============================================================================
// Box with a flap and a separately missing face: both
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_flap_and_hole", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_box_3d<index_t, real_t>({2});
    hf_glue_flap(mesh);

    REQUIRE(tf::has_holes(mesh.polygons()));
    REQUIRE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 1);
    REQUIRE(census.flaps == 1);
}

// =============================================================================
// Three faces on one edge: every rim hangs from it
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_three_on_one_edge", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_three_on_one_edge_3d<index_t, real_t>(false);

    REQUIRE_FALSE(tf::has_holes(mesh.polygons()));
    REQUIRE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 0);
    REQUIRE(census.flaps == 3);
}

// =============================================================================
// An open rim with one attached end is a hole
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_one_attached_end", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    // The pinch at vertex 2 splits the first triangle's rim into two open
    // rims, each running from the attachment to the pinch.
    auto mesh = hf_three_on_one_edge_3d<index_t, real_t>(true);
    auto rims = tf::make_boundary_rims(mesh.polygons());
    std::size_t open = 0;
    for (std::size_t i = 0; i < rims.size(); ++i)
        open += !rims.closed[i];
    REQUIRE(rims.size() == 5);
    REQUIRE(open == 4);

    REQUIRE(tf::has_holes(mesh.polygons()));
    REQUIRE(tf::has_flaps(mesh.polygons()));

    auto census = hf_census_of(mesh);
    REQUIRE(census.holes == 3);
    REQUIRE(census.flaps == 2);
}

// =============================================================================
// A tagged face membership is used as given
// =============================================================================

TEMPLATE_TEST_CASE("has_holes_has_flaps_tagged_membership", "[topology][boundary]",
    (tf::test::type_pair<std::int32_t, float>),
    (tf::test::type_pair<std::int64_t, double>))
{
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto mesh = hf_box_3d<index_t, real_t>({2});
    hf_glue_flap(mesh);
    auto fm = tf::make_face_membership(mesh.polygons());
    auto tagged = mesh.polygons() | tf::tag(fm);

    REQUIRE(tf::has_holes(tagged));
    REQUIRE(tf::has_flaps(tagged));
}

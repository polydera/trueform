/**
 * @file test_tet_connectivity.cpp
 * @brief Tet membership, manifold facet link and tet link on hand-built complexes
 *
 * Tet 0 = [0,1,2,3] meets tet 1 = [1,2,3,4] across {1,2,3} and tet 2 =
 * [0,1,2,5] across {0,1,2}; every other facet is on the boundary. The
 * non-manifold complex adds tet 3 = [1,2,3,5], a third holder of {1,2,3} and
 * tet 2's peer across {1,2,5}. The components complex is tets 0 and 2
 * meeting across {1,2,3} beside tet 1 standing alone. A tet repeating a
 * corner reads as a triangle repeating one does in the face membership and
 * the manifold edge link: nothing is deduplicated, the tet stands in the
 * repeated vertex's block once per corner naming it, and every facet peers
 * whatever its corners' blocks share; removing it is tf::cleaned's work.
 *
 * Copyright (c) 2025 Ziga Sajovic, XLAB
 */

#include "type_traits.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <trueform/trueform.hpp>
#include <vector>

namespace {

template <typename Index, typename RealT>
auto make_tet_connectivity_complex(std::size_t n_tets)
    -> tf::tetrahedra_buffer<Index, RealT> {
    const Index corners[] = {0, 1, 2, 3, 1, 2, 3, 4, 0, 1, 2, 5, 1, 2, 3, 5};
    tf::tetrahedra_buffer<Index, RealT> out;
    out.tetras_buffer().allocate(n_tets);
    for (std::size_t i = 0; i < 4 * n_tets; ++i)
        out.tetras_buffer().data_buffer()[i] = corners[i];
    out.points_buffer().emplace_back(RealT(0), RealT(0), RealT(0));
    out.points_buffer().emplace_back(RealT(1), RealT(0), RealT(0));
    out.points_buffer().emplace_back(RealT(0), RealT(1), RealT(0));
    out.points_buffer().emplace_back(RealT(0), RealT(0), RealT(1));
    out.points_buffer().emplace_back(RealT(1), RealT(1), RealT(1));
    out.points_buffer().emplace_back(RealT(0), RealT(0), RealT(-1));
    return out;
}

template <typename Block, typename Index>
auto tet_connectivity_block_is(const Block &block, std::vector<Index> expected)
    -> bool {
    return std::vector<Index>(block.begin(), block.end()) == expected;
}

template <typename Block>
auto manifold_facet_peers_of(const Block &block) -> std::vector<std::int64_t> {
    std::vector<std::int64_t> out;
    for (const auto &peer : block)
        out.push_back(std::int64_t(peer.tet_peer));
    return out;
}

} // namespace

TEMPLATE_TEST_CASE("make_tet_membership_shared_vertices",
                   "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto complex = make_tet_connectivity_complex<index_t, real_t>(3);
    auto tm = tf::make_tet_membership(complex.tetrahedra());

    REQUIRE(tm.size() == 6);
    REQUIRE(tet_connectivity_block_is(tm[0], std::vector<index_t>{2, 0}));
    REQUIRE(tet_connectivity_block_is(tm[1], std::vector<index_t>{2, 1, 0}));
    REQUIRE(tet_connectivity_block_is(tm[2], std::vector<index_t>{2, 1, 0}));
    REQUIRE(tet_connectivity_block_is(tm[3], std::vector<index_t>{1, 0}));
    REQUIRE(tet_connectivity_block_is(tm[4], std::vector<index_t>{1}));
    REQUIRE(tet_connectivity_block_is(tm[5], std::vector<index_t>{2}));
}

TEMPLATE_TEST_CASE("make_manifold_facet_link_peers_both_ways",
                   "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    using peer_t = tf::manifold_facet_peer<index_t>;

    auto complex = make_tet_connectivity_complex<index_t, real_t>(3);
    auto link = tf::make_manifold_facet_link(complex.tetrahedra());

    REQUIRE(link.size() == 3);
    REQUIRE(manifold_facet_peers_of(link[0]) ==
            std::vector<std::int64_t>{1, peer_t::boundary, peer_t::boundary, 2});
    REQUIRE(manifold_facet_peers_of(link[1]) ==
            std::vector<std::int64_t>{peer_t::boundary, peer_t::boundary,
                                      peer_t::boundary, 0});
    REQUIRE(manifold_facet_peers_of(link[2]) ==
            std::vector<std::int64_t>{peer_t::boundary, peer_t::boundary,
                                      peer_t::boundary, 0});

    int representatives = 0;
    for (std::size_t t = 0; t < 3; ++t)
        for (const auto &peer : link[t]) {
            REQUIRE(peer.is_manifold());
            representatives += peer.is_representative(index_t(t));
        }
    REQUIRE(representatives == 10);
}

TEMPLATE_TEST_CASE("make_manifold_facet_link_non_manifold_facet",
                   "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;
    using peer_t = tf::manifold_facet_peer<index_t>;

    auto complex = make_tet_connectivity_complex<index_t, real_t>(4);
    auto tm = tf::make_tet_membership(complex.tetrahedra());
    auto tagged = complex.tetrahedra() | tf::tag(tm);
    auto link = tf::make_manifold_facet_link(tagged);

    REQUIRE(link.size() == 4);
    REQUIRE(manifold_facet_peers_of(link[0]) ==
            std::vector<std::int64_t>{peer_t::non_manifold_representative,
                                      peer_t::boundary, peer_t::boundary, 2});
    REQUIRE(manifold_facet_peers_of(link[1]) ==
            std::vector<std::int64_t>{peer_t::boundary, peer_t::boundary,
                                      peer_t::boundary, peer_t::non_manifold});
    REQUIRE(manifold_facet_peers_of(link[2]) ==
            std::vector<std::int64_t>{3, peer_t::boundary, peer_t::boundary, 0});
    REQUIRE(manifold_facet_peers_of(link[3]) ==
            std::vector<std::int64_t>{peer_t::boundary, peer_t::boundary, 2,
                                      peer_t::non_manifold});
}

TEMPLATE_TEST_CASE("make_tet_link_facet_neighbors", "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto complex = make_tet_connectivity_complex<index_t, real_t>(4);
    auto tl = tf::make_tet_link(complex.tetrahedra());

    REQUIRE(tl.size() == 4);
    REQUIRE(tet_connectivity_block_is(tl[0], std::vector<index_t>{3, 1, 2}));
    REQUIRE(tet_connectivity_block_is(tl[1], std::vector<index_t>{3, 0}));
    REQUIRE(tet_connectivity_block_is(tl[2], std::vector<index_t>{3, 0}));
    REQUIRE(tet_connectivity_block_is(tl[3], std::vector<index_t>{2, 1, 0}));
}

namespace {

template <typename Index, typename RealT>
auto make_tet_components_complex() -> tf::tetrahedra_buffer<Index, RealT> {
    const Index corners[] = {0, 1, 2, 3, 5, 6, 7, 8, 1, 2, 3, 4};
    tf::tetrahedra_buffer<Index, RealT> out;
    out.tetras_buffer().allocate(3);
    for (std::size_t i = 0; i < 12; ++i)
        out.tetras_buffer().data_buffer()[i] = corners[i];
    for (int p = 0; p < 9; ++p)
        out.points_buffer().emplace_back(RealT(p), RealT(p % 2), RealT(p % 3));
    return out;
}

} // namespace

TEMPLATE_TEST_CASE("tet_links_label_connected_components",
                   "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    auto complex = make_tet_components_complex<index_t, real_t>();
    auto tm = tf::make_tet_membership(complex.tetrahedra());
    auto tagged = complex.tetrahedra() | tf::tag(tm);
    auto facet_link = tf::make_manifold_facet_link(tagged);
    auto link = tf::make_tet_link(tagged);

    tf::buffer<index_t> by_facet_link;
    by_facet_link.allocate(3);
    auto n_by_facet_link = tf::label_connected_components(
        by_facet_link, tf::make_applier(facet_link));
    tf::buffer<index_t> by_link;
    by_link.allocate(3);
    auto n_by_link =
        tf::label_connected_components(by_link, tf::make_applier(link));

    REQUIRE(n_by_facet_link == 2);
    REQUIRE(n_by_link == 2);
    REQUIRE(by_facet_link[0] == by_facet_link[2]);
    REQUIRE(by_facet_link[0] != by_facet_link[1]);
    REQUIRE(by_link[0] == by_link[2]);
    REQUIRE(by_link[0] != by_link[1]);
}

TEMPLATE_TEST_CASE("tet_connectivity_degenerate_tet_as_the_triangle_reads",
                   "[topology][connectivity]",
                   (tf::test::type_pair<std::int32_t, float>),
                   (tf::test::type_pair<std::int64_t, double>)) {
    using index_t = typename TestType::index_type;
    using real_t = typename TestType::real_type;

    tf::tetrahedra_buffer<index_t, real_t> complex;
    const index_t corners[] = {0, 1, 2, 3, 1, 1, 2, 3};
    complex.tetras_buffer().allocate(2);
    for (std::size_t i = 0; i < 8; ++i)
        complex.tetras_buffer().data_buffer()[i] = corners[i];
    for (int p = 0; p < 4; ++p)
        complex.points_buffer().emplace_back(real_t(p), real_t(0),
                                             real_t(p % 2));

    auto tm = tf::make_tet_membership(complex.tetrahedra());
    REQUIRE(tet_connectivity_block_is(tm[1], std::vector<index_t>{1, 1, 0}));

    auto link = tf::make_manifold_facet_link(complex.tetrahedra());
    using peer_t = tf::manifold_facet_peer<index_t>;
    REQUIRE(manifold_facet_peers_of(link[0]) ==
            std::vector<std::int64_t>{1, peer_t::boundary, peer_t::boundary,
                                      peer_t::boundary});
    REQUIRE(manifold_facet_peers_of(link[1]) == std::vector<std::int64_t>{0, 0, 0, 0});

    auto tl = tf::make_tet_link(complex.tetrahedra());
    REQUIRE(tet_connectivity_block_is(tl[0], std::vector<index_t>{1}));
    REQUIRE(tet_connectivity_block_is(tl[1], std::vector<index_t>{0, 0, 0, 0}));
}

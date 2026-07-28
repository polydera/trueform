/**
 * @file test_settle_region_triangulations.cpp
 * @brief tf::cut::settle_region_triangulations — the state the stock and the
 *        refined build must both be left in.
 *
 * Three obligations, stated over a real arrangement so the identities are
 * the ones the consumers use. A region that emitted nothing is a hole in
 * the surface and goes back through the wave, and what is still empty
 * afterwards is what the step answers. Pending welds are folded into the
 * authority the single-binary-search consumer reads. And every weld,
 * whoever found it, reaches the triangles that were emitted before it
 * existed: gating that carry on the last round having produced one leaves
 * a recovery-era weld applied to the identity table and to nothing else,
 * so two corners of one triangle resolve to one output vertex and the edge
 * between them goes unmatched.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>

#include <trueform/core/buffer.hpp>
#include <trueform/core/point.hpp>
#include <trueform/core/range.hpp>
#include <trueform/cut/detail/region_triangulation_types.hpp>
#include <trueform/cut/face_regions.hpp>
#include <trueform/cut/impl/region_triangulator.hpp>
#include <trueform/cut/impl/settle_region_triangulations.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/exact/make_kernel.hpp>
#include <trueform/intersect/graph/face_descriptor.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/graph/vertex.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/trueform.hpp>

#include <array>
#include <cstddef>

namespace {

using Index = int;
using Real = double;
using Int = tf::exact::int64;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;
using vertex_t = tf::intersect::graph::vertex<Index>;
using vsource = tf::intersect::graph::vertex_source;
using vertex_key_t = std::array<Index, 2>;
using merge_t = tf::cut::detail::region_triangulation_merge<Index>;
using split_t = tf::cut::detail::region_triangulation_split<Index, Int>;
using triangle_t = std::array<vertex_t, 3>;

constexpr Index n_tags = 2;

/// A sheet cut by a prism: two tags, a real intersection graph, real
/// region walks, and no promoted faces, so `ranges` is one block per
/// region and the loop index IS the region index.
template <typename Body> auto with_arrangement(const Body &body) -> void {
  mesh_t sheet;
  sheet.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  sheet.points_buffer().emplace_back(Real(10), Real(0), Real(0));
  sheet.points_buffer().emplace_back(Real(0), Real(10), Real(0));
  sheet.points_buffer().emplace_back(Real(10), Real(10), Real(0));
  sheet.faces_buffer().emplace_back(0, 1, 2);
  sheet.faces_buffer().emplace_back(1, 3, 2);
  mesh_t prism;
  {
    const std::array<std::array<Real, 2>, 3> cs = {
        {{2.0, 2.0}, {4.0, 2.0}, {2.0, 4.0}}};
    for (auto &c : cs)
      prism.points_buffer().emplace_back(Real(c[0]), Real(c[1]), Real(-1));
    for (auto &c : cs)
      prism.points_buffer().emplace_back(Real(c[0]), Real(c[1]), Real(1));
    for (int e = 0; e < 3; ++e) {
      int a = e, b = (e + 1) % 3;
      prism.faces_buffer().emplace_back(a, b, b + 3);
      prism.faces_buffer().emplace_back(a, b + 3, a + 3);
    }
  }
  std::array<mesh_t *, 2> meshes{&sheet, &prism};

  std::array<tf::aabb_tree<Index, Real, 3>, 2> trees;
  std::array<tf::face_membership<Index>, 2> fms;
  std::array<tf::manifold_edge_link<Index, 3>, 2> mels;
  for (int i = 0; i < 2; ++i) {
    trees[i] = tf::aabb_tree<Index, Real, 3>(meshes[i]->polygons(),
                                             tf::config_tree(4, 4));
    fms[i].build(meshes[i]->polygons());
    mels[i] = tf::make_manifold_edge_link(meshes[i]->polygons());
  }
  auto forms = tf::make_mapped_range(tf::make_sequence_range(2), [&](int i) {
    return meshes[i]->polygons() | tf::tag(trees[i]) | tf::tag(fms[i]) |
           tf::tag(mels[i]);
  });
  const auto mode = tf::intersect_mode::primitives |
                    tf::intersect_mode::resolve_crossing_contours;
  tf::intersections_between_polygons<Index, Real, Int> ibp;
  ibp.build(forms, mode);
  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<Int, 3> {
    return conv.convert(forms[tag].points()[id]);
  };
  tf::intersection_graph<Index, Int> ig;
  ig.build(ibp, apply_to_face, get_mesh_point, mode,
           tf::exact::make_kernel(conv, 0.0));
  tf::face_regions<Index, Int> fr;
  fr.build(ig, apply_to_face, get_mesh_point);
  tf::cut::region_triangulator<Index, Int> rt;
  rt.build(fr, ig, forms, apply_to_face, get_mesh_point);
  REQUIRE(rt.n_failed == 0);
  REQUIRE(rt.promoted_descriptors().size() == 0);

  // the region that carries the most triangles under a plain build: the
  // one whose emptiness is unmistakable
  Index region = 0;
  Index best = 0;
  for (Index loop = 0; loop < Index(fr.loops().size()); ++loop) {
    const Index count = rt.ranges[std::size_t(loop)][1] -
                        rt.ranges[std::size_t(loop)][0];
    if (count > best) {
      best = count;
      region = loop;
    }
  }
  REQUIRE(best > 0);

  body(fr, ig, apply_to_face, get_mesh_point, region);
}

auto original(Index id) -> vertex_t {
  return vertex_t{vsource::original, id, {0, tf::topo_type::face}};
}

auto make_merge(const vertex_key_t &from, const vertex_key_t &to_key,
                const vertex_t &to) -> merge_t {
  merge_t merge{};
  merge.from = from;
  merge.to_key = to_key;
  merge.to = to;
  return merge;
}

/// The empty inputs settle takes alongside the ones a case cares about.
struct rest {
  tf::buffer<char> no_dead;
  tf::buffer<tf::point<Int, 3>> extra_points;
  tf::buffer<split_t> splits;
  tf::buffer<tf::intersect::graph::face_descriptor<Index>> promoted;
};

} // namespace

TEST_CASE("settle_region_triangulations: a region that emitted nothing goes "
          "back through the wave",
          "[cut][settle_region_triangulations]") {
  with_arrangement([](const auto &fr, const auto &ig, const auto &apply_to_face,
                      const auto &get_mesh_point, Index region) {
    rest other;
    tf::buffer<merge_t> merges, pending;
    tf::buffer<std::array<Index, 2>> ranges;
    tf::buffer<triangle_t> triangles;
    tf::buffer<Index> failed;

    // The wave retries what a round can still offer something to, so it
    // is entered the way the refined path enters it: an authority that
    // already carries a weld from an earlier round. This one names
    // created identities past the table, so it is fresh to the wave and
    // owns no walk — the region is retried on its own merits.
    const Index unreachable = Index(ig.points().size()) + 1000;
    merges.push_back(
        make_merge({n_tags, unreachable}, {n_tags, unreachable + 1},
                   vertex_t{vsource::created, unreachable + 1,
                            {0, tf::topo_type::face}}));
    ranges.allocate(std::size_t(fr.loops().size()));
    for (auto &range : ranges)
      range = {Index(0), Index(0)};
    failed.push_back(region);

    const Index still_empty = tf::cut::settle_region_triangulations(
        fr, ig, apply_to_face, get_mesh_point, tf::make_range(other.no_dead),
        n_tags, Index(ig.points().size()), other.extra_points, other.splits,
        merges, pending, other.promoted, ranges, triangles, failed);

    CHECK(still_empty == 0);
    CHECK(failed.size() == 0u);
    const auto recovered = ranges[std::size_t(region)];
    CHECK(recovered[1] > recovered[0]);
    CHECK(triangles.size() == std::size_t(recovered[1] - recovered[0]));
  });
}

TEST_CASE("settle_region_triangulations: pending welds are folded into the "
          "authority and reach the stream",
          "[cut][settle_region_triangulations]") {
  with_arrangement([](const auto &fr, const auto &ig, const auto &apply_to_face,
                      const auto &get_mesh_point, Index region) {
    rest other;
    const Index tag = fr.descriptors()[region].tag;
    tf::buffer<merge_t> merges, pending;
    tf::buffer<std::array<Index, 2>> ranges;
    tf::buffer<triangle_t> triangles;
    tf::buffer<Index> failed;

    pending.push_back(make_merge({tag, 0}, {tag, 1}, original(1)));
    triangles.push_back({original(0), original(2), original(3)});
    ranges.allocate(std::size_t(fr.loops().size()));
    for (auto &range : ranges)
      range = {Index(0), Index(0)};
    ranges[std::size_t(region)] = {Index(0), Index(1)};

    const Index still_empty = tf::cut::settle_region_triangulations(
        fr, ig, apply_to_face, get_mesh_point, tf::make_range(other.no_dead),
        n_tags, Index(ig.points().size()), other.extra_points, other.splits,
        merges, pending, other.promoted, ranges, triangles, failed);

    CHECK(still_empty == 0);
    CHECK(pending.size() == 0u);
    REQUIRE(merges.size() == 1u);
    CHECK(merges[0].from == vertex_key_t{tag, 0});
    CHECK(merges[0].to_key == vertex_key_t{tag, 1});
    REQUIRE(triangles.size() == 1u);
    CHECK(triangles[0][0].id == 1);
  });
}

TEST_CASE("settle_region_triangulations: a weld reaches triangles emitted "
          "before it existed, with no round of its own to announce it",
          "[cut][settle_region_triangulations]") {
  with_arrangement([](const auto &fr, const auto &ig, const auto &apply_to_face,
                      const auto &get_mesh_point, Index region) {
    rest other;
    const Index tag = fr.descriptors()[region].tag;
    tf::buffer<merge_t> merges, pending;
    tf::buffer<std::array<Index, 2>> ranges;
    tf::buffer<triangle_t> triangles;
    tf::buffer<Index> failed;

    // the authority already holds the weld and nothing is incoming: this
    // is a recovery-era weld arriving at a settle whose own round found
    // none, and it still owns every triangle already in the stream
    merges.push_back(make_merge({tag, 0}, {tag, 1}, original(1)));
    triangles.push_back({original(0), original(1), original(2)});
    triangles.push_back({original(1), original(2), original(3)});
    ranges.allocate(std::size_t(fr.loops().size()));
    for (auto &range : ranges)
      range = {Index(0), Index(0)};
    ranges[std::size_t(region)] = {Index(0), Index(2)};

    const Index still_empty = tf::cut::settle_region_triangulations(
        fr, ig, apply_to_face, get_mesh_point, tf::make_range(other.no_dead),
        n_tags, Index(ig.points().size()), other.extra_points, other.splits,
        merges, pending, other.promoted, ranges, triangles, failed);

    CHECK(still_empty == 0);
    // the first triangle has two corners on one identity: it is an edge,
    // not a triangle, and it is gone
    REQUIRE(triangles.size() == 1u);
    CHECK(triangles[0][0].id == 1);
    CHECK(triangles[0][1].id == 2);
    CHECK(triangles[0][2].id == 3);
    CHECK(ranges[std::size_t(region)] == std::array<Index, 2>{0, 1});
  });
}

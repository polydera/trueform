/**
 * @file test_triangulated_csg.cpp
 * @brief Tests for the csg graph's per-loop triangulation store
 *        (tf::csg::graph::loop_triangulations) through the mesh and
 *        domain extractions.
 *
 * Two-box fixtures (generic overlap, stacked full-coplanar contact,
 * partial coplanar stack). For each: build the graph in stock and
 * refined triangulation modes, extract union and intersection meshes
 * through compute_chosen_sides / evaluate_per_domain, and require
 * closedness. Stock mode must reproduce the dispatcher path's triangle
 * count exactly; the refined store must be bit-identical across two
 * independent builds (determinism). A WantLabels section checks face
 * provenance on the generic fixture.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/csg.hpp>
#include <trueform/csg/graph/compute_chosen_sides.hpp>
#include <trueform/csg/graph/compute_domain_membership.hpp>
#include <trueform/csg/graph/compute_domain_partition.hpp>
#include <trueform/csg/graph/evaluate_per_domain.hpp>
#include <trueform/topology/is_closed.hpp>
#include <trueform/trueform.hpp>

#include <tbb/parallel_invoke.h>

#include <deque>
#include <string>
#include <utility>
#include <vector>

using Index = int;
using Real = float;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

namespace {

auto make_box(Real x0, Real y0, Real z0, Real x1, Real y1, Real z1)
    -> mesh_t {
  mesh_t m;
  Real xs[2] = {x0, x1}, ys[2] = {y0, y1}, zs[2] = {z0, z1};
  for (int k = 0; k < 8; ++k)
    m.points_buffer().emplace_back(xs[k & 1], ys[(k >> 1) & 1],
                                   zs[(k >> 2) & 1]);
  const int f[12][3] = {{0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6},
                        {0, 1, 4}, {1, 5, 4}, {2, 6, 3}, {3, 6, 7},
                        {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5}};
  for (auto &t : f)
    m.faces_buffer().emplace_back(t[0], t[1], t[2]);
  return m;
}

// second operand per fixture case: 0 generic overlap, 1 stacked
// full-coplanar contact, 2 small box stacked on big (partial coplanar)
auto make_second_box(int cs) -> mesh_t {
  if (cs == 1)
    return make_box(0.f, 0.f, 1.f, 1.f, 1.f, 2.f);
  if (cs == 2)
    return make_box(0.25f, 0.25f, 1.f, 0.75f, 0.75f, 2.f);
  return make_box(0.37f, 0.41f, 0.53f, 1.37f, 1.41f, 1.53f);
}

struct op_t {
  mesh_t mesh;
  tf::face_membership<Index> fm;
  tf::manifold_edge_link<Index, 3> mel;
  tf::aabb_tree<Index, Real, 3> tree;
  explicit op_t(mesh_t m) : mesh(std::move(m)) {
    auto pr = mesh.polygons();
    tbb::parallel_invoke([&] { tree.build(pr, tf::config_tree(4, 12)); },
                         [&] {
                           fm.build(pr);
                           mel.build(pr.faces(), fm);
                         });
  }
};

auto tag_of(const op_t &op) {
  return op.mesh.polygons() | tf::tag(op.fm) | tf::tag(op.mel) |
         tf::tag(op.tree);
}

const std::vector<std::pair<std::string, tf::csg::expr>> k_exprs = {
    {"union", tf::csg::op(0) | tf::csg::op(1)},
    {"intersection", tf::csg::op(0) & tf::csg::op(1)}};

template <typename Graph, typename Forms>
auto extract_mesh(const Graph &g, const Forms &forms,
                  const tf::csg::expr &e) {
  auto E = e.compile().evaluator();
  auto mem = tf::csg::graph::evaluate_per_domain(g.inclusion(), E);
  auto chosen = tf::csg::graph::compute_chosen_sides(g.descriptor(), mem);
  return tf::csg::graph::make_csg_mesh<Real, false>(
      g.arrangement(), g.face_cuts(), g.created_points(), forms, chosen,
      g.converter(), g.loop_triangulations());
}

} // namespace

TEST_CASE("triangulation store: box fixtures closed in both modes, stock "
          "parity, refined determinism",
          "[csg][graph][store]") {
  for (int cs = 0; cs < 3; ++cs) {
    DYNAMIC_SECTION("fixture case " << cs) {
      std::deque<op_t> ops;
      ops.emplace_back(make_box(0.f, 0.f, 0.f, 1.f, 1.f, 1.f));
      ops.emplace_back(make_second_box(cs));

      auto identity =
          tf::make_frame(tf::make_identity_transformation<Real, 3>());
      using form_t = decltype(tag_of(ops[0]) | tf::tag(identity));
      std::vector<form_t> forms;
      for (auto &op : ops)
        forms.push_back(tag_of(op) | tf::tag(identity));
      auto rng = tf::make_range(forms);

      // dispatcher path (default = stock triangulation store)
      auto ref_graph = tf::make_csg_graph(rng);

      for (auto mode : {tf::triangulation_type::cdt,
                        tf::triangulation_type::refined_cdt}) {
        tf::csg_graph<decltype(rng), tf::none_t> graph(
            rng, {},
            {tf::intersect_mode::primitives |
             tf::intersect_mode::resolve_crossing_contours},
            {}, mode);
        for (const auto &[name, e] : k_exprs) {
          DYNAMIC_SECTION((mode == tf::triangulation_type::cdt
                               ? "stock "
                               : "refined ")
                          << name) {
            auto m = extract_mesh(graph, rng, e);
            REQUIRE(tf::is_closed(m.polygons()));
            if (mode == tf::triangulation_type::cdt) {
              auto ref = tf::make_csg_mesh(ref_graph, e);
              REQUIRE(m.polygons().size() == ref.polygons().size());
            }
          }
        }
      }

      SECTION("refined store is deterministic across builds") {
        auto build = [&] {
          return tf::csg_graph<decltype(rng), tf::none_t>(
              rng, {},
              {tf::intersect_mode::primitives |
               tf::intersect_mode::resolve_crossing_contours},
              {}, tf::triangulation_type::refined_cdt);
        };
        auto g0 = build();
        auto g1 = build();
        const auto &s0 = g0.loop_triangulations();
        const auto &s1 = g1.loop_triangulations();

        // created points (ig ++ splits ++ steiner), exact int lattice
        const auto &c0 = g0.created_points();
        const auto &c1 = g1.created_points();
        REQUIRE(c0.size() == c1.size());
        for (std::size_t i = 0; i < c0.size(); ++i)
          for (int k = 0; k < 3; ++k)
            REQUIRE(c0[i][k] == c1[i][k]);

        REQUIRE(s0.tris.size() == s1.tris.size());
        for (std::size_t i = 0; i < s0.tris.size(); ++i)
          for (int k = 0; k < 3; ++k)
            REQUIRE(s0.tris[i][std::size_t(k)] ==
                    s1.tris[i][std::size_t(k)]);

        REQUIRE(s0.conf_tris.size() == s1.conf_tris.size());
        for (std::size_t i = 0; i < s0.conf_tris.size(); ++i)
          for (int k = 0; k < 3; ++k)
            REQUIRE(s0.conf_tris[i][std::size_t(k)] ==
                    s1.conf_tris[i][std::size_t(k)]);
      }
    }
  }
}

TEST_CASE("triangulation store: WantLabels face provenance on the generic "
          "fixture",
          "[csg][graph][store][source_ids]") {
  std::deque<op_t> ops;
  ops.emplace_back(make_box(0.f, 0.f, 0.f, 1.f, 1.f, 1.f));
  ops.emplace_back(make_second_box(0));

  auto identity = tf::make_frame(tf::make_identity_transformation<Real, 3>());
  using form_t = decltype(tag_of(ops[0]) | tf::tag(identity));
  std::vector<form_t> forms;
  for (auto &op : ops)
    forms.push_back(tag_of(op) | tf::tag(identity));
  auto rng = tf::make_range(forms);

  for (auto mode : {tf::triangulation_type::cdt,
                    tf::triangulation_type::refined_cdt}) {
    tf::csg_graph<decltype(rng), tf::none_t> graph(
        rng, {},
        {tf::intersect_mode::primitives |
         tf::intersect_mode::resolve_crossing_contours},
        {}, mode);
    auto E = (tf::csg::op(0) | tf::csg::op(1)).compile().evaluator();
    auto membership = tf::csg::graph::compute_domain_membership(
        graph.descriptor(), graph.inclusion(),
        graph.arrangement().open_component_mask(),
        graph.domain_nesting_merges(),
        tf::domain_config::exclude_outer_shell |
            tf::domain_config::ignore_open_fragments,
        E);
    auto part = tf::csg::graph::compute_domain_partition(
        membership.domain_of_side, membership.n_components, membership.keep);
    auto [cells, ids, tag_blocks, face_blocks] =
        tf::csg::graph::make_csg_domains<Real, true, false>(
            graph.arrangement(), graph.face_cuts(), graph.created_points(),
            rng, part, graph.converter(), graph.loop_triangulations());

    REQUIRE(cells.size() > 0);
    REQUIRE(std::size_t(tag_blocks.size()) == cells.size());
    REQUIRE(std::size_t(face_blocks.size()) == cells.size());
    for (std::size_t k = 0; k < cells.size(); ++k) {
      auto tb = tag_blocks[Index(k)];
      auto fb = face_blocks[Index(k)];
      REQUIRE(std::size_t(tb.size()) == cells[k].polygons().size());
      REQUIRE(std::size_t(fb.size()) == cells[k].polygons().size());
      for (std::size_t j = 0; j < std::size_t(tb.size()); ++j) {
        auto t = tb[j];
        REQUIRE(t >= Index(0));
        REQUIRE(std::size_t(t) < forms.size());
        REQUIRE(fb[j] >= Index(0));
        REQUIRE(std::size_t(fb[j]) < forms[std::size_t(t)].faces().size());
      }
    }
  }
}

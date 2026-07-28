/**
 * @file test_component_labels.cpp
 * @brief Hand-verifiable structural tests for tf::cut::loop_connectivity.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/arrangements/component_labels.hpp>
#include <trueform/cut/arrangements/make_arrangement_descriptor.hpp>
#include <trueform/cut/detail/make_connectivity_face_membership.hpp>
#include <trueform/cut/face_regions.hpp>
#include <trueform/cut/impl/loop_connectivity.hpp>
#include <trueform/cut/impl/region_triangulator.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>
#include <trueform/topology/structures/compute_face_link_per_edge.hpp>

#include <algorithm>

using Index = int;
using mesh_t = tf::polygons_buffer<int, float, 3, 3>;

namespace {

auto make_triangle(const std::array<std::array<float, 3>, 3> &pts) -> mesh_t {
  mesh_t mesh;
  mesh.points_buffer().allocate(3);
  mesh.faces_buffer().allocate(1);
  for (int i = 0; i < 3; ++i) {
    auto p = mesh.points()[i];
    for (int d = 0; d < 3; ++d)
      p[d] = pts[i][d];
  }
  auto f = mesh.faces()[0];
  f[0] = 0;
  f[1] = 1;
  f[2] = 2;
  return mesh;
}

} // namespace

TEST_CASE("loop_connectivity: empty state has no tags", "[component_labels]") {
  tf::cut::loop_connectivity<Index> connectivity;
  CHECK(connectivity.n_tags() == 0);
  connectivity.clear();
  CHECK(connectivity.n_tags() == 0);
}

TEST_CASE("loop_connectivity: compact IDs preserve shared-edge neighbours",
          "[component_labels]") {
  auto compute = [](const std::array<Index, 6> &source_ids,
                    Index bounded_id_count, tf::buffer<Index> &offsets,
                    tf::buffer<Index> &neighbours) {
    tf::buffer<Index> flat_ids;
    flat_ids.allocate(source_ids.size());
    std::copy(source_ids.begin(), source_ids.end(), flat_ids.begin());
    const std::array<Index, 3> face_offsets{0, 3, 6};
    auto faces =
        tf::make_faces(tf::make_range(face_offsets), tf::make_range(flat_ids));
    auto membership = tf::cut::detail::make_connectivity_face_membership(
        faces, flat_ids, bounded_id_count);
    tf::topology::compute_face_link_per_edge(faces, membership, offsets,
                                             neighbours);
  };

  tf::buffer<Index> dense_offsets;
  tf::buffer<Index> dense_neighbours;
  compute({0, 1, 2, 2, 1, 3}, 4, dense_offsets, dense_neighbours);

  tf::buffer<Index> sparse_offsets;
  tf::buffer<Index> sparse_neighbours;
  compute({0, 1000, 2000, 2000, 1000, 3000}, 4001, sparse_offsets,
          sparse_neighbours);

  REQUIRE(dense_offsets.size() == sparse_offsets.size());
  REQUIRE(dense_neighbours.size() == sparse_neighbours.size());
  CHECK(std::equal(dense_offsets.begin(), dense_offsets.end(),
                   sparse_offsets.begin()));
  CHECK(std::equal(dense_neighbours.begin(), dense_neighbours.end(),
                   sparse_neighbours.begin()));

  auto dense_edges =
      tf::make_offset_block_range(dense_offsets, dense_neighbours);
  REQUIRE(dense_edges.size() == 6);
  REQUIRE(dense_edges[1].size() == 1);
  CHECK(dense_edges[1][0] == 1);
  REQUIRE(dense_edges[3].size() == 1);
  CHECK(dense_edges[3][0] == 0);
}

TEST_CASE(
    "loop_connectivity: two triangles sharing an edge in different planes",
    "[component_labels][cross-tag]") {
  // Triangle A in XY plane, triangle B in XZ plane, sharing the edge
  // from (0,0,0) to (1,0,0). They are NOT coplanar — coplanar dedup
  // must not fuse them just because their local vertex indices
  // happen to collide.
  auto mesh_a = make_triangle({{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}});
  auto mesh_b = make_triangle({{{0, 0, 0}, {1, 0, 0}, {0, 0, 1}}});

  tf::aabb_tree<Index, float, 3> tr_a(mesh_a.polygons(), tf::config_tree(4, 4));
  tf::aabb_tree<Index, float, 3> tr_b(mesh_b.polygons(), tf::config_tree(4, 4));
  auto fm_a = tf::make_face_membership(mesh_a.polygons());
  auto fm_b = tf::make_face_membership(mesh_b.polygons());
  auto mel_a = tf::make_manifold_edge_link(mesh_a.polygons());
  auto mel_b = tf::make_manifold_edge_link(mesh_b.polygons());

  using form_t = decltype(mesh_a.polygons() | tf::tag(tr_a) | tf::tag(fm_a) |
                          tf::tag(mel_a));
  std::vector<form_t> forms;
  forms.push_back(mesh_a.polygons() | tf::tag(tr_a) | tf::tag(fm_a) |
                  tf::tag(mel_a));
  forms.push_back(mesh_b.polygons() | tf::tag(tr_b) | tf::tag(fm_b) |
                  tf::tag(mel_b));
  auto forms_range = tf::make_range(forms);

  tf::intersections_between_polygons<Index, float> ibp;
  ibp.build(forms_range, tf::intersect_mode::primitives |
                             tf::intersect_mode::resolve_crossing_contours);

  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms_range[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<std::int32_t, 3> {
    return conv.convert(forms_range[tag].points()[id]);
  };
  auto get_point = [&](const auto &v, Index tag) -> tf::point<std::int32_t, 3> {
    if (v.source == tf::intersect::graph::vertex_source::created)
      return tf::point<std::int32_t, 3>{};
    return get_mesh_point(int(tag), v.id);
  };

  tf::intersection_graph<Index> ig;
  ig.build(ibp, apply_to_face, get_mesh_point,
           tf::intersect_mode::primitives |
               tf::intersect_mode::resolve_crossing_contours);
  tf::face_regions<Index> fr;
  fr.build(ig, apply_to_face, get_mesh_point);
  tf::cut::region_triangulator<Index> rt;
  rt.build(fr, ig, forms_range, apply_to_face, get_mesh_point);
  // the arrangement tier detects stacks; the label tier builds
  // connectivity (cleaned with the arrangement's dead mask) + labels
  tf::buffer<std::array<Index, 3>> pairs; // no stacks in this fixture
  tf::buffer<char> dead;
  dead.allocate(std::size_t(fr.loops().size()));
  tf::parallel_fill(dead, char(0));
  const auto n_created =
      static_cast<Index>(ig.points().size() + rt.extra_points.size());
  const std::array<Index, 2> dense_point_counts{3, 3};
  const std::array<Index, 2> sparse_point_counts{30000, 30000};
  tf::cut::loop_connectivity<Index> dense_connectivity;
  tf::cut::loop_connectivity<Index> sparse_connectivity;
  dense_connectivity.build(fr, tf::make_range(dead),
                           tf::make_range(dense_point_counts), n_created);
  sparse_connectivity.build(fr, tf::make_range(dead),
                            tf::make_range(sparse_point_counts), n_created);

  auto dense_edges = dense_connectivity.connectivity_per_carrier_edge();
  auto sparse_edges = sparse_connectivity.connectivity_per_carrier_edge();
  REQUIRE(dense_edges.size() == sparse_edges.size());
  for (std::size_t carrier_id = 0; carrier_id < dense_edges.size();
       ++carrier_id) {
    REQUIRE(dense_edges[carrier_id].size() == sparse_edges[carrier_id].size());
    for (std::size_t edge_id = 0; edge_id < dense_edges[carrier_id].size();
         ++edge_id) {
      auto dense_neighbours = dense_edges[carrier_id][edge_id];
      auto sparse_neighbours = sparse_edges[carrier_id][edge_id];
      REQUIRE(dense_neighbours.size() == sparse_neighbours.size());
      for (std::size_t neighbour_id = 0; neighbour_id < dense_neighbours.size();
           ++neighbour_id)
        CHECK(dense_neighbours[neighbour_id] ==
              sparse_neighbours[neighbour_id]);
    }
  }

  tf::cut::component_labels<Index> ag;
  ag.build(fr, forms_range, tf::make_range(pairs), tf::make_range(dead),
           n_created);

  REQUIRE(ag.n_components() == Index(2));
  CHECK(ag.coplanar_pairs().size() == 0);
  auto labels_a = ag.polygon_labels(0);
  auto labels_b = ag.polygon_labels(1);
  REQUIRE(labels_a.size() == 1);
  REQUIRE(labels_b.size() == 1);
  CHECK(labels_a[0] == tf::cut::component_labels<Index>::none_label);
  CHECK(labels_b[0] == tf::cut::component_labels<Index>::none_label);
  auto ll = ag.loop_labels();
  REQUIRE(ll.size() == 2);
  CHECK(ll[0] != tf::cut::component_labels<Index>::none_label);
  CHECK(ll[1] != tf::cut::component_labels<Index>::none_label);
  CHECK(ll[0] != ll[1]);

  auto mask = ag.open_component_mask();
  REQUIRE(mask.size() == 2);
  CHECK(mask[0] == char(1));
  CHECK(mask[1] == char(1));

  auto desc = tf::cut::make_arrangement_descriptor<std::int32_t>(
      ag, fr, get_point, apply_to_face);

  REQUIRE(desc.tag_of_component.size() == 2);
  CHECK(desc.tag_of_component[0] != desc.tag_of_component[1]);
  for (Index c = 0; c < Index(2); ++c) {
    const Index t = desc.tag_of_component[c];
    CHECK((t == Index(0) || t == Index(1)));
  }

  REQUIRE(desc.n_bundles == Index(2));
  for (Index b = 0; b < desc.n_bundles; ++b) {
    auto tags = desc.bundle_to_tags[b];
    CHECK(tags.size() == 1);
  }

  for (Index c = 0; c < Index(2); ++c) {
    CHECK(desc.domain_of_side[2 * c + 0] == desc.domain_of_side[2 * c + 1]);
  }
  REQUIRE(desc.n_domains == Index(2));
}

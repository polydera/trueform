/**
 * @file test_arrangement_graph.cpp
 * @brief Hand-verifiable structural tests for tf::arrangement_graph.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/core/polygons_buffer.hpp>
#include <trueform/core/views/mapped_range.hpp>
#include <trueform/core/views/sequence_range.hpp>
#include <trueform/cut/arrangement_graph.hpp>
#include <trueform/cut/arrangements/make_arrangement_descriptor.hpp>
#include <trueform/cut/face_cuts.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/spatial/aabb_tree.hpp>
#include <trueform/topology/make_face_membership.hpp>
#include <trueform/topology/make_manifold_edge_link.hpp>

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

TEST_CASE("arrangement_graph: two triangles sharing an edge in different planes",
          "[arrangement_graph][cross-tag]") {
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
  tf::face_cuts<Index> fc;
  fc.build(ig, apply_to_face, get_mesh_point);
  tf::arrangement_graph<Index> ag;
  ag.build(ig, fc, forms_range);

  REQUIRE(ag.n_components() == Index(2));
  CHECK(ag.coplanar_pairs().size() == 0);
  auto labels_a = ag.polygon_labels(0);
  auto labels_b = ag.polygon_labels(1);
  REQUIRE(labels_a.size() == 1);
  REQUIRE(labels_b.size() == 1);
  CHECK(labels_a[0] == tf::arrangement_graph<Index>::none_label);
  CHECK(labels_b[0] == tf::arrangement_graph<Index>::none_label);
  auto ll = ag.loop_labels();
  REQUIRE(ll.size() == 2);
  CHECK(ll[0] != tf::arrangement_graph<Index>::none_label);
  CHECK(ll[1] != tf::arrangement_graph<Index>::none_label);
  CHECK(ll[0] != ll[1]);

  auto mask = ag.open_component_mask();
  REQUIRE(mask.size() == 2);
  CHECK(mask[0] == char(1));
  CHECK(mask[1] == char(1));

  auto desc = tf::cut::make_arrangement_descriptor<std::int32_t>(
      ag, fc, get_point, apply_to_face);

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

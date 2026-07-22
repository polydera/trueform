#include <catch2/catch_test_macros.hpp>

#include <trueform/cut/face_cuts.hpp>
#include <trueform/cut/face_regions.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/intersect/exact/make_kernel.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/trueform.hpp>

#include <algorithm>
#include <array>
#include <map>
#include <vector>

namespace {

using Index = int;
using Real = double;
using Int = tf::exact::int64;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;
using vertex_t = tf::intersect::graph::vertex<Index>;
using vsource = tf::intersect::graph::vertex_source;
using T2 = tf::exact::meta<Int>::T2;

auto make_sheet() -> mesh_t {
  mesh_t m;
  m.points_buffer().emplace_back(0.f, 0.f, 0.f);
  m.points_buffer().emplace_back(10.f, 0.f, 0.f);
  m.points_buffer().emplace_back(0.f, 10.f, 0.f);
  m.faces_buffer().emplace_back(0, 1, 2);
  return m;
}

/// Open triangular prism (sides only) piercing the sheet's interior.
auto make_prism() -> mesh_t {
  mesh_t m;
  const std::array<std::array<Real, 2>, 3> cs = {
      {{2.0, 2.0}, {4.0, 2.0}, {2.0, 4.0}}};
  for (auto &c : cs)
    m.points_buffer().emplace_back(Real(c[0]), Real(c[1]), Real(-1));
  for (auto &c : cs)
    m.points_buffer().emplace_back(Real(c[0]), Real(c[1]), Real(1));
  for (int e = 0; e < 3; ++e) {
    int a = e, b = (e + 1) % 3;
    m.faces_buffer().emplace_back(a, b, b + 3);
    m.faces_buffer().emplace_back(a, b + 3, a + 3);
  }
  return m;
}

} // namespace

TEST_CASE("face_regions: hole stays structure, its interior is a region",
          "[cut][face_regions]") {
  auto sheet = make_sheet();
  auto prism = make_prism();
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
  tf::face_cuts<Index, Int> fc;
  fc.build(ig, apply_to_face, get_mesh_point);

  auto ipts = ig.points();
  auto get_point = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    return v.source == vsource::created ? ipts[v.id]
                                        : get_mesh_point(tag, v.id);
  };
  auto face0 = forms[0].faces()[0];
  auto axes = tf::exact::projection_axes(
      get_mesh_point(0, Index(face0[0])), get_mesh_point(0, Index(face0[1])),
      get_mesh_point(0, Index(face0[2])));
  auto abs_area2 = [&](Index tag, const auto &walk) -> T2 {
    using T1 = tf::exact::meta<Int>::T1;
    T2 a(0);
    const auto m = walk.size();
    for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
      auto p = get_point(tag, walk[j]);
      auto q = get_point(tag, walk[i]);
      a += T2(T1(p[axes.first]) * T1(q[axes.second]) -
              T1(q[axes.first]) * T1(p[axes.second]));
    }
    return a < T2(0) ? -a : a;
  };

  auto fr_loops = fr.loops();
  auto fr_holes = fr.holes();
  auto fr_lh = fr.loop_holes();
  auto descs = fr.descriptors();
  REQUIRE(fr_loops.size() == fr_lh.size());
  REQUIRE(std::size_t(fr.descriptors().size()) == std::size_t(fr_loops.size()));

  // the structure duality: exactly one hole, on a tag-0 loop; region
  // counts match face_cuts per descriptor
  REQUIRE(fr_holes.size() == 1);
  std::map<std::pair<Index, Index>, int> fr_count, fc_count;
  for (Index l = 0; l < Index(fr_loops.size()); ++l)
    ++fr_count[{descs[l].tag, descs[l].object}];
  for (Index l = 0; l < Index(fc.loops().size()); ++l)
    ++fc_count[{fc.descriptors()[l].tag, fc.descriptors()[l].object}];
  REQUIRE(fr_count == fc_count);
  REQUIRE(fr_count[{0, 0}] == 2);

  Index ring = -1, inner = -1;
  for (Index l = 0; l < Index(fr_loops.size()); ++l) {
    if (descs[l].tag != 0)
      continue;
    if (fr_lh[l].size() == 1)
      ring = l;
    else if (fr_lh[l].size() == 0)
      inner = l;
  }
  REQUIRE(ring >= 0);
  REQUIRE(inner >= 0);

  // the hole's walk and the interior piece are the same identities
  auto ids_of = [](const auto &walk) {
    std::vector<std::array<Index, 2>> ids;
    for (const auto &v : walk)
      ids.push_back({Index(v.source), v.id});
    std::sort(ids.begin(), ids.end());
    return ids;
  };
  auto hole_walk = fr_holes[fr_lh[ring][0]];
  REQUIRE(ids_of(hole_walk) == ids_of(fr_loops[inner]));

  // conservation, exact: (ring boundary - hole) + interior == face
  vertex_t c0{vsource::original, Index(face0[0]), {0, tf::topo_type::vertex}};
  vertex_t c1{vsource::original, Index(face0[1]), {1, tf::topo_type::vertex}};
  vertex_t c2{vsource::original, Index(face0[2]), {2, tf::topo_type::vertex}};
  std::array<vertex_t, 3> base{c0, c1, c2};
  auto face_area = abs_area2(0, base);
  auto ring_area = abs_area2(0, fr_loops[ring]) - abs_area2(0, hole_walk);
  REQUIRE(ring_area + abs_area2(0, fr_loops[inner]) == face_area);

  // fast-path faces (the prism sides) carry no holes
  for (Index l = 0; l < Index(fr_loops.size()); ++l)
    if (descs[l].tag == 1)
      REQUIRE(fr_lh[l].size() == 0);

  // nothing deleted on either tag
  REQUIRE(fr.deleted(0).size() == 0);
  REQUIRE(fr.deleted(1).size() == 0);
}

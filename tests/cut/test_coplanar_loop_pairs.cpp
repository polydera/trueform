/**
 * @file test_coplanar_loop_pairs.cpp
 * @brief tf::cut::make_coplanar_loop_pairs — standalone coincident-loop
 *        detection agrees with cut_graph, and the fold map folds every
 *        dead loop onto a live survivor with the right winding flag.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>
#include <trueform/cut/dispatch/build_exact_pipeline.hpp>
#include <trueform/cut/dispatch/build_self_pipeline.hpp>
#include <trueform/cut/make_coplanar_loop_pairs.hpp>
#include <trueform/trueform.hpp>

#include <map>
#include <set>
#include <tuple>

namespace {

using Index = int;
using Real = double;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

auto box_and_sheets() -> std::array<mesh_t, 3> {
  mesh_t box = tf::triangulated(
      tf::make_box_mesh<Index, Real>(Real(2), Real(2), Real(2)).polygons());
  tf::ensure_positive_orientation(box.polygons());
  mesh_t plane = tf::triangulated(
      tf::make_plane_mesh<Index, Real>(Real(4), Real(4)).polygons());
  mesh_t plane_rev = plane;
  for (auto &&face : plane_rev.faces_buffer())
    std::swap(face[0], face[2]);
  return {std::move(box), std::move(plane), std::move(plane_rev)};
}

template <typename Pairs>
auto canonical(const Pairs &pairs) {
  std::set<std::tuple<Index, Index, bool>> s;
  for (const auto &p : pairs) {
    const Index a = std::min(p.loop_a, p.loop_b);
    const Index b = std::max(p.loop_a, p.loop_b);
    s.insert({a, b, p.opposing});
  }
  return s;
}

} // namespace

TEST_CASE("coplanar loop pairs: self mode matches cut_graph on a stacked "
          "soup",
          "[cut][coplanar]") {
  auto [box, plane, plane_rev] = box_and_sheets();
  auto a = tf::concatenated(box.polygons(), plane.polygons());
  auto b = tf::concatenated(a.polygons(), plane_rev.polygons());
  auto c = tf::concatenated(b.polygons(), plane.polygons());
  mesh_t soup;
  soup.faces_buffer() = c.faces_buffer();
  soup.points_buffer() = c.points_buffer();

  auto p = soup.polygons();
  tf::aabb_tree<Index, Real, 3> tree(p, tf::config_tree(4, 4));
  auto fm = tf::make_face_membership(p);
  auto mel = tf::make_manifold_edge_link(p);
  auto tagged = p | tf::tag(tree) | tf::tag(fm) | tf::tag(mel);

  auto [iwp, ig, fc, cg] =
      tf::cut::dispatch::build_self_pipeline<Index, Real, tf::exact::int64>(
          tagged, tf::intersect_config{
                      tf::intersect_mode::primitives |
                      tf::intersect_mode::resolve_crossing_contours});

  auto pairs = tf::cut::make_coplanar_loop_pairs_self(fc);
  REQUIRE(pairs.size() > 0);
  REQUIRE(canonical(pairs) == canonical(cg.coplanar_pairs()));
  bool any_opposing = false;
  for (const auto &pr : pairs)
    any_opposing = any_opposing || pr.opposing;
  REQUIRE(any_opposing);

  auto fold_of = tf::cut::make_loop_fold_map(pairs, fc.loops().size());
  std::set<Index> dead;
  for (const auto &pr : pairs)
    dead.insert(pr.loop_b);
  Index n_folds = 0;
  for (std::size_t l = 0; l < fold_of.size(); ++l) {
    if (fold_of[l][0] == Index(-1))
      continue;
    ++n_folds;
    // every fold's survivor is live and smaller
    REQUIRE(dead.count(fold_of[l][0]) == 0);
    REQUIRE(fold_of[l][0] < Index(l));
  }
  REQUIRE(n_folds == Index(dead.size()));
}

TEST_CASE("coplanar loop pairs: between mode matches cut_graph, reversed "
          "pair is opposing",
          "[cut][coplanar]") {
  // Three forms: the box cuts both coincident sheets cross-form; the
  // sheets' loops coincide across tags 1 and 2 with opposite windings.
  auto [box, plane, plane_rev] = box_and_sheets();
  std::array<mesh_t *, 3> meshes{&box, &plane, &plane_rev};

  std::array<tf::aabb_tree<Index, Real, 3>, 3> trees;
  std::array<tf::face_membership<Index>, 3> fms;
  std::array<tf::manifold_edge_link<Index, 3>, 3> mels;
  for (int i = 0; i < 3; ++i) {
    trees[i] =
        tf::aabb_tree<Index, Real, 3>(meshes[i]->polygons(),
                                      tf::config_tree(4, 4));
    fms[i].build(meshes[i]->polygons());
    mels[i] = tf::make_manifold_edge_link(meshes[i]->polygons());
  }
  auto forms = tf::make_mapped_range(tf::make_sequence_range(3), [&](int i) {
    return meshes[i]->polygons() | tf::tag(trees[i]) | tf::tag(fms[i]) |
           tf::tag(mels[i]);
  });

  tf::intersections_between_polygons<Index, Real, tf::exact::int64> ibp;
  ibp.build(forms, tf::intersect_mode::primitives |
                       tf::intersect_mode::resolve_crossing_contours);
  auto &conv = ibp.converter();
  auto apply_to_face = [&](int tag, Index object, const auto &f) {
    f(forms[tag].faces()[object]);
  };
  auto get_mesh_point = [&](int tag, Index id) -> tf::point<tf::exact::int64, 3> {
    return conv.convert(forms[tag].points()[id]);
  };
  tf::intersection_graph<Index, tf::exact::int64> ig;
  ig.build(ibp, apply_to_face, get_mesh_point,
           tf::intersect_mode::primitives |
               tf::intersect_mode::resolve_crossing_contours,
           tf::exact::make_kernel(conv, 0.0));
  tf::face_cuts<Index, tf::exact::int64> fc;
  fc.build(ig, apply_to_face, get_mesh_point);
  tf::cut_graph<Index> cg;
  cg.build(fc, static_cast<Index>(ig.points().size()));

  auto pairs = tf::cut::make_coplanar_loop_pairs(fc);
  REQUIRE(pairs.size() > 0);
  REQUIRE(canonical(pairs) == canonical(cg.coplanar_pairs()));
  for (const auto &pr : pairs)
    REQUIRE(pr.opposing);
}

TEST_CASE("fold emit: reversed coincident stack triangulates identically",
          "[cut][coplanar]") {
  // The mesh path re-emits a dead loop from its survivor's
  // triangulation: the two walls must come out coordinate-identical
  // with opposite windings.
  auto [box, plane, plane_rev] = box_and_sheets();
  std::vector<decltype(box.polygons())> forms{
      box.polygons(), plane.polygons(), plane_rev.polygons()};
  auto arr = tf::make_mesh_arrangements(tf::make_range(forms));
  auto &mesh = std::get<0>(arr);
  auto &tags = std::get<1>(arr);

  using key_t = std::array<double, 9>;
  auto pts = mesh.polygons().points();
  std::map<int, std::map<key_t, double>> walls; // tag -> canon tri -> nz
  for (std::size_t f = 0; f < mesh.faces().size(); ++f) {
    auto face = mesh.faces()[f];
    bool wall = true;
    for (int k = 0; k < 3; ++k)
      wall = wall && std::abs(double(pts[face[k]][2])) < 1e-12;
    if (!wall)
      continue;
    std::array<std::array<double, 3>, 3> v;
    for (int k = 0; k < 3; ++k)
      for (int d = 0; d < 3; ++d)
        v[k][d] = double(pts[face[k]][d]);
    const double nz = (v[1][0] - v[0][0]) * (v[2][1] - v[0][1]) -
                      (v[1][1] - v[0][1]) * (v[2][0] - v[0][0]);
    std::sort(v.begin(), v.end());
    key_t key;
    int i = 0;
    for (auto &q : v)
      for (int d = 0; d < 3; ++d)
        key[i++] = q[d];
    walls[int(tags[f])][key] = nz;
  }
  auto &w1 = walls[1];
  auto &w2 = walls[2];
  REQUIRE(w1.size() > 0);
  REQUIRE(w1.size() == w2.size());
  for (const auto &[key, nz] : w2) {
    auto it = w1.find(key);
    REQUIRE(it != w1.end());
    REQUIRE(it->second * nz < 0.0); // opposite winding
  }
}

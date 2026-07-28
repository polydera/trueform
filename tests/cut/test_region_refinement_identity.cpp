/**
 * @file test_region_refinement_identity.cpp
 * @brief What "this region is already refined" means to
 *        tf::cut::region_triangulator::refine.
 *
 * The whole request is the identity, not its quality alone.
 * tf::cdt_refine_config::split_encroached decides whether boundary
 * constraints split at all, so a run that turns it on asks a different
 * question of every region than the run before it did at the same quality,
 * and must do work. The skip itself still has to hold, in both directions:
 * a request already answered stays a no-op, and so does a strictly weaker
 * one.
 *
 * Copyright (c) 2026 Ziga Sajovic, XLAB
 */

#include <catch2/catch_test_macros.hpp>

#include <trueform/cut/face_regions.hpp>
#include <trueform/cut/impl/region_triangulator.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/intersect/exact/make_kernel.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/topology/cdt_refine_config.hpp>
#include <trueform/trueform.hpp>

#include <array>
#include <cstddef>

namespace {

using Index = int;
using Real = double;
using Int = tf::exact::int64;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;

} // namespace

TEST_CASE("region_triangulator: split_encroached is part of the refined "
          "identity, so asking for it after a run without it is work",
          "[cut][region_triangulator][refined]") {
  // An ACUTE sheet is what makes the two answers separable: near an acute
  // corner no boundary edge's diametral circle reaches, so a frozen
  // boundary still admits interior insertions, while the rest of the face
  // stays pinned behind constraints only a release can split. Over a
  // right-angled face every candidate encroaches the hypotenuse, the
  // frozen pass is empty, and it records no fingerprint to be consulted.
  mesh_t sheet;
  sheet.points_buffer().emplace_back(Real(0), Real(0), Real(0));
  sheet.points_buffer().emplace_back(Real(100), Real(0), Real(0));
  sheet.points_buffer().emplace_back(Real(50), Real(86.60254037844386),
                                     Real(0));
  sheet.faces_buffer().emplace_back(0, 1, 2);
  mesh_t prism;
  {
    const std::array<std::array<Real, 2>, 3> cs = {
        {{30.0, 20.0}, {70.0, 20.0}, {30.0, 60.0}}};
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
  const std::size_t plain_points = rt.extra_points.size();

  tf::cdt_refine_config frozen;
  frozen.min_quality = 0.3f;
  frozen.split_encroached = false;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, frozen);
  REQUIRE(rt.n_failed == 0);
  const std::size_t frozen_points = rt.extra_points.size();
  const std::size_t frozen_tris = rt.tris.size();
  INFO("frozen: " << frozen_tris << " tris, " << frozen_points << " points");
  // the frozen pass must have answered something, or it records no
  // fingerprint and the identity below is never consulted
  REQUIRE(frozen_points > plain_points);

  // the same quality with the boundary released is a different question of
  // every region, and the fingerprint may not answer it
  tf::cdt_refine_config released = frozen;
  released.split_encroached = true;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, released);
  REQUIRE(rt.n_failed == 0);
  INFO("released: " << rt.tris.size() << " tris, " << rt.extra_points.size()
                    << " points");
  REQUIRE(rt.extra_points.size() > frozen_points);
  REQUIRE(rt.tris.size() > frozen_tris);

  // and the skip still holds: the request just answered is a no-op
  const std::size_t settled_points = rt.extra_points.size();
  const std::size_t settled_tris = rt.tris.size();
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, released);
  REQUIRE(rt.n_failed == 0);
  REQUIRE(rt.extra_points.size() == settled_points);
  REQUIRE(rt.tris.size() == settled_tris);

  // the identity is an ordering, not an equality: freezing the boundary
  // again asks strictly less than what the store already holds
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, frozen);
  REQUIRE(rt.n_failed == 0);
  REQUIRE(rt.extra_points.size() == settled_points);
  REQUIRE(rt.tris.size() == settled_tris);
}

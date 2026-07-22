#include <catch2/catch_test_macros.hpp>

#include <trueform/cut/face_regions.hpp>
#include <trueform/cut/make_coplanar_loop_pairs.hpp>
#include <trueform/cut/region_triangulator.hpp>
#include <trueform/exact/meta.hpp>
#include <trueform/exact/projection_axes.hpp>
#include <trueform/intersect/exact/make_kernel.hpp>
#include <trueform/intersect/graph/intersection_graph.hpp>
#include <trueform/intersect/intersections_between_polygons.hpp>
#include <trueform/trueform.hpp>

#include <algorithm>
#include <cmath>
#include <array>
#include <set>
#include <utility>

namespace {

using Index = int;
using Real = double;
using Int = tf::exact::int64;
using mesh_t = tf::polygons_buffer<Index, Real, 3, 3>;
using vertex_t = tf::intersect::graph::vertex<Index>;
using vsource = tf::intersect::graph::vertex_source;
using T1 = tf::exact::meta<Int>::T1;
using T2 = tf::exact::meta<Int>::T2;

auto make_sheet() -> mesh_t {
  mesh_t m;
  m.points_buffer().emplace_back(0.f, 0.f, 0.f);
  m.points_buffer().emplace_back(10.f, 0.f, 0.f);
  m.points_buffer().emplace_back(0.f, 10.f, 0.f);
  m.faces_buffer().emplace_back(0, 1, 2);
  return m;
}

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

/// The triangle-grain exposure against its spec: per tag the
/// structure's loops in loop order then the tag's promoted faces, each
/// loop's triangles in range order under its OWN descriptor with the
/// rev corner swap; face_offsets and tag_offsets partition the stream;
/// deleted mirrors the structure's and is disjoint from the exposure.
void require_exposure(const tf::face_regions<Index, Int> &fr,
                      const tf::cut::region_triangulator<Index, Int> &rt) {
  auto descs = fr.descriptors();
  auto promoted = rt.promoted_descriptors();
  const std::size_t n_struct = descs.size();
  const Index n_tags = Index(fr.tag_offsets().size()) - 1;
  auto exposed = rt.loops();
  auto edescs = rt.descriptors();
  auto toffs = rt.tag_offsets();
  auto foffs = rt.face_offsets();

  std::size_t expect = 0;
  for (std::size_t li = 0; li < std::size_t(rt.ranges.size()); ++li)
    expect += std::size_t(rt.ranges[li][1] - rt.ranges[li][0]);
  REQUIRE(exposed.size() == expect);
  REQUIRE(edescs.size() == expect);

  auto vid = [](const vertex_t &v) {
    return std::pair<Index, Index>{Index(v.source), v.id};
  };
  std::size_t o = 0, pk = 0;
  auto check_loop = [&](std::size_t li, Index tag, Index object) {
    const auto r = rt.ranges[li];
    const bool rv = li < rt.rev.size() && bool(rt.rev[li]);
    for (Index k = r[0]; k < r[1]; ++k) {
      const auto &tr = rt.tris[std::size_t(k)];
      const auto &et = exposed[o];
      REQUIRE(vid(et[0]) == vid(tr[0]));
      REQUIRE(vid(et[1]) == vid(tr[rv ? 2 : 1]));
      REQUIRE(vid(et[2]) == vid(tr[rv ? 1 : 2]));
      REQUIRE(edescs[o].tag == tag);
      REQUIRE(edescs[o].object == object);
      ++o;
    }
  };
  for (Index t = 0; t < n_tags; ++t) {
    REQUIRE(std::size_t(toffs[t]) == o);
    for (Index li = fr.tag_offsets()[t]; li < fr.tag_offsets()[t + 1]; ++li)
      check_loop(std::size_t(li), descs[li].tag, descs[li].object);
    while (pk < promoted.size() && promoted[pk].tag == t) {
      check_loop(n_struct + pk, promoted[pk].tag, promoted[pk].object);
      ++pk;
    }
  }
  REQUIRE(o == exposed.size());
  REQUIRE(pk == promoted.size());
  if (n_tags > 0)
    REQUIRE(std::size_t(toffs[n_tags]) == exposed.size());

  REQUIRE(foffs.size() >= 1);
  REQUIRE(foffs[0] == 0);
  REQUIRE(std::size_t(foffs[foffs.size() - 1]) == exposed.size());
  for (std::size_t f = 0; f + 1 < foffs.size(); ++f) {
    const auto b = std::size_t(foffs[f]);
    const auto e = std::size_t(foffs[f + 1]);
    REQUIRE(b < e);
    for (std::size_t i = b; i < e; ++i) {
      REQUIRE(edescs[i].tag == edescs[b].tag);
      REQUIRE(edescs[i].object == edescs[b].object);
    }
  }

  REQUIRE(rt.deleted_offsets().size() == fr.deleted_offsets().size());
  std::set<std::pair<Index, Index>> exposed_faces;
  for (std::size_t i = 0; i < edescs.size(); ++i)
    exposed_faces.insert({edescs[i].tag, edescs[i].object});
  for (Index t = 0; t < n_tags; ++t) {
    REQUIRE(rt.deleted(t).size() == fr.deleted(t).size());
    for (std::size_t i = 0; i < rt.deleted(t).size(); ++i) {
      REQUIRE(rt.deleted(t)[i] == fr.deleted(t)[i]);
      REQUIRE(exposed_faces.count({t, rt.deleted(t)[i]}) == 0);
    }
  }
}

} // namespace

TEST_CASE("region_triangulator: exact cover of every region, holes "
          "respected, winding follows the loop",
          "[cut][region_triangulator]") {
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
  tf::cut::region_triangulator<Index, Int> rt;
  rt.build(fr, ig, forms, apply_to_face, get_mesh_point);

  REQUIRE(rt.n_failed == 0);
  REQUIRE(rt.failed.size() == 0);
  REQUIRE(std::size_t(rt.ranges.size()) ==
          std::size_t(fr.loops().size()) + rt.promoted_descriptors().size());

  auto ipts = ig.points();
  auto get_point = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    return v.source == vsource::created ? ipts[v.id]
                                        : get_mesh_point(tag, v.id);
  };
  auto descs = fr.descriptors();
  auto loops = fr.loops();
  auto holes = fr.holes();
  auto loop_holes = fr.loop_holes();

  for (Index l = 0; l < Index(loops.size()); ++l) {
    const Index tag = descs[l].tag;
    auto face = forms[tag].faces()[descs[l].object];
    auto axes = tf::exact::projection_axes(
        get_mesh_point(tag, Index(face[0])),
        get_mesh_point(tag, Index(face[1])),
        get_mesh_point(tag, Index(face[2])));
    auto signed2 = [&](const auto &walk) -> T2 {
      T2 a(0);
      const auto m = walk.size();
      for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
        auto p = get_point(tag, walk[j]);
        auto q = get_point(tag, walk[i]);
        a += T2(T1(p[axes.first]) * T1(q[axes.second]) -
                T1(q[axes.first]) * T1(p[axes.second]));
      }
      return a;
    };
    auto abs2 = [](T2 v) { return v < T2(0) ? -v : v; };

    // region area = |boundary| - sum |holes|; triangles cover it
    // exactly, oriented with the boundary walk
    T2 region = abs2(signed2(loops[l]));
    for (auto h : loop_holes[l])
      region = region - abs2(signed2(holes[h]));
    T2 tri_signed(0);
    for (auto &&t : rt.loop_tris(l)) {
      auto pa = get_point(tag, t[0]);
      auto pb = get_point(tag, t[1]);
      auto pc = get_point(tag, t[2]);
      T1 abx = T1(pb[axes.first]) - T1(pa[axes.first]);
      T1 aby = T1(pb[axes.second]) - T1(pa[axes.second]);
      T1 acx = T1(pc[axes.first]) - T1(pa[axes.first]);
      T1 acy = T1(pc[axes.second]) - T1(pa[axes.second]);
      tri_signed += T2(abx) * acy - T2(aby) * acx;
    }
    REQUIRE(abs2(tri_signed) == region);
    if (!(region == T2(0))) {
      const bool loop_ccw = T2(0) < signed2(loops[l]);
      const bool tris_ccw = T2(0) < tri_signed;
      REQUIRE(loop_ccw == tris_ccw);
    }
  }
}


TEST_CASE("region_triangulator: triangle-grain exposure mirrors the store",
          "[cut][region_triangulator]") {
  // the promotion fixture: promoted faces force the owned recopy path
  mesh_t sheet;
  sheet.points_buffer().emplace_back(0.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(10.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(0.f, 10.f, 0.f);
  sheet.points_buffer().emplace_back(10.f, 10.f, 0.f);
  sheet.faces_buffer().emplace_back(0, 1, 2);
  sheet.faces_buffer().emplace_back(1, 3, 2);
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
  tf::cut::region_triangulator<Index, Int> rt;
  rt.build(fr, ig, forms, apply_to_face, get_mesh_point);

  require_exposure(fr, rt);
  // nothing forces a recopy here: the exposure is a view over the store
  REQUIRE(rt.promoted_descriptors().size() == 0);
  REQUIRE(&rt.loops()[0] == &rt.tris[0]);

  // coincident opposing sheets: the dead member's triangles materialize
  // under its own descriptor with the rev corner swap — the recopy path
  {
    mesh_t a, b;
    a.points_buffer().emplace_back(0.f, 0.f, 0.f);
    a.points_buffer().emplace_back(10.f, 0.f, 0.f);
    a.points_buffer().emplace_back(0.f, 10.f, 0.f);
    a.faces_buffer().emplace_back(0, 1, 2);
    b.points_buffer().emplace_back(0.f, 0.f, 0.f);
    b.points_buffer().emplace_back(10.f, 0.f, 0.f);
    b.points_buffer().emplace_back(0.f, 10.f, 0.f);
    b.faces_buffer().emplace_back(0, 2, 1);
    std::array<mesh_t *, 2> meshes2{&a, &b};
    std::array<tf::aabb_tree<Index, Real, 3>, 2> trees2;
    std::array<tf::face_membership<Index>, 2> fms2;
    std::array<tf::manifold_edge_link<Index, 3>, 2> mels2;
    for (int i = 0; i < 2; ++i) {
      trees2[i] = tf::aabb_tree<Index, Real, 3>(meshes2[i]->polygons(),
                                                tf::config_tree(4, 4));
      fms2[i].build(meshes2[i]->polygons());
      mels2[i] = tf::make_manifold_edge_link(meshes2[i]->polygons());
    }
    auto forms2 =
        tf::make_mapped_range(tf::make_sequence_range(2), [&](int i) {
          return meshes2[i]->polygons() | tf::tag(trees2[i]) |
                 tf::tag(fms2[i]) | tf::tag(mels2[i]);
        });
    tf::intersections_between_polygons<Index, Real, Int> ibp2;
    ibp2.build(forms2, mode);
    auto &conv2 = ibp2.converter();
    auto apply_to_face2 = [&](int tag, Index object, const auto &f) {
      f(forms2[tag].faces()[object]);
    };
    auto get_mesh_point2 = [&](int tag, Index id) -> tf::point<Int, 3> {
      return conv2.convert(forms2[tag].points()[id]);
    };
    tf::intersection_graph<Index, Int> ig2;
    ig2.build(ibp2, apply_to_face2, get_mesh_point2, mode,
              tf::exact::make_kernel(conv2, 0.0));
    tf::face_regions<Index, Int> fr2;
    fr2.build(ig2, apply_to_face2, get_mesh_point2);

    auto pairs = tf::cut::make_coplanar_loop_pairs_all(fr2);
    REQUIRE(pairs.size() == 1);
    tf::buffer<char> dead;
    dead.allocate(fr2.loops().size());
    std::fill(dead.begin(), dead.end(), char(0));
    tf::buffer<std::array<Index, 3>> cpairs;
    for (const auto &p : pairs) {
      cpairs.push_back({p.loop_a, p.loop_b, Index(p.opposing)});
      dead[std::size_t(p.loop_b)] = char(1);
    }
    tf::cut::region_triangulator<Index, Int> rt2;
    rt2.build(fr2, ig2, forms2, apply_to_face2, get_mesh_point2,
              tf::make_range(dead), tf::make_range(cpairs));

    require_exposure(fr2, rt2);
    REQUIRE(bool(rt2.rev[std::size_t(cpairs[0][1])]) == bool(cpairs[0][2]));
    REQUIRE(rt2.loops().size() != 0);
    REQUIRE(&rt2.loops()[0] != &rt2.tris[0]);
  }
}


TEST_CASE("region_triangulator: refinement rides the same machinery",
          "[cut][region_triangulator]") {
  // a sheet LARGE against the cut: the tiny seam's spacing gradient
  // cannot be satisfied by boundary subdivision alone, so pass-1 must
  // insert interior points (at cut scale ~sheet/5 boundary splitting
  // suffices and no interiors ever appear — measured)
  mesh_t sheet;
  sheet.points_buffer().emplace_back(0.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(100.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(0.f, 100.f, 0.f);
  sheet.faces_buffer().emplace_back(0, 1, 2);
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
  const std::size_t plain_tris = rt.tris.size();
  const std::size_t plain_pts = rt.extra_points.size();

  tf::cdt_refine_config rc;
  // near the Delaunay-refinement limit: boundary splitting alone
  // cannot satisfy this, so pass-1 interiors (and their survival
  // through the pass-2 injection) are forced
  rc.min_quality = 0.48f;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);

  // refinement adds points and triangles, and keeps the surface exact
  REQUIRE(rt.n_failed == 0);
  REQUIRE(rt.extra_points.size() > plain_pts);
  REQUIRE(rt.tris.size() > plain_tris);

  // the refiner's INTERIOR insertions survive: they are created
  // vertices carrying the face marker, distinct from boundary splits
  std::size_t interior = 0, boundary_splits = 0;
  std::set<std::pair<Index, Index>> seen;
  for (auto &&t : rt.tris)
    for (int c = 0; c < 3; ++c) {
      const auto &v = t[c];
      if (v.source != vsource::created || v.id < rt.n_ig_created)
        continue;
      if (!seen.insert({Index(v.sub_id.label), v.id}).second)
        continue;
      if (v.sub_id.label == tf::topo_type::face)
        ++interior;
      else
        ++boundary_splits;
    }
  REQUIRE(interior > 0);
  REQUIRE(boundary_splits > 0);

  auto ipts = ig.points();
  auto pt3 = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    if (v.source == vsource::created)
      return v.id >= rt.n_ig_created
                 ? rt.extra_points[std::size_t(v.id - rt.n_ig_created)]
                 : ipts[v.id];
    return get_mesh_point(tag, v.id);
  };
  auto descs = fr.descriptors();
  auto loops = fr.loops();
  auto holes = fr.holes();
  auto loop_holes = fr.loop_holes();
  for (Index l = 0; l < Index(loops.size()); ++l) {
    const Index tag = descs[l].tag;
    auto face = forms[tag].faces()[descs[l].object];
    auto axes = tf::exact::projection_axes(
        get_mesh_point(tag, Index(face[0])),
        get_mesh_point(tag, Index(face[1])),
        get_mesh_point(tag, Index(face[2])));
    auto abs2 = [](T2 v) { return v < T2(0) ? -v : v; };
    auto walk2 = [&](const auto &w) -> T2 {
      T2 a(0);
      const auto m = w.size();
      for (std::size_t i = 0, j = m - 1; i < m; j = i++) {
        auto p = pt3(tag, w[j]);
        auto q = pt3(tag, w[i]);
        a += T2(T1(p[axes.first]) * T1(q[axes.second]) -
                T1(q[axes.first]) * T1(p[axes.second]));
      }
      return a;
    };
    T2 region = abs2(walk2(loops[l]));
    for (auto h : loop_holes[l])
      region = region - abs2(walk2(holes[h]));
    // signed cover: a single mirrored triangle cancels instead of
    // adding, so |Σ signed| == region is winding-sensitive
    T2 sum(0);
    for (auto &&t : rt.loop_tris(l)) {
      auto pa = pt3(tag, t[0]);
      auto pb = pt3(tag, t[1]);
      auto pc = pt3(tag, t[2]);
      T1 abx = T1(pb[axes.first]) - T1(pa[axes.first]);
      T1 aby = T1(pb[axes.second]) - T1(pa[axes.second]);
      T1 acx = T1(pc[axes.first]) - T1(pa[axes.first]);
      T1 acy = T1(pc[axes.second]) - T1(pa[axes.second]);
      sum += T2(abx) * acy - T2(aby) * acx;
    }
    REQUIRE(abs2(sum) == region);
  }
}

TEST_CASE("region_triangulator: refined proposals on sub-edges re-key "
          "onto the parent and conform across both carriers",
          "[cut][region_triangulator][refined]") {
  // Two-pass refine: pass 1 lands quality splits on the sheet's
  // diagonal, so pass 2 walks that edge as sub-edges — its proposals
  // must re-key onto the parent edge, land on the diagonal, and be
  // consumed identically by both incident faces.
  mesh_t sheet;
  sheet.points_buffer().emplace_back(0.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(10.f, 0.f, 0.f);
  sheet.points_buffer().emplace_back(0.f, 10.f, 0.f);
  sheet.points_buffer().emplace_back(10.f, 10.f, 0.f);
  sheet.faces_buffer().emplace_back(0, 1, 2);
  sheet.faces_buffer().emplace_back(1, 3, 2);
  mesh_t prism;
  {
    const std::array<std::array<Real, 2>, 3> cs = {
        {{2.0, 2.0}, {7.7, 2.0}, {2.0, 7.7}}};
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

  const auto da = get_mesh_point(0, 1);
  const auto db = get_mesh_point(0, 2);
  auto on_diagonal = [&](const tf::point<Int, 3> &p) -> bool {
    std::array<T1, 3> ab, ap;
    for (int c = 0; c < 3; ++c) {
      ab[std::size_t(c)] = T1(db[c]) - T1(da[c]);
      ap[std::size_t(c)] = T1(p[c]) - T1(da[c]);
    }
    for (int c = 0; c < 3; ++c) {
      const int u = (c + 1) % 3, v = (c + 2) % 3;
      if (T2(ab[std::size_t(u)]) * ap[std::size_t(v)] !=
          T2(ab[std::size_t(v)]) * ap[std::size_t(u)])
        return false;
    }
    T2 dot(0), len2(0);
    for (int c = 0; c < 3; ++c) {
      dot += T2(ab[std::size_t(c)]) * ap[std::size_t(c)];
      len2 += T2(ab[std::size_t(c)]) * ab[std::size_t(c)];
    }
    return !(dot < T2(0)) && !(len2 < dot);
  };
  auto count_on_diagonal = [&]() -> std::size_t {
    std::size_t n = 0;
    for (const auto &p : rt.extra_points)
      if (on_diagonal(p))
        ++n;
    return n;
  };

  tf::cdt_refine_config rc;
  rc.min_quality = 0.2f;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
  REQUIRE(rt.n_failed == 0);
  const std::size_t n_pass1 = count_on_diagonal();
  const std::size_t n_extra1 = rt.extra_points.size();
  CAPTURE(n_pass1, n_extra1);
  REQUIRE(n_pass1 > 0);

  // pass 2 walks the diagonal as sub-edges: only re-keyed proposals
  // can add points to it
  rc.min_quality = 0.48f;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
  REQUIRE(rt.n_failed == 0);
  const std::size_t n_pass2 = count_on_diagonal();
  const std::size_t n_extra2 = rt.extra_points.size();
  CAPTURE(n_pass2, n_extra2);
  REQUIRE(n_pass2 > n_pass1);

  // both carriers consume the same diagonal identities — one-sided
  // consumption is a T-junction
  auto ipts = ig.points();
  auto pt3 = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    if (v.source == vsource::created)
      return v.id >= rt.n_ig_created
                 ? rt.extra_points[std::size_t(v.id - rt.n_ig_created)]
                 : ipts[v.id];
    return get_mesh_point(tag, v.id);
  };
  auto descs = fr.descriptors();
  auto promoted = rt.promoted_descriptors();
  auto diagonal_ids_of_face =
      [&](Index object) -> std::set<std::array<Index, 2>> {
    std::set<std::array<Index, 2>> ids;
    auto collect = [&](Index li, Index tag, Index obj) {
      if (tag != Index(0) || obj != object)
        return;
      for (auto &&t : rt.loop_tris(li))
        for (int c = 0; c < 3; ++c) {
          const auto &v = t[std::size_t(c)];
          if (on_diagonal(pt3(tag, v)))
            ids.insert({v.source == vsource::original ? Index(0) : Index(2),
                        v.id});
        }
    };
    for (Index li = 0; li < Index(descs.size()); ++li)
      collect(li, descs[li].tag, descs[li].object);
    for (Index p = 0; p < Index(promoted.size()); ++p)
      collect(Index(descs.size()) + p, promoted[p].tag, promoted[p].object);
    return ids;
  };
  auto side0 = diagonal_ids_of_face(0);
  auto side1 = diagonal_ids_of_face(1);
  REQUIRE(side0.size() > 2);
  REQUIRE(side0 == side1);

  // no orphans: every materialized point is referenced by a triangle
  // or retired by a weld
  std::set<Index> referenced;
  for (Index li = 0; li < Index(rt.ranges.size()); ++li)
    for (auto &&t : rt.loop_tris(li))
      for (int c = 0; c < 3; ++c) {
        const auto &v = t[std::size_t(c)];
        if (v.source == vsource::created && v.id >= rt.n_ig_created)
          referenced.insert(v.id);
      }
  for (std::size_t i = 0; i < rt.extra_points.size(); ++i) {
    const Index id = rt.n_ig_created + Index(i);
    if (referenced.count(id))
      continue;
    bool retired = false;
    for (const auto &m : rt.merges())
      if (m.from == std::array<Index, 2>{Index(2), id}) {
        retired = true;
        break;
      }
    REQUIRE(retired);
  }
}

TEST_CASE("region_triangulator: slit + promotion stress - three refine "
          "passes keep every shared original edge conforming",
          "[cut][region_triangulator][refined][stress]") {
  // A mostly-uncut tiled sheet; a fin crossing one tile (its splits
  // promote uncut neighbours); a second fin whose bottom edge lies IN
  // the sheet plane (rim slit: a wall walk carries the contact edge
  // twice). Three refine passes force re-splitting of previously
  // split uncut boundaries.
  auto sheet = tf::make_plane_mesh<Index, Real>(Real(6), Real(6), 3, 3);
  auto make_fin = [](Real size, Real y, Real z_off) -> mesh_t {
    auto p = tf::make_plane_mesh<Index, Real>(size, size, 1, 1);
    mesh_t m = tf::triangulated(p.polygons());
    for (auto &&pt : m.points()) {
      const Real py = pt[1];
      pt[1] = y;
      pt[2] = py + z_off;
    }
    return m;
  };
  mesh_t sheet_t = tf::triangulated(sheet.polygons());
  mesh_t finA = make_fin(Real(1.6), Real(0.15), Real(0));   // crosses sheet
  mesh_t finB = make_fin(Real(1.4), Real(-0.3), Real(0.7)); // rim ON sheet
  std::array<mesh_t *, 3> meshes{&sheet_t, &finA, &finB};

  std::array<tf::aabb_tree<Index, Real, 3>, 3> trees;
  std::array<tf::face_membership<Index>, 3> fms;
  std::array<tf::manifold_edge_link<Index, 3>, 3> mels;
  for (int i = 0; i < 3; ++i) {
    trees[i] = tf::aabb_tree<Index, Real, 3>(meshes[i]->polygons(),
                                             tf::config_tree(4, 4));
    fms[i].build(meshes[i]->polygons());
    mels[i] = tf::make_manifold_edge_link(meshes[i]->polygons());
  }
  auto forms = tf::make_mapped_range(tf::make_sequence_range(3), [&](int i) {
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

  auto ipts = ig.points();
  auto pt3 = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    if (v.source == vsource::created)
      return v.id >= rt.n_ig_created
                 ? rt.extra_points[std::size_t(v.id - rt.n_ig_created)]
                 : ipts[v.id];
    return get_mesh_point(tag, v.id);
  };
  auto on_segment = [&](const tf::point<Int, 3> &a, const tf::point<Int, 3> &b,
                        const tf::point<Int, 3> &p) -> bool {
    std::array<T1, 3> ab, ap;
    for (int c = 0; c < 3; ++c) {
      ab[std::size_t(c)] = T1(b[c]) - T1(a[c]);
      ap[std::size_t(c)] = T1(p[c]) - T1(a[c]);
    }
    for (int c = 0; c < 3; ++c) {
      const int u = (c + 1) % 3, v = (c + 2) % 3;
      if (T2(ab[std::size_t(u)]) * ap[std::size_t(v)] !=
          T2(ab[std::size_t(v)]) * ap[std::size_t(u)])
        return false;
    }
    T2 dot(0), len2(0);
    for (int c = 0; c < 3; ++c) {
      dot += T2(ab[std::size_t(c)]) * ap[std::size_t(c)];
      len2 += T2(ab[std::size_t(c)]) * ab[std::size_t(c)];
    }
    return !(dot < T2(0)) && !(len2 < dot);
  };
  auto descs = fr.descriptors();
  // identities a face's stream places on a given original-edge segment
  auto ids_on_edge = [&](Index tag, Index object, const tf::point<Int, 3> &a,
                         const tf::point<Int, 3> &b)
      -> std::set<std::array<Index, 2>> {
    std::set<std::array<Index, 2>> ids;
    auto promoted = rt.promoted_descriptors();
    auto collect = [&](Index li, Index t, Index obj) {
      if (t != tag || obj != object)
        return;
      for (auto &&tr : rt.loop_tris(li))
        for (int c = 0; c < 3; ++c) {
          const auto &v = tr[std::size_t(c)];
          if (on_segment(a, b, pt3(tag, v)))
            ids.insert({v.source == vsource::original ? tag : Index(3), v.id});
        }
    };
    for (Index li = 0; li < Index(descs.size()); ++li)
      collect(li, descs[li].tag, descs[li].object);
    for (Index p = 0; p < Index(promoted.size()); ++p)
      collect(Index(descs.size()) + p, promoted[p].tag, promoted[p].object);
    return ids;
  };
  auto check_sheet_conformity = [&]() {
    // only faces IN the stream (cut or promoted) carry a triangulation
    // to conform; untouched faces are emitted from the original mesh
    std::set<Index> exposed;
    for (Index li = 0; li < Index(descs.size()); ++li)
      if (descs[li].tag == Index(0))
        exposed.insert(descs[li].object);
    {
      auto promoted = rt.promoted_descriptors();
      for (Index p = 0; p < Index(promoted.size()); ++p)
        if (promoted[p].tag == Index(0))
          exposed.insert(promoted[p].object);
    }
    auto faces = forms[0].faces();
    auto mel = mels[0];
    for (Index f = 0; f < Index(faces.size()); ++f) {
      if (!exposed.count(f))
        continue;
      const auto face = faces[f];
      const Index fs = Index(face.size());
      for (Index e = 0; e < fs; ++e) {
        auto m = mel[f][std::size_t(e)];
        if (!m.is_simple() || m.face_peer < f || !exposed.count(m.face_peer))
          continue;
        auto a = get_mesh_point(0, Index(face[std::size_t(e)]));
        auto b = get_mesh_point(0, Index(face[std::size_t((e + 1) % fs)]));
        auto s0 = ids_on_edge(0, f, a, b);
        auto s1 = ids_on_edge(0, m.face_peer, a, b);
        if (s0.empty() && s1.empty())
          continue; // neither side emitted (both faces untouched)
        REQUIRE(s0 == s1);
      }
    }
  };
  auto count_orphans = [&]() -> std::size_t {
    std::set<Index> referenced;
    for (Index li = 0; li < Index(rt.ranges.size()); ++li)
      for (auto &&tr : rt.loop_tris(li))
        for (int c = 0; c < 3; ++c) {
          const auto &v = tr[std::size_t(c)];
          if (v.source == vsource::created && v.id >= rt.n_ig_created)
            referenced.insert(v.id);
        }
    std::size_t orphans = 0;
    for (std::size_t i = 0; i < rt.extra_points.size(); ++i) {
      const Index id = rt.n_ig_created + Index(i);
      if (referenced.count(id))
        continue;
      bool retired = false;
      for (const auto &m : rt.merges())
        if (m.from == std::array<Index, 2>{Index(3), id}) {
          retired = true;
          break;
        }
      if (!retired)
        ++orphans;
    }
    return orphans;
  };

  check_sheet_conformity();
  std::size_t prev_extra = rt.extra_points.size();
  const float qualities[3] = {0.2f, 0.45f, 0.48f};
  for (int pass = 0; pass < 3; ++pass) {
    tf::cdt_refine_config rc;
    rc.min_quality = qualities[pass];
    rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
    REQUIRE(rt.n_failed == 0);
    REQUIRE(rt.extra_points.size() > prev_extra);
    prev_extra = rt.extra_points.size();
    check_sheet_conformity();
    REQUIRE(count_orphans() == 0);
  }
  // idempotence: repeating the last quality must not grow the table
  // (catches steiner re-harvest and retired-steiner re-injection)
  {
    tf::cdt_refine_config rc;
    rc.min_quality = qualities[2];
    const std::size_t stable = rt.extra_points.size();
    rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
    REQUIRE(rt.n_failed == 0);
    REQUIRE(rt.extra_points.size() == stable);
    check_sheet_conformity();
    REQUIRE(count_orphans() == 0);
  }
  REQUIRE(rt.promoted_descriptors().size() > 0);
}

TEST_CASE("region_triangulator: refined quality guard - interiors are "
          "harvested, promoted faces refine, min-angle floors hold",
          "[cut][region_triangulator][refined][quality]") {
  // Pins three regressions: interior steiners deleted at harvest,
  // promoted faces emitted without quality insertion, and the
  // min-angle distribution collapsing unmeasured.
  // tiles LARGE against the fin: the tiny seam's spacing gradient
  // cannot be satisfied by boundary subdivision alone, and the fin
  // straddles a tile boundary so its splits promote the uncut
  // neighbour tile
  auto sheet = tf::make_plane_mesh<Index, Real>(Real(60), Real(60), 3, 3);
  mesh_t sheet_t = tf::triangulated(sheet.polygons());
  mesh_t fin;
  {
    auto p = tf::make_plane_mesh<Index, Real>(Real(1.5), Real(1.5), 1, 1);
    fin = tf::triangulated(p.polygons());
    for (auto &&pt : fin.points()) {
      const Real py = pt[1];
      pt[0] += Real(9);
      pt[1] = Real(0.15);
      pt[2] = py;
    }
  }
  std::array<mesh_t *, 2> meshes{&sheet_t, &fin};
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

  tf::cdt_refine_config rc;
  rc.min_quality = 0.3f;
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
  REQUIRE(rt.n_failed == 0);

  auto ipts = ig.points();
  auto pt3 = [&](Index tag, const vertex_t &v) -> tf::point<Int, 3> {
    if (v.source == vsource::created)
      return v.id >= rt.n_ig_created
                 ? rt.extra_points[std::size_t(v.id - rt.n_ig_created)]
                 : ipts[v.id];
    return get_mesh_point(tag, v.id);
  };
  auto min_angle = [&](Index tag, const std::array<vertex_t, 3> &t) -> double {
    const auto a = pt3(tag, t[0]);
    const auto b = pt3(tag, t[1]);
    const auto c = pt3(tag, t[2]);
    auto len = [](const tf::point<Int, 3> &u, const tf::point<Int, 3> &v) {
      double s = 0;
      for (int k = 0; k < 3; ++k) {
        const double d = double(u[k]) - double(v[k]);
        s += d * d;
      }
      return std::sqrt(s);
    };
    const double la = len(b, c), lb = len(a, c), lc = len(a, b);
    if (la <= 0 || lb <= 0 || lc <= 0)
      return 0;
    auto ang = [](double x, double y, double z) {
      double v = (y * y + z * z - x * x) / (2 * y * z);
      v = v < -1 ? -1 : (v > 1 ? 1 : v);
      return std::acos(v) * 57.29577951308232;
    };
    return std::min({ang(la, lb, lc), ang(lb, la, lc), ang(lc, la, lb)});
  };

  // 1. interior steiners exist: the harvest collected the refiner's
  // insertions (face-marker created vertices in the emitted stream)
  std::size_t interior = 0;
  for (auto &&t : rt.tris)
    for (int c = 0; c < 3; ++c)
      if (t[c].source == vsource::created && t[c].id >= rt.n_ig_created &&
          t[c].sub_id.label == tf::topo_type::face)
        ++interior;
  INFO("interior-marker corners: " << interior);
  REQUIRE(interior > 0);

  // 2. promotion happened and promoted faces carry interior insertions
  auto pdescs = rt.promoted_descriptors();
  REQUIRE(pdescs.size() > 0);
  auto exposed = rt.loops();
  std::size_t prom_interior = 0, prom_tris = 0;
  double prom_ge30 = 0;
  for (Index pi = 0; pi < Index(pdescs.size()); ++pi) {
    const auto r = rt.promoted_range(pi);
    for (Index t = r[0]; t < r[1]; ++t) {
      ++prom_tris;
      if (min_angle(pdescs[pi].tag, exposed[t]) >= 30.0)
        ++prom_ge30;
      for (int c = 0; c < 3; ++c)
        if (exposed[t][c].source == vsource::created &&
            exposed[t][c].id >= rt.n_ig_created &&
            exposed[t][c].sub_id.label == tf::topo_type::face)
          ++prom_interior;
    }
  }
  INFO("promoted: " << pdescs.size() << " faces, " << prom_tris
                    << " tris, interior corners " << prom_interior
                    << ", >=30deg " << prom_ge30 / double(prom_tris));
  REQUIRE(prom_interior > 0);

  // 3. min-angle floors over the refined streams (fixture-measured;
  // floors carry margin)
  auto descs = fr.descriptors();
  std::size_t live_tris = 0, live_ge30 = 0;
  for (std::size_t li = 0; li < std::size_t(fr.loops().size()); ++li) {
    if (descs[Index(li)].tag == Index(-1))
      continue;
    const auto r = rt.ranges[li];
    for (Index t = r[0]; t < r[1]; ++t) {
      ++live_tris;
      if (min_angle(descs[Index(li)].tag, rt.tris[std::size_t(t)]) >= 30.0)
        ++live_ge30;
    }
  }
  INFO("live: " << live_tris << " tris, >=30deg "
                << double(live_ge30) / double(live_tris));
  REQUIRE(live_tris > 0);
  REQUIRE(double(live_ge30) / double(live_tris) > 0.55);
  REQUIRE(prom_ge30 / double(prom_tris) > 0.45);

  // 4. the fixed point: a same-quality repeat adds nothing
  const std::size_t stable = rt.extra_points.size();
  rt.refine(fr, ig, forms, apply_to_face, get_mesh_point, rc);
  REQUIRE(rt.extra_points.size() == stable);
}

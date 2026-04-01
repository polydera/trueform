/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once
#include "../core/algorithm/circular_increment.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/intersects.hpp"
#include "../core/local_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/small_vector.hpp"
#include "../exact/triangle_segment_intersection.hpp"
#include "../exact/vertex.hpp"
#include "../exact/vertex_converter.hpp"
#include "../spatial/policy/tree.hpp"
#include "../spatial/search.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./exact/coplanar_primitives.hpp"
#include "./exact/crossing_edges_vs_face.hpp"
#include "./exact/dedup_vertex_points.hpp"
#include "./exact/duplicate_tagged_intersection.hpp"
#include "./exact/tagged_intersections.hpp"
#include "./exact/vertex_face.hpp"
#include "./intersect_mode.hpp"
#include "tbb/task_group.h"

namespace tf {

/// Exact intersection data between two polygon meshes.
///
/// Stores intersection points as int32 coordinates computed via exact
/// arithmetic (SoS or primitives). No float round-trip — int32 points are the
/// primary representation. Supports convex polygon faces of any size.
template <typename Index, typename RealType, typename Int = tf::exact::int32>
class intersections_between_polygons
    : public tf::intersect::tagged_intersections<Index, Int, 3> {
  static constexpr std::size_t Dims = 3;
  using base_t = tf::intersect::tagged_intersections<Index, Int, Dims>;
  using intersection_t = tf::intersect::tagged_intersection<Index>;

public:
  auto converter() const -> const auto & { return _converter; }

  template <typename Policy0, typename Policy1>
  auto build(const tf::polygons<Policy0> &form0,
             const tf::polygons<Policy1> &form1,
             tf::intersect_mode mode = tf::intersect_mode::primitives) {
    static_assert(tf::has_tree_policy<Policy0>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_tree_policy<Policy1>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_manifold_edge_link_policy<Policy0>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_manifold_edge_link_policy<Policy1>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_face_membership_policy<Policy0>,
                  "Use polygons | tf::tag(face_membership)");
    static_assert(tf::has_face_membership_policy<Policy1>,
                  "Use polygons | tf::tag(face_membership)");

    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(form0, form1);

    if (mode & tf::intersect_mode::primitives)
      build_primitives(form0, form1);
    else
      build_sos(form0, form1);
  }

  template <typename Iterator, std::size_t N>
  auto build(tf::range<Iterator, N> forms,
             tf::intersect_mode mode = tf::intersect_mode::primitives) {
    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(forms);

    if (mode & tf::intersect_mode::primitives)
      build_primitives(forms);
    else
      build_sos(forms);
  }

private:
  template <typename Policy0, typename Policy1>
  auto build_sos(const tf::polygons<Policy0> &form0,
                 const tf::polygons<Policy1> &form1) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    intersect_pair(form0, form1, 0, 1, l_intersections, l_points, l_face_verts);

    finalize_sos_build(l_intersections, l_points,
                       tf::intersect::make_duplicator(form0, form1), Index(2));
  }

  template <typename Policy0, typename Policy1>
  auto build_primitives(const tf::polygons<Policy0> &form0,
                        const tf::polygons<Policy1> &form1) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts0;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts1;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    primitives_intersect_pair(form0, form1, 0, 1, l_intersections, l_points,
                              l_face_verts0, l_face_verts1);

    auto points = l_points.to_buffer();
    if (points.size() == 0)
      return base_t::finalize(Index(2));

    auto raw = merge_local_intersections(l_intersections);
    tf::intersect::dedup_vertex_points(
        raw, points, [&](Index tag, Index object, Index local_id) -> Index {
          if (tag == 0)
            return Index(form0.faces()[object][local_id]);
          return Index(form1.faces()[object][local_id]);
        });

    base_t::_intersection_points = std::move(points);
    tf::generic_generate(raw, base_t::_intersections,
                         tf::intersect::make_duplicator(form0, form1));
    base_t::finalize(Index(2));
  }

  template <typename Iterator, std::size_t N>
  auto build_sos(tf::range<Iterator, N> forms) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    auto n = Index(forms.size());
    tbb::task_group tg;
    for (Index i = 0; i < n; ++i)
      for (Index j = i + 1; j < n; ++j)
        tg.run([&, i, j]() {
          intersect_pair(forms[i], forms[j], i, j, l_intersections, l_points,
                         l_face_verts);
        });
    tg.wait();

    finalize_sos_build(l_intersections, l_points,
                       tf::intersect::make_duplicator(forms), n);
  }

  template <typename Iterator, std::size_t N>
  auto build_primitives(tf::range<Iterator, N> forms) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts0;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts1;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    auto n = Index(forms.size());
    tbb::task_group tg;
    for (Index i = 0; i < n; ++i)
      for (Index j = i + 1; j < n; ++j)
        tg.run([&, i, j]() {
          primitives_intersect_pair(forms[i], forms[j], int(i), int(j),
                                    l_intersections, l_points, l_face_verts0,
                                    l_face_verts1);
        });
    tg.wait();

    auto points = l_points.to_buffer();
    if (points.size() == 0)
      return base_t::finalize(n);

    auto raw = merge_local_intersections(l_intersections);
    dedup_vertex_points(raw, points,
                        [&](Index tag, Index object, Index local_id) -> Index {
                          return Index(forms[tag].faces()[object][local_id]);
                        });

    base_t::_intersection_points = std::move(points);
    tf::generic_generate(raw, base_t::_intersections,
                         tf::intersect::make_duplicator(forms));
    base_t::finalize(n);
  }

  /// Run pairwise edge-vs-face search between two tagged polygons.
  template <typename Policy0, typename Policy1>
  auto intersect_pair(
      const tf::polygons<Policy0> &form0, const tf::polygons<Policy1> &form1,
      int tag0, int tag1, tf::local_buffer<intersection_t> &l_intersections,
      tf::local_buffer<tf::exact::pt3<Int>> &l_points,
      tf::local_buffer<tf::exact::vertex<Index, Int>> &l_face_verts) {
    auto &conv = _converter;
    tf::search(
        form0, form1,
        [&](const auto &bv0, const auto &bv1) {
          return tf::intersects(
              tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
              tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)));
        },
        [&](const auto &poly0, const auto &poly1) {
          auto &ints = *l_intersections;
          auto &pts = *l_points;
          auto &face_buf = *l_face_verts;
          auto &&mel0 = form0.manifold_edge_link();
          auto &&mel1 = form1.manifold_edge_link();
          auto id0 = Index(poly0.id()), id1 = Index(poly1.id());
          auto erep0 = [&](std::size_t j) {
            return mel0[id0][j].is_representative(id0);
          };
          auto erep1 = [&](std::size_t j) {
            return mel1[id1][j].is_representative(id1);
          };
          edges_vs_face_sos(poly0, poly1, tag0, tag1, erep0, _converter,
                            face_buf, ints, pts);
          edges_vs_face_sos(poly1, poly0, tag1, tag0, erep1, _converter,
                            face_buf, ints, pts);
        });
  }

  /// Combine thread-local buffers, duplicate intersections, finalize.
  template <typename Duplicator>
  auto finalize_sos_build(tf::local_buffer<intersection_t> &l_intersections,
                          tf::local_buffer<tf::exact::pt3<Int>> &l_points,
                          Duplicator &&duplicator, Index n_tags) {
    auto points = l_points.to_buffer();
    if (points.size() == 0)
      return base_t::finalize(n_tags);

    auto raw_intersections = [&]() {
      tf::buffer<intersection_t> out;
      out.allocate(l_intersections.total_size());
      std::size_t offset = 0;
      auto it = out.begin();
      for (const auto &v : l_intersections.buffers()) {
        for (auto e : v) {
          e.id += offset;
          *it++ = e;
        }
        offset += v.size();
      }
      return out;
    }();

    base_t::_intersection_points = std::move(points);

    tf::generic_generate(raw_intersections, base_t::_intersections,
                         std::forward<Duplicator>(duplicator));

    base_t::finalize(n_tags);
  }

  /// Merge thread-local intersection buffers, adjusting point IDs by
  /// per-thread point offsets (1:1 correspondence with point buffers).
  static auto
  merge_local_intersections(tf::local_buffer<intersection_t> &l_ints)
      -> tf::buffer<intersection_t> {
    tf::buffer<intersection_t> out;
    out.allocate(l_ints.total_size());
    std::size_t offset = 0;
    auto it = out.begin();
    for (const auto &v : l_ints.buffers()) {
      for (auto e : v) {
        e.id += offset;
        *it++ = e;
      }
      offset += v.size();
    }
    return out;
  }

  /// Convert a polygon's vertices to exact vertices into the provided buffer.
  template <typename Poly, typename Conv>
  static auto convert_face(const Poly &poly, int tag, const Conv &conv,
                           tf::buffer<tf::exact::vertex<Index, Int>> &out) {
    auto n = poly.size();
    out.reallocate(n);
    for (decltype(n) k = 0; k < n; ++k)
      out[k] = conv(tag, Index(poly.indices()[k]), poly[k]);
  }

  /// Test one edge against a convex face via fan triangulation.
  static auto
  edge_vs_convex_face_sos(const tf::buffer<tf::exact::vertex<Index, Int>> &face,
                          const tf::exact::vertex<Index, Int> &v0,
                          const tf::exact::vertex<Index, Int> &v1)
      -> std::optional<tf::exact::pt3<Int>> {
    auto n = face.size();
    for (decltype(n) t = 0; t + 2 < n; ++t) {
      if (auto pt = tf::exact::triangle_segment_intersect_point_sos(
              std::array<tf::exact::vertex<Index, Int>, 5>{
                  face[0], face[t + 1], face[t + 2], v0, v1}))
        return pt;
    }
    return std::nullopt;
  }

  /// Test all representative edges of `edge_poly` against the face of
  /// `face_poly`. `edge_tag`/`face_tag` are the mesh tags.
  template <typename EdgePoly, typename FacePoly, typename EdgeIsRep,
            typename Conv, typename Ints, typename Pts>
  static auto
  edges_vs_face_sos(const EdgePoly &edge_poly, const FacePoly &face_poly,
                    int edge_tag, int face_tag, const EdgeIsRep &edge_is_rep,
                    const Conv &conv,
                    tf::buffer<tf::exact::vertex<Index, Int>> &face_buf,
                    Ints &ints, Pts &pts) {
    auto edge_id = Index(edge_poly.id());
    auto face_id = Index(face_poly.id());

    convert_face(face_poly, face_tag, conv, face_buf);

    auto n = edge_poly.size();
    for (decltype(n) j = 0; j < n; ++j) {
      if (!edge_is_rep(j))
        continue;
      auto next_j = tf::circular_increment(j, n);
      auto v0 = conv(edge_tag, Index(edge_poly.indices()[j]), edge_poly[j]);
      auto v1 =
          conv(edge_tag, Index(edge_poly.indices()[next_j]), edge_poly[next_j]);

      if (auto pt = edge_vs_convex_face_sos(face_buf, v0, v1)) {
        Index id = pts.size();
        pts.push_back(*pt);
        auto edge_target = tf::topo_id<Index>{Index(j), tf::topo_type::edge};
        auto face_target = tf::topo_id<Index>{face_id, tf::topo_type::face};
        ints.push_back({Index(edge_tag), Index(face_tag), edge_id, face_id,
                        edge_target, face_target, id});
      }
    }
  }

  /// Run primitives classification for one overlapping polygon pair.
  /// Always runs crossing_edges_vs_face (EF/EE/VE for plane-crossing edges).
  /// When any vertex has sign==0, also runs coplanar_primitives (VV,
  /// coplanar VE/EE) and vertex_face (VF).
  template <typename Poly0, typename Poly1, typename MEL0, typename MEL1,
            typename FM0, typename FM1, typename Conv, typename Ints,
            typename Pts>
  static auto
  primitives_polygon_pair(const Poly0 &poly0, const Poly1 &poly1, int tag0,
                          int tag1, const MEL0 &mel0, const MEL1 &mel1,
                          const FM0 &fm0, const FM1 &fm1, const Conv &conv,
                          tf::buffer<tf::exact::vertex<Index, Int>> &face_buf0,
                          tf::buffer<tf::exact::vertex<Index, Int>> &face_buf1,
                          Ints &ints, Pts &pts) {
    auto face0_id = Index(poly0.id());
    auto face1_id = Index(poly1.id());

    convert_face(poly0, tag0, conv, face_buf0);
    convert_face(poly1, tag1, conv, face_buf1);

    auto n0 = face_buf0.size();
    auto n1 = face_buf1.size();

    auto plane0 = compute_face_plane(face_buf0);
    auto plane1 = compute_face_plane(face_buf1);

    // Compute orient3d_sign for all vertices vs opposite face plane.
    // sign_mask bits: 0 = has_negative, 1 = has_zero, 2 = has_positive.
    constexpr int has_negative = 1 << 0;
    constexpr int has_zero = 1 << 1;
    constexpr int has_positive = 1 << 2;
    constexpr int has_crossing = has_negative | has_positive;

    tf::small_vector<int, 16> signs0, signs1;
    signs0.resize(n0);
    signs1.resize(n1);
    int mask0 = 0, mask1 = 0;
    if (plane1.valid)
      for (decltype(n0) i = 0; i < n0; ++i) {
        signs0[i] = orient3d_sign(std::array<tf::exact::vertex<Index, Int>, 4>{
            face_buf1[plane1.i0], face_buf1[plane1.i1], face_buf1[plane1.i2],
            face_buf0[i]});
        mask0 |= 1 << (signs0[i] + 1);
      }
    if (plane0.valid)
      for (decltype(n1) j = 0; j < n1; ++j) {
        signs1[j] = orient3d_sign(std::array<tf::exact::vertex<Index, Int>, 4>{
            face_buf0[plane0.i0], face_buf0[plane0.i1], face_buf0[plane0.i2],
            face_buf1[j]});
        mask1 |= 1 << (signs1[j] + 1);
      }

    // Both faces strictly on one side of the other's plane → no intersection.
    int combined = mask0 | mask1;
    if (!(combined & has_zero) && (mask0 & has_crossing) != has_crossing &&
        (mask1 & has_crossing) != has_crossing)
      return;

    bool any_zero = combined & has_zero;
    auto is_rep0 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = Index(poly0.indices()[i]);
      return {Index(fm0[v_global].front()) == face0_id,
              mel0[face0_id][i].is_representative(face0_id)};
    };
    auto is_rep1 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = Index(poly1.indices()[i]);
      return {Index(fm1[v_global].front()) == face1_id,
              mel1[face1_id][i].is_representative(face1_id)};
    };

    // EF / crossing-EE / crossing-VE (both directions)
    bool both_crossing = (mask0 & has_crossing) == has_crossing &&
                         (mask1 & has_crossing) == has_crossing;
    if ((mask0 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face(face_buf0, n0, face_buf1, n1, signs0,
                                        tag0, tag1, face0_id, face1_id, is_rep0,
                                        is_rep1, ints, pts, both_crossing);
    if ((mask1 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face(face_buf1, n1, face_buf0, n0, signs1,
                                        tag1, tag0, face1_id, face0_id, is_rep1,
                                        is_rep0, ints, pts, both_crossing);
    if (!any_zero)
      return;

    // VV / coplanar-VE / coplanar-EE
    tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                   tag0, tag1, face0_id, face1_id, is_rep0,
                                   is_rep1, plane0, plane1, ints, pts);

    // VF (both directions)
    auto vrep0 = [&](std::size_t i) {
      return Index(fm0[Index(poly0.indices()[i])].front()) == face0_id;
    };
    auto vrep1 = [&](std::size_t j) {
      return Index(fm1[Index(poly1.indices()[j])].front()) == face1_id;
    };
    tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, tag0, tag1,
                           face0_id, face1_id, vrep0, plane1, ints, pts);
    tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, tag1, tag0,
                           face1_id, face0_id, vrep1, plane0, ints, pts);
  }

  /// Run primitives pairwise intersection between two tagged polygons.
  template <typename Policy0, typename Policy1>
  auto primitives_intersect_pair(
      const tf::polygons<Policy0> &form0, const tf::polygons<Policy1> &form1,
      int tag0, int tag1, tf::local_buffer<intersection_t> &l_ints,
      tf::local_buffer<tf::exact::pt3<Int>> &l_pts,
      tf::local_buffer<tf::exact::vertex<Index, Int>> &l_fb0,
      tf::local_buffer<tf::exact::vertex<Index, Int>> &l_fb1) {
    auto &conv = _converter;
    tf::search(
        form0, form1,
        [&](const auto &bv0, const auto &bv1) {
          return tf::intersects(
              tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
              tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)));
        },
        [&](const auto &poly0, const auto &poly1) {
          primitives_polygon_pair(
              poly0, poly1, tag0, tag1, form0.manifold_edge_link(),
              form1.manifold_edge_link(), form0.face_membership(),
              form1.face_membership(), _converter, *l_fb0, *l_fb1, *l_ints,
              *l_pts);
        });
  }

  tf::exact::vertex_converter<Int, RealType, Dims> _converter;
};

} // namespace tf

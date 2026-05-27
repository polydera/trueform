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
#include "../core/inflated_aabb.hpp"
#include "../core/intersects.hpp"
#include "../core/local_buffer.hpp"
#include "../core/small_vector.hpp"
#include "../exact/triangle_segment_intersection.hpp"
#include "../exact/vertex.hpp"
#include "../exact/vertex_converter.hpp"
#include "../spatial/search_self.hpp"
#include "../topology/policy/face_membership.hpp"
#include "../topology/policy/manifold_edge_link.hpp"
#include "./exact/coplanar_primitives.hpp"
#include "./exact/crossing_edges_vs_face.hpp"
#include "./exact/dedup_coincident_points.hpp"
#include "./exact/duplicate_tagged_intersection.hpp"
#include "./exact/face_plane_info.hpp"
#include "./exact/make_kernel.hpp"
#include "./exact/predicate_kernel.hpp"
#include "./exact/tagged_intersections.hpp"
#include "./exact/vertex_face.hpp"
#include "./intersect_config.hpp"

namespace tf {

template <typename Index, typename RealType, typename Int = tf::exact::int32>
class intersections_within_polygons
    : public tf::intersect::tagged_intersections<Index, Int, 3> {
  static constexpr std::size_t Dims = 3;
  using base_t = tf::intersect::tagged_intersections<Index, Int, Dims>;
  using intersection_t = tf::intersect::tagged_intersection<Index>;

public:
  auto converter() const -> const auto & { return _converter; }

  template <typename Policy>
  auto build(const tf::polygons<Policy> &form,
             tf::intersect_config config = {}) {
    static_assert(tf::has_tree_policy<Policy>, "Use polygons | tf::tag(tree)");
    static_assert(tf::has_manifold_edge_link_policy<Policy>,
                  "Use polygons | tf::tag(manifold_edge_link)");
    static_assert(tf::has_face_membership_policy<Policy>,
                  "Use polygons | tf::tag(face_membership)");

    base_t::clear();
    _converter = tf::exact::make_vertex_converter<Int, RealType>(form);

    if (config.mode & tf::intersect_mode::primitives)
      build_primitives(form, tf::exact::make_kernel(_converter, config.tolerance));
    else
      build_sos(form);
  }

private:

  template <typename Indices0, typename Indices1>
  static auto compute_shared_masks(const Indices0 &ids0, const Indices1 &ids1,
                                   tf::buffer<bool> &s0, tf::buffer<bool> &s1) {
    s0.allocate(ids0.size());
    s1.allocate(ids1.size());
    std::fill(s0.begin(), s0.end(), false);
    std::fill(s1.begin(), s1.end(), false);
    for (decltype(ids0.size()) i = 0; i < ids0.size(); ++i)
      for (decltype(ids1.size()) j = 0; j < ids1.size(); ++j)
        if (Index(ids0[i]) == Index(ids1[j]))
          s0[i] = s1[j] = true;
  }

  static auto has_shared_edge(const tf::buffer<bool> &mask) -> bool {
    std::size_t n = mask.size();
    for (std::size_t i = 0; i < n; ++i)
      if (mask[i] && mask[tf::circular_increment(i, n)])
        return true;
    return false;
  }

  template <typename Poly, typename Conv>
  static auto convert_face(const Poly &poly, int tag, const Conv &conv,
                           tf::buffer<tf::exact::vertex<Index, Int>> &out) {
    auto n = poly.size();
    out.reallocate(n);
    for (decltype(n) k = 0; k < n; ++k)
      out[k] = conv(tag, poly.indices()[k], poly[k]);
  }

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

  template <typename EdgePoly, typename FacePoly, typename EdgeIsRep,
            typename Conv, typename Ints, typename Pts>
  static auto
  edges_vs_face_sos(const EdgePoly &edge_poly, const FacePoly &face_poly,
                    int edge_tag, int face_tag, const EdgeIsRep &edge_is_rep,
                    const Conv &conv,
                    tf::buffer<tf::exact::vertex<Index, Int>> &face_buf,
                    Ints &ints, Pts &pts, const tf::buffer<bool> &shared) {
    auto edge_id = Index(edge_poly.id());
    auto face_id = Index(face_poly.id());

    convert_face(face_poly, face_tag, conv, face_buf);

    auto n = edge_poly.size();
    for (decltype(n) j = 0; j < n; ++j) {
      auto next_j = tf::circular_increment(j, n);
      if (shared[j] || shared[next_j])
        continue;
      if (!edge_is_rep(j))
        continue;
      auto v0 = conv(edge_tag, edge_poly.indices()[j], edge_poly[j]);
      auto v1 = conv(edge_tag, edge_poly.indices()[next_j], edge_poly[next_j]);

      if (auto pt = edge_vs_convex_face_sos(face_buf, v0, v1)) {
        Index id = pts.size();
        pts.push_back(*pt);
        ints.push_back({Index(edge_tag),
                        Index(face_tag),
                        edge_id,
                        face_id,
                        {Index(j), tf::topo_type::edge},
                        {face_id, tf::topo_type::face},
                        id});
      }
    }
  }

  template <typename Poly0, typename Poly1, typename MEL, typename FM,
            typename Conv, typename Ints, typename Pts>
  static auto
  primitives_polygon_pair(const Poly0 &poly0, const Poly1 &poly1, int tag0,
                          int tag1, const MEL &mel, const FM &fm,
                          const Conv &conv,
                          tf::buffer<tf::exact::vertex<Index, Int>> &face_buf0,
                          tf::buffer<tf::exact::vertex<Index, Int>> &face_buf1,
                          Ints &ints, Pts &pts, const tf::buffer<bool> &shared0,
                          const tf::buffer<bool> &shared1,
                          const tf::exact::predicate_kernel<Int> &kernel = {}) {
    auto face0_id = Index(poly0.id());
    auto face1_id = Index(poly1.id());

    convert_face(poly0, tag0, conv, face_buf0);
    convert_face(poly1, tag1, conv, face_buf1);

    auto n0 = face_buf0.size();
    auto n1 = face_buf1.size();

    auto plane0 = tf::exact::compute_face_plane(face_buf0);
    auto plane1 = tf::exact::compute_face_plane(face_buf1);

    constexpr int has_negative = 1 << 0;
    constexpr int has_zero = 1 << 1;
    constexpr int has_positive = 1 << 2;
    constexpr int has_crossing = has_negative | has_positive;

    tf::small_vector<int, 16> signs0, signs1;
    signs0.resize(n0);
    signs1.resize(n1);
    int mask0 = 0, mask1 = 0;
    int mask0_ns = 0, mask1_ns = 0;
    if (plane1.valid)
      for (decltype(n0) i = 0; i < n0; ++i) {
        signs0[i] = orient3d_sign(std::array<tf::exact::vertex<Index, Int>, 4>{
            face_buf1[plane1.i0], face_buf1[plane1.i1], face_buf1[plane1.i2],
            face_buf0[i]});
        auto bit = 1 << (signs0[i] + 1);
        mask0 |= bit;
        if (!shared0[i])
          mask0_ns |= bit;
      }
    if (plane0.valid)
      for (decltype(n1) j = 0; j < n1; ++j) {
        signs1[j] = orient3d_sign(std::array<tf::exact::vertex<Index, Int>, 4>{
            face_buf0[plane0.i0], face_buf0[plane0.i1], face_buf0[plane0.i2],
            face_buf1[j]});
        auto bit = 1 << (signs1[j] + 1);
        mask1 |= bit;
        if (!shared1[j])
          mask1_ns |= bit;
      }

    int combined = mask0 | mask1;
    if (!(combined & has_zero) && (mask0 & has_crossing) != has_crossing &&
        (mask1 & has_crossing) != has_crossing)
      return;

    int combined_ns = mask0_ns | mask1_ns;
    if (!(combined_ns & has_zero) &&
        (mask0_ns & has_crossing) != has_crossing &&
        (mask1_ns & has_crossing) != has_crossing)
      return;

    bool any_zero = combined & has_zero;
    auto is_rep0 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = poly0.indices()[i];
      return {Index(fm[v_global].front()) == face0_id,
              mel[face0_id][i].is_representative(face0_id)};
    };
    auto is_rep1 = [&](std::size_t i) -> std::pair<bool, bool> {
      auto v_global = poly1.indices()[i];
      return {Index(fm[v_global].front()) == face1_id,
              mel[face1_id][i].is_representative(face1_id)};
    };

    auto shared0_f = [&shared0](std::size_t i) { return shared0[i]; };
    auto shared1_f = [&shared1](std::size_t j) { return shared1[j]; };

    bool both_crossing = (mask0 & has_crossing) == has_crossing &&
                         (mask1 & has_crossing) == has_crossing;
    if ((mask0 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face_self(
          face_buf0, n0, face_buf1, n1, signs0, tag0, tag1, face0_id, face1_id,
          is_rep0, is_rep1, ints, pts, both_crossing, kernel);
    if ((mask1 & has_crossing) == has_crossing)
      tf::exact::crossing_edges_vs_face_self(
          face_buf1, n1, face_buf0, n0, signs1, tag1, tag0, face1_id, face0_id,
          is_rep1, is_rep0, ints, pts, both_crossing, kernel);
    if (!any_zero)
      return;

    tf::exact::coplanar_primitives(face_buf0, n0, face_buf1, n1, signs0, signs1,
                                   tag0, tag1, face0_id, face1_id, is_rep0,
                                   is_rep1, plane0, plane1, shared0_f,
                                   shared1_f, ints, pts, kernel);

    auto vrep0 = [&](std::size_t i) {
      return Index(fm[poly0.indices()[i]].front()) == face0_id;
    };
    auto vrep1 = [&](std::size_t j) {
      return Index(fm[poly1.indices()[j]].front()) == face1_id;
    };
    tf::exact::vertex_face(face_buf0, n0, face_buf1, n1, signs0, tag0, tag1,
                           face0_id, face1_id, vrep0, shared0_f, plane1, ints,
                           pts, kernel);
    tf::exact::vertex_face(face_buf1, n1, face_buf0, n0, signs1, tag1, tag0,
                           face1_id, face0_id, vrep1, shared1_f, plane0, ints,
                           pts, kernel);
  }

  template <typename Duplicator>
  auto finalize_sos_build(tf::local_buffer<intersection_t> &l_intersections,
                          tf::local_buffer<tf::exact::pt3<Int>> &l_points,
                          Duplicator &&duplicator) {
    auto points = l_points.to_buffer();
    if (points.size() == 0)
      return base_t::finalize(1);

    tf::buffer<intersection_t> raw;
    raw.allocate(l_intersections.total_size());
    std::size_t offset = 0;
    auto it = raw.begin();
    for (const auto &v : l_intersections.buffers()) {
      for (auto e : v) {
        e.id += offset;
        *it++ = e;
      }
      offset += v.size();
    }

    tf::intersect::dedup_coincident_points(raw, points);

    base_t::_intersection_points = std::move(points);
    tf::generic_generate(raw, base_t::_intersections,
                         std::forward<Duplicator>(duplicator));
    base_t::finalize(1);
  }

  template <typename Policy> auto build_sos(const tf::polygons<Policy> &form) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts;
    tf::local_buffer<bool> l_shared0;
    tf::local_buffer<bool> l_shared1;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    auto &conv = _converter;
    tf::search_self(
        form,
        [&](const auto &bv0, const auto &bv1) {
          return tf::intersects(
              tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
              tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)));
        },
        [&](const auto &poly0, const auto &poly1) {
          auto &shared0 = *l_shared0;
          auto &shared1 = *l_shared1;
          compute_shared_masks(poly0.indices(), poly1.indices(), shared0,
                               shared1);

          if (has_shared_edge(shared0))
            return;

          auto &ints = *l_intersections;
          auto &pts = *l_points;
          auto &face_buf = *l_face_verts;
          auto &&mel = form.manifold_edge_link();
          auto id0 = Index(poly0.id()), id1 = Index(poly1.id());

          auto erep0 = [&](std::size_t j) {
            return mel[id0][j].is_representative(id0);
          };
          auto erep1 = [&](std::size_t j) {
            return mel[id1][j].is_representative(id1);
          };

          edges_vs_face_sos(poly0, poly1, 0, 0, erep0, _converter, face_buf,
                            ints, pts, shared0);
          edges_vs_face_sos(poly1, poly0, 0, 0, erep1, _converter, face_buf,
                            ints, pts, shared1);
        });

    auto self_duplicator = [&](auto rec, auto &buffer) {
      tf::intersect::duplicate_intersection_self(
          form.faces(), form.face_membership(), form.manifold_edge_link(), rec,
          buffer);
    };

    finalize_sos_build(l_intersections, l_points, self_duplicator);
  }

  template <typename Policy>
  auto build_primitives(const tf::polygons<Policy> &form,
                        const tf::exact::predicate_kernel<Int> &kernel = {}) {
    tf::local_buffer<intersection_t> l_intersections;
    tf::local_buffer<tf::exact::pt3<Int>> l_points;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts0;
    tf::local_buffer<tf::exact::vertex<Index, Int>> l_face_verts1;
    tf::local_buffer<bool> l_shared0;
    tf::local_buffer<bool> l_shared1;
    l_intersections.reserve_all(1000);
    l_points.reserve_all(1000);

    auto &conv = _converter;
    auto pad = kernel.tolerance_int();
    tf::search_self(
        form,
        [&](const auto &bv0, const auto &bv1) {
          return tf::intersects(
              tf::make_aabb(conv.convert(bv0.min), conv.convert(bv0.max)),
              tf::inflated_aabb(
                  tf::make_aabb(conv.convert(bv1.min), conv.convert(bv1.max)),
                  pad));
        },
        [&](const auto &poly0, const auto &poly1) {
          auto &shared0 = *l_shared0;
          auto &shared1 = *l_shared1;
          compute_shared_masks(poly0.indices(), poly1.indices(), shared0,
                               shared1);

          primitives_polygon_pair(
              poly0, poly1, 0, 0, form.manifold_edge_link(),
              form.face_membership(), _converter, *l_face_verts0,
              *l_face_verts1, *l_intersections, *l_points, shared0, shared1,
              kernel);
        });

    auto points = l_points.to_buffer();
    if (points.size() == 0)
      return base_t::finalize(Index(1));

    auto raw = merge_local_intersections(l_intersections);
    tf::intersect::dedup_coincident_points(raw, points);

    base_t::_intersection_points = std::move(points);

    auto self_duplicator = [&](auto rec, auto &buffer) {
      tf::intersect::duplicate_intersection_self(
          form.faces(), form.face_membership(), form.manifold_edge_link(), rec,
          buffer);
    };
    tf::generic_generate(raw, base_t::_intersections, self_duplicator);
    base_t::finalize(Index(1));
  }

  // Reuse from intersections_between_polygons — these are static/shared
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

  tf::exact::vertex_converter<Int, RealType, Dims> _converter;
};

} // namespace tf
